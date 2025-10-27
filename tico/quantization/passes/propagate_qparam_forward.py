# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import copy

import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import (
    CatArgs,
    NegArgs,
    PermuteArgs,
    ReshapeArgs,
    SliceArgs,
)


@trace_graph_diff_on_pass
class PropagateQParamForward(PassBase):
    """
    A pass propagates quantization parameters through operations that do not alter them.

    This pass identifies and propagates quantization parameters through operations that
     do not change their values, such as `permute`, `reshape`, `transpose`, `view` and
    similar tensor transformations.

    By ensuring that quantization parameters remain consistent across such operations,
    this pass helps maintain correctness in quantization-aware representations.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        def _propagate_qparam_if_possible(src: torch.fx.Node, dst: torch.fx.Node):
            if QPARAM_KEY not in src.meta:
                return

            if (
                QPARAM_KEY in dst.meta
                and src.meta[QPARAM_KEY].dtype != dst.meta[QPARAM_KEY].dtype
            ):
                return

            dst.meta[QPARAM_KEY] = copy.deepcopy(src.meta[QPARAM_KEY])

            logger.debug(f"{src.name}'s quantparam is propagated to {dst.name}.")

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.permute.default:
                permute_args = PermuteArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(permute_args.input, node)
            elif node.target == torch.ops.aten.reshape.default:
                reshape_args = ReshapeArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(reshape_args.input, node)
            elif node.target == torch.ops.aten.slice.Tensor:
                slice_args = SliceArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(slice_args.input, node)
            elif node.target == torch.ops.aten.neg.default:
                neg_args = NegArgs(*node.args, **node.kwargs)

                if QPARAM_KEY not in neg_args.input.meta:
                    continue
                # Only support int16 for now
                if neg_args.input.meta[QPARAM_KEY].dtype != "int16":
                    continue

                _propagate_qparam_if_possible(neg_args.input, node)

            elif node.target == torch.ops.aten.cat.default:
                concat_args = CatArgs(*node.args, **node.kwargs)
                concat_inputs = concat_args.tensors

                cond = True
                for concat_input in concat_inputs:
                    # Check all inputs have qparam
                    if QPARAM_KEY not in concat_input.meta:
                        cond = False
                        break

                    # Only support int16 for now
                    if concat_input.meta[QPARAM_KEY].dtype != "int16":
                        cond = False
                        break

                    if concat_input.meta[QPARAM_KEY].scale is None:
                        cond = False
                        break

                    if len(concat_input.meta[QPARAM_KEY].scale) != 1:
                        cond = False
                        break

                if not cond:
                    continue

                # Find max scale node
                max_scale = 0.0
                max_scale_node = None
                for concat_input in concat_inputs:
                    scale = concat_input.meta[QPARAM_KEY].scale[0]
                    if max_scale < scale:
                        max_scale = scale
                        max_scale_node = concat_input

                assert max_scale_node is not None
                _propagate_qparam_if_possible(max_scale_node, node)

            # TODO Support more ops.

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
