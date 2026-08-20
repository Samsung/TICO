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
import operator

import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import (
    CatArgs,
    ExpandArgs,
    NegArgs,
    PermuteArgs,
    ReshapeArgs,
    SliceArgs,
    SplitWithSizesArgs,
)


_QPARAM_PRESERVING_UNARY_TARGETS = frozenset({torch.ops.aten.constant_pad_nd.default})
_QPARAM_FALLBACK_UNARY_TARGETS = frozenset({torch.ops.circle_custom.maxpool2d.default})


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
            elif node.target in _QPARAM_PRESERVING_UNARY_TARGETS:
                input_ = node.args[0]
                if isinstance(input_, torch.fx.Node):
                    _propagate_qparam_if_possible(input_, node)
            elif node.target in _QPARAM_FALLBACK_UNARY_TARGETS:
                input_ = node.args[0]
                if isinstance(input_, torch.fx.Node) and QPARAM_KEY not in node.meta:
                    _propagate_qparam_if_possible(input_, node)
            elif node.target == torch.ops.aten.reshape.default:
                reshape_args = ReshapeArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(reshape_args.input, node)
            elif node.target == torch.ops.aten.split_with_sizes.default:
                split_args = SplitWithSizesArgs(*node.args, **node.kwargs)
                input_ = split_args.input
                if QPARAM_KEY not in input_.meta:
                    continue

                # A split preserves the input quantization domain when the input
                # uses one per-tensor qparam. Per-channel propagation requires
                # axis-aware qparam slicing and is intentionally not handled here.
                input_qparam = input_.meta[QPARAM_KEY]
                if input_qparam.quantized_dimension is not None:
                    continue

                for user in node.users:
                    if (
                        user.op == "call_function"
                        and user.target == operator.getitem
                        and len(user.args) >= 2
                        and user.args[0] is node
                        and isinstance(user.args[1], int)
                    ):
                        _propagate_qparam_if_possible(input_, user)
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

                if not concat_inputs:
                    continue
                if any(
                    QPARAM_KEY not in concat_input.meta
                    for concat_input in concat_inputs
                ):
                    continue

                first_qparam = concat_inputs[0].meta[QPARAM_KEY]
                if (
                    first_qparam.scale is None
                    or first_qparam.zero_point is None
                    or first_qparam.quantized_dimension is not None
                ):
                    continue
                identical_qparams = all(
                    concat_input.meta[QPARAM_KEY].dtype == first_qparam.dtype
                    and concat_input.meta[QPARAM_KEY].scale == first_qparam.scale
                    and concat_input.meta[QPARAM_KEY].zero_point
                    == first_qparam.zero_point
                    and concat_input.meta[QPARAM_KEY].quantized_dimension
                    == first_qparam.quantized_dimension
                    for concat_input in concat_inputs[1:]
                )
                if identical_qparams:
                    _propagate_qparam_if_possible(concat_inputs[0], node)
                    continue

                # Preserve the existing symmetric INT16 fallback.
                cond = True
                for concat_input in concat_inputs:
                    qparam = concat_input.meta[QPARAM_KEY]
                    if qparam.dtype != "int16":
                        cond = False
                        break
                    if qparam.scale is None or len(qparam.scale) != 1:
                        cond = False
                        break
                if not cond:
                    continue

                max_scale_node = max(
                    concat_inputs,
                    key=lambda concat_input: concat_input.meta[QPARAM_KEY].scale[0],
                )
                _propagate_qparam_if_possible(max_scale_node, node)
            elif node.target == torch.ops.aten.expand.default:
                expand_args = ExpandArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(expand_args.input, node)
            # TODO Support more ops.

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
