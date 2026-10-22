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

from tico.quantization.passes.concat_qparam import is_width_direction_concat
from tico.quantization.passes.mx_qparam import remap_mx_qparam
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils import logging
from tico.utils.mx.dtypes import is_mx_dtype
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import CatArgs, PermuteArgs, ReshapeArgs


@trace_graph_diff_on_pass
class PropagateQParamBackward(PassBase):
    """
    This pass propagates quantization parameters backward.

    BEFORE)

    node -> reshape (with meta[QPARAM_KEY])

    AFTER)

    node (with meta[QPARAM_KEY]) -> reshape (with meta[QPARAM_KEY])
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph

        def _propagate_qparam_if_possible(src: torch.fx.Node, dst: torch.fx.Node):
            if QPARAM_KEY not in src.meta:
                return

            src_qparam = src.meta[QPARAM_KEY]
            if is_mx_dtype(src_qparam.dtype):
                # A producer's established domain belongs to all its users. In
                # particular, never rewrite an explicit Quantize destination
                # using the domain of a later view or requantization boundary.
                if QPARAM_KEY in dst.meta or len(dst.users) != 1:
                    return
                schema = getattr(dst.target, "_schema", None)
                if getattr(schema, "name", None) in {
                    "circle_custom::quantize_mx",
                    "quantized_decomposed::quantize_per_tensor",
                    "quantized_decomposed::quantize_per_channel",
                }:
                    return
                mapped = remap_mx_qparam(src_qparam, src, backward=True)
                assert mapped is not None
                dst.meta[QPARAM_KEY] = mapped
                logger.debug(f"{src.name}'s MX quantparam is remapped to {dst.name}.")
                return

            if (
                QPARAM_KEY in dst.meta
                and src_qparam.dtype != dst.meta[QPARAM_KEY].dtype
            ):
                return

            dst.meta[QPARAM_KEY] = copy.deepcopy(src_qparam)

            logger.debug(f"{src.name}'s quantparam is propagated to {dst.name}.")

        # Do reverse-order traversal for backward propagation
        for node in reversed(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.cat.default:
                concat_args = CatArgs(*node.args, **node.kwargs)
                concat_inputs = concat_args.tensors

                node_qparam = node.meta.get(QPARAM_KEY)
                if is_width_direction_concat(node, concat_args) and (
                    node_qparam is None or not is_mx_dtype(node_qparam.dtype)
                ):
                    continue

                for concat_input in concat_inputs:
                    _propagate_qparam_if_possible(node, concat_input)
            elif node.target == torch.ops.aten.reshape.default:
                args = ReshapeArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(node, args.input)
            elif node.target == torch.ops.aten.permute.default:
                permute_args = PermuteArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(node, permute_args.input)
            # TODO Support more ops.

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
