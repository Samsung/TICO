# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

"""Serialize the internal ResizeBilinear operator as a Circle builtin."""

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx

import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.circle_mapping import circle_legalize_dtype_to
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import ResizeBilinearArgs


@register_node_visitor
class ResizeBilinearVisitor(NodeVisitor):
    """Serialize an NHWC custom resize node as Circle RESIZE_BILINEAR."""

    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.resize_bilinear.default
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        """Initialize the visitor with shared opcode and graph state."""
        super().__init__(op_codes, graph)

    def define_node(self, node: torch.fx.Node) -> circle.Operator.OperatorT:
        """Create one Circle RESIZE_BILINEAR operator."""
        args = ResizeBilinearArgs(
            *node.args,
            **node.kwargs,  # type: ignore[arg-type]
        )
        size_i32 = circle_legalize_dtype_to(args.size, dtype=torch.int32)
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.RESIZE_BILINEAR,
            self._op_codes,
        )
        operator = create_builtin_operator(
            self.graph,
            op_index,
            [args.input, size_i32],
            [node],
        )
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.ResizeBilinearOptions
        )
        options = circle.ResizeBilinearOptions.ResizeBilinearOptionsT()
        options.alignCorners = args.align_corners
        options.halfPixelCenters = args.half_pixel_centers
        operator.builtinOptions = options
        return operator
