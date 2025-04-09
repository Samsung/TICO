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

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_mapping import circle_legalize_dtype_to
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import ResizeNearestNeighborArgs


@register_node_visitor
class ResizeNearestNeighborVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.resize_nearest_neighbor
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        # Only consider `torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')` case.
        # TODO Support generic algorithm
        args = ResizeNearestNeighborArgs(*node.args, **node.kwargs)
        input = args.input
        size = args.size

        size_i32 = circle_legalize_dtype_to(size, dtype=torch.int32)
        inputs = [input, size_i32]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
            self._op_codes,
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.ResizeNearestNeighborOptions
        )
        option = circle.ResizeNearestNeighborOptions.ResizeNearestNeighborOptionsT()
        # TODO Consider these options
        # If True, the centers of the 4 corner pixels of the input and output tensors are aligned, preserving the values at the corner pixels.
        option.alignCorners = False
        # If True, the pixel centers are assumed to be at (0.5, 0.5). If this parameter is True, then align_corners parameter must be False.
        option.halfPixelCenters = False
        operator.builtinOptions = option

        return operator
