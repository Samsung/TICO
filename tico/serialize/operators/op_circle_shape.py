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

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
    TorchOperator,
)
from tico.serialize.operators.utils import create_builtin_operator, get_op_index


@register_node_visitor
class CircleShapeVisitor(NodeVisitor):
    """
    Visitor for circle_custom::shape operator.

    This operator extracts the shape of a tensor.
    It's implemented using Circle's SHAPE operator.
    """

    target: List[TorchOperator] = [
        torch.ops.circle_custom.shape,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        # Args: (input)
        input_node = node.args[0]

        # SHAPE operator to get the full shape of input
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SHAPE, self._op_codes
        )

        inputs = [input_node]
        outputs = [node]
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ShapeOptions
        operator.builtinOptions = circle.ShapeOptions.ShapeOptionsT()
        operator.builtinOptions.outType = circle.TensorType.TensorType.INT32

        return operator
