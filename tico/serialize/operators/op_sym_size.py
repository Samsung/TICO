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

from tico.serialize.circle_mapping import extract_shape

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import SymSizeIntArgs


@register_node_visitor
class SymSizeVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.sym_size.int]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_shape_node(
        self, inputs: List[torch.fx.Node], outputs: List[circle.Tensor.TensorT]
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SHAPE, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ShapeOptions

        option = circle.ShapeOptions.ShapeOptionsT()
        option.outType = circle.TensorType.TensorType.INT32
        operator.builtinOptions = option
        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = SymSizeIntArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        dim = args.dim

        # shape
        shape_shape = [len(extract_shape(input))]
        shape_tensor = self.graph.add_tensor_from_scratch(
            prefix=f"{input.name}_shape",
            shape=shape_shape,
            dtype=circle.TensorType.TensorType.INT32,
        )
        shape_operator = self.define_shape_node([input], [shape_tensor])
        self.graph.add_operator(shape_operator)

        # gather
        inputs = [shape_tensor, dim]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.GATHER, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.GatherOptions
        option = circle.GatherOptions.GatherOptionsT()
        option.axis = 0
        operator.builtinOptions = option

        return operator
