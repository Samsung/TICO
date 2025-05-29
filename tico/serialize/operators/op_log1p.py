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
from tico.serialize.circle_mapping import (
    extract_circle_dtype,
    extract_shape,
    extract_torch_dtype,
)
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import Log1pArgs


@register_node_visitor
class Log1pVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.log1p.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_add_node(self, inputs: List, outputs: List) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.ADD, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.AddOptions
        option = circle.AddOptions.AddOptionsT()
        option.fusedActivationFunction = (
            circle.ActivationFunctionType.ActivationFunctionType.NONE
        )
        option.potScaleInt16 = False
        operator.builtinOptions = option

        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = Log1pArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input

        input_shape = list(extract_shape(input))
        dst_dtype_circle = extract_circle_dtype(input)
        add_tensor: circle.Tensor.TensorT = self.graph.add_tensor_from_scratch(
            prefix=f"{input.name}_add",
            shape=input_shape,
            dtype=dst_dtype_circle,
            source_node=node,
        )
        const_one = torch.tensor([1]).to(extract_torch_dtype(input))

        add_node = self.define_add_node([input, const_one], [add_tensor])
        self.graph.add_operator(add_node)

        inputs = [add_tensor]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.LOG, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        return operator
