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
    circle_legalize_dtype_to,
    extract_torch_dtype,
    to_circle_dtype,
)
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import CumsumArgs


@register_node_visitor
class CumsumVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.cumsum.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = CumsumArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        dim = args.dim

        dim_i32 = circle_legalize_dtype_to(dim, dtype=torch.int32)

        casted_input: torch.fx.Node | circle.Tensor.TensorT = input
        # torch.cumsum doesn't follow input dtype when input dtype is int32.
        # Since circle-interpreter needs a model to have same dtype between input and output,
        #   let's cast the input to torch.int64.
        input_dtype = extract_torch_dtype(input)
        if input_dtype == torch.int32:
            input_tensor: circle.Tensor.TensorT = self.graph.get_tensor(input)
            input_shape: List[int] = input_tensor.shape
            cast_op_index = get_op_index(
                circle.BuiltinOperator.BuiltinOperator.CAST, self._op_codes
            )
            cast_name = f"{input.name}_cast"
            cast_dtype = circle.TensorType.TensorType.INT64
            cast_tensor = self.graph.add_tensor_from_scratch(
                prefix=cast_name,
                dtype=cast_dtype,
                shape=input_shape,
                source_node=node,
            )
            cast_operator = create_builtin_operator(
                self.graph, cast_op_index, [input], [cast_tensor]
            )
            cast_operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.CastOptions
            )
            cast_option = circle.CastOptions.CastOptionsT()
            cast_option.inDataType = to_circle_dtype(input_dtype)
            cast_option.outDataType = cast_dtype
            cast_operator.builtinOptions = cast_option
            self.graph.add_operator(cast_operator)
            casted_input = cast_tensor

        inputs = [casted_input, dim_i32]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.CUMSUM, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.CumsumOptions
        option = circle.CumsumOptions.CumsumOptionsT()
        operator.builtinOptions = option

        return operator
