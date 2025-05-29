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
from tico.serialize.circle_mapping import extract_torch_dtype, to_circle_dtype
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import PowTensorScalarArgs, PowTensorTensorArgs


class BasePowVisitor(NodeVisitor):
    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def cast_to_float(self, node: torch.fx.Node) -> circle.Tensor.TensorT:
        assert isinstance(node, torch.fx.Node), type(node)
        node_tensor: circle.Tensor.TensorT = self.graph.get_tensor(node)
        node_shape: List[int] = node_tensor.shape
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.CAST, self._op_codes
        )
        cast_name = f"{node.name}_cast"
        cast_dtype = circle.TensorType.TensorType.FLOAT32
        cast_tensor = self.graph.add_tensor_from_scratch(
            prefix=cast_name,
            dtype=cast_dtype,
            shape=node_shape,
            source_node=node,
        )
        cast_operator = create_builtin_operator(
            self.graph, op_index, [node], [cast_tensor]
        )
        cast_operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.CastOptions
        )
        option = circle.CastOptions.CastOptionsT()
        node_dtype = extract_torch_dtype(node)
        option.inDataType = to_circle_dtype(node_dtype)
        option.outDataType = cast_dtype
        cast_operator.builtinOptions = option
        self.graph.add_operator(cast_operator)

        return cast_tensor

    def define_pow_node(self, inputs: List, outputs: List) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.POW, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.PowOptions
        # Pow opearation does not have any options.
        option = circle.PowOptions.PowOptionsT()

        operator.builtinOptions = option

        return operator


# TODO Support `aten::pow.Scalar` (base=scalar, exponenent=tensor)
# ExecuTorch currently does not support it as of now (2024/02/13).


@register_node_visitor
class PowTensorScalarVisitor(BasePowVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.pow.Tensor_Scalar]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:

        args = PowTensorScalarArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        exponent = args.exponent

        lhs_dtype = extract_torch_dtype(input)
        # Circle supports only same dtype between lhs and rhs.
        if lhs_dtype == torch.float32 and isinstance(exponent, int):
            exponent = float(exponent)
        if lhs_dtype == torch.int32 or lhs_dtype == torch.int64:
            if isinstance(exponent, float):
                input = self.cast_to_float(input)  # type: ignore[assignment]

        operator = self.define_pow_node([input, exponent], [node])

        return operator


@register_node_visitor
class PowTensorTensorVisitor(BasePowVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.pow.Tensor_Tensor]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:

        args = PowTensorTensorArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        exponent = args.exponent  # type: ignore[arg-type]

        lhs_dtype = extract_torch_dtype(input)
        rhs_dtype = extract_torch_dtype(exponent)  # type: ignore[arg-type]
        # Circle supports only same dtype between lhs and rhs.
        if lhs_dtype == torch.float32 and rhs_dtype == torch.int:
            exponent = self.cast_to_float(exponent)  # type: ignore[arg-type, assignment]
        if lhs_dtype == torch.int32 or lhs_dtype == torch.int64:
            if rhs_dtype == torch.float32:
                input = self.cast_to_float(input)  # type: ignore[assignment]

        operator = self.define_pow_node([input, exponent], [node])

        return operator
