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
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.circle_mapping import (
    circle_legalize_dtype_to,
    extract_circle_dtype,
    extract_shape,
    extract_torch_dtype,
)
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import AnyArgs


@register_node_visitor
class AnyVisitor(NodeVisitor):
    """
    Let's take NotEqual0 -> ReduceMax workaround for float, int
    [RESTRICTION]
        1. ReduceAny is not supported (luci-interpreter)
    [CASE: BOOL]
        (Bool tensors don't need 'Not Equal 0' at the first step.)
        bool[d0..dN]      --- Reduce Max      ---> bool[]
    [CASE: FLOAT, INT]
        int/float[d0..dN] --- Not Equal 0     ---> bool[d0,...dN]
                          --- Reduce Max      ---> bool[]
        * [d0..dN] means a tensor with any shape
        * [] means Scalar
    """

    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.any.default,
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_max_node(
        self, inputs: List, outputs: List, keepdims: bool
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.REDUCE_MAX, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.ReducerOptions
        )
        option = circle.ReducerOptions.ReducerOptionsT()
        option.keepDims = keepdims

        operator.builtinOptions = option

        return operator

    def define_ne_node(self, inputs: List, outputs: List) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.NOT_EQUAL, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.NotEqualOptions
        )
        option = circle.NotEqualOptions.NotEqualOptionsT()
        operator.builtinOptions = option
        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = AnyArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        dim = args.dim
        keepdim = args.keepdim

        input_shape = list(extract_shape(input))
        output_shape = list(extract_shape(node))

        if dim is None:
            dims = tuple(i for i in range(0, len(input_shape)))
            dim_i32 = tuple(
                circle_legalize_dtype_to(dim, dtype=torch.int32) for dim in dims
            )
        if isinstance(dim, int):
            dim_i32 = circle_legalize_dtype_to(dim, dtype=torch.int32)
        if isinstance(dim, tuple):
            dim_i32 = tuple(circle_legalize_dtype_to(d, dtype=torch.int32) for d in dim)

        inputs = [input, dim_i32]
        outputs = [node]

        dtype_torch = extract_torch_dtype(input)
        input_tensor: torch.fx.node.Node | circle.Tensor.TensorT = input

        if dtype_torch in [torch.int32, torch.int64, torch.float32, torch.float64]:
            dst_dtype_circle = circle.TensorType.TensorType.BOOL
            dst_dtype_torch = torch.bool
            ne_tensor: circle.Tensor.TensorT = self.graph.add_tensor_from_scratch(
                prefix=f"{input.name}_ne",
                shape=input_shape,
                dtype=dst_dtype_circle,
                source_node=input,
            )
            ne_node = self.define_ne_node(
                [input_tensor, torch.Tensor([0]).to(dtype_torch)], [ne_tensor]
            )
            self.graph.add_operator(ne_node)

            dtype_torch = dst_dtype_torch
            input_tensor = ne_tensor
            inputs = [ne_tensor, dim_i32]

        inputs = [input_tensor, dim_i32]

        reduce_node: circle.Operator.OperatorT = self.define_max_node(
            inputs, outputs, keepdim
        )

        return reduce_node
