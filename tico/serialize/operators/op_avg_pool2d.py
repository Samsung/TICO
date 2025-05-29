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
from tico.serialize.circle_mapping import extract_circle_dtype, extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.define import define_pad_node
from tico.utils.validate_args_kwargs import AvgPool2dArgs


@register_node_visitor
class AvgPool2DVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.circle_custom.avgpool2d]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = AvgPool2dArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        kernel_size = args.kernel_size
        stride = args.stride
        padding = args.padding

        avgpool_input: torch.fx.Node | circle.Tensor.TensorT = input

        def define_padding_node():
            assert isinstance(padding, list), type(padding)
            padding_vec = torch.tensor(
                [
                    [0, 0],
                    [padding[0], padding[0]],
                    [padding[1], padding[1]],
                    [0, 0],
                ],
                dtype=torch.int32,
            )
            input_shape = list(extract_shape(input))
            input_dtype: int = extract_circle_dtype(input)
            padded_input_shape = [
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ]
            padded_input_shape[1] += padding[0] * 2
            padded_input_shape[2] += padding[1] * 2
            # create padded input tensor
            padded_input_tensor = self.graph.add_tensor_from_scratch(
                prefix=f"{input.name}_pad_output",
                shape=padded_input_shape,
                dtype=input_dtype,
                source_node=node,
            )
            pad_operator = define_pad_node(
                self.graph, self._op_codes, [input, padding_vec], [padded_input_tensor]
            )
            self.graph.add_operator(pad_operator)
            return padded_input_tensor

        if padding is not None:
            avgpool_input = define_padding_node()

        inputs = [avgpool_input]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.AVERAGE_POOL_2D,
            self._op_codes,
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.Pool2DOptions
        option = circle.Pool2DOptions.Pool2DOptionsT()

        SAME, VALID = 0, 1
        option.padding = VALID
        option.strideH = stride[0]
        option.strideW = stride[1]
        option.filterHeight = kernel_size[0]
        option.filterWidth = kernel_size[1]
        option.fusedActivationFunction = (
            circle.ActivationFunctionType.ActivationFunctionType.NONE
        )

        operator.builtinOptions = option

        return operator
