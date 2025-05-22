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

from enum import IntEnum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.circle_mapping import extract_circle_dtype, extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import (
    create_builtin_operator,
    get_integer_dtype_min,
    get_op_index,
)
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.validate_args_kwargs import MaxPool2dWithIndicesArgs


class PaddingType(IntEnum):
    SAME = 0
    VALID = 1


@register_node_visitor
class MaxPool2DWithIndicesVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.circle_custom.maxpool2d]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_padV2_node(
        self, inputs: List, outputs: List
    ) -> circle.Operator.OperatorT:
        def set_padv2_option(operator: circle.Operator.OperatorT):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.PadV2Options
            )
            option = circle.PadV2Options.PadV2OptionsT()
            operator.builtinOptions = option

        pad_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.PADV2, self._op_codes
        )
        operator = create_builtin_operator(self.graph, pad_op_index, inputs, outputs)
        set_padv2_option(operator)
        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        # max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
        args = MaxPool2dWithIndicesArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        kernel_size = args.kernel_size
        stride = args.stride
        padding = args.padding

        maxpool_input: torch.fx.Node | circle.Tensor.TensorT = input

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
            input_qparam: Optional[QuantParam] = (
                input.meta[QPARAM_KEY] if QPARAM_KEY in input.meta else None
            )
            # create padded input tensor
            padded_input_tensor = self.graph.add_tensor_from_scratch(
                prefix=f"{input.name}_pad_output",
                shape=padded_input_shape,
                dtype=input_dtype,
                qparam=input_qparam,
            )
            if input_qparam is not None:
                padding_value = get_integer_dtype_min(input_qparam.dtype)
            else:
                padding_value = float("-inf")
            pad_operator = self.define_padV2_node(
                [input, padding_vec, padding_value], [padded_input_tensor]
            )
            self.graph.add_operator(pad_operator)
            return padded_input_tensor

        padding_type = PaddingType.VALID
        if padding is not None:
            if extract_shape(input) == extract_shape(node):
                padding_type = PaddingType.SAME
            else:
                padding_type = PaddingType.VALID
                maxpool_input = define_padding_node()

        inputs = [maxpool_input]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.MAX_POOL_2D,
            self._op_codes,
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.Pool2DOptions
        option = circle.Pool2DOptions.Pool2DOptionsT()

        option.padding = int(padding_type)
        option.strideH = stride[0]
        option.strideW = stride[1]
        option.filterHeight = kernel_size[0]
        option.filterWidth = kernel_size[1]
        option.fusedActivationFunction = (
            circle.ActivationFunctionType.ActivationFunctionType.NONE
        )

        operator.builtinOptions = option

        return operator
