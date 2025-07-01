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

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_mapping import (
    circle_legalize_dtype_to,
    extract_circle_dtype,
    extract_shape,
)
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.define import define_pad_node
from tico.utils.padding import identify_padding
from tico.utils.validate_args_kwargs import ConvTranspose2DArgs


@register_node_visitor
class TransposeConvVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.transpose_conv,
    ]

    def define_transpose_conv_node(
        self, padding: int, stride: List, inputs: List, outputs: List
    ) -> circle.Operator.OperatorT:
        def set_transpose_conv_option(operator, stride):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.TransposeConvOptions
            )
            option = circle.TransposeConvOptions.TransposeConvOptionsT()
            option.padding = padding
            option.strideH = stride[0]
            option.strideW = stride[1]
            option.fusedActivationFunction = (
                circle.ActivationFunctionType.ActivationFunctionType.NONE
            )
            operator.builtinOptions = option

        transpose_conv_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.TRANSPOSE_CONV, self._op_codes
        )
        operator = create_builtin_operator(
            self.graph, transpose_conv_op_index, inputs, outputs
        )
        set_transpose_conv_option(operator, stride)
        return operator

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_node(self, node: torch.fx.Node) -> circle.Operator.OperatorT:
        args = ConvTranspose2DArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        input_ = args.input
        weight = args.weight
        bias = args.bias
        stride = args.stride
        padding = args.padding
        output_padding = args.output_padding
        groups = args.groups
        dilation = args.dilation

        assert groups == 1, "Only support group 1"

        input_shape = list(extract_shape(input_))
        output_shape = list(extract_shape(node))
        weight_shape = list(extract_shape(weight))
        assert len(input_shape) == 4, len(input_shape)
        assert len(output_shape) == 4, len(output_shape)
        assert len(weight_shape) == 4, len(weight_shape)

        pad_decision = identify_padding(padding, input_shape, output_shape, stride)

        conv_input: torch.fx.Node | circle.Tensor.TensorT = input_
        if pad_decision.explicit_pad_hw is not None:
            pad_h, pad_w = pad_decision.explicit_pad_hw
            paddings = torch.tensor(
                [
                    [0, 0],
                    [pad_h, pad_h],
                    [pad_w, pad_w],
                    [0, 0],
                ],
                dtype=torch.int32,
            )
            pad_output_shape = [
                input_shape[0],
                input_shape[1] + pad_h * 2,
                input_shape[2] + pad_w * 2,
                input_shape[3],
            ]
            # create padded output tensor
            input_qparam: Optional[QuantParam] = input_.meta.get(QPARAM_KEY)
            pad_output = self.graph.add_tensor_from_scratch(
                prefix=f"{node.name}_input_pad_output",
                shape=pad_output_shape,
                dtype=extract_circle_dtype(input_),
                qparam=input_qparam,
                source_node=node,
            )
            # CirclePad
            pad_operator = define_pad_node(
                self.graph, self._op_codes, [input_, paddings], [pad_output]
            )
            self.graph.add_operator(pad_operator)
            conv_input = pad_output

        if bias is None:
            # luci-interpreter can't run no bias conv. Let's add zero vector for bias.
            bias = [0.0] * weight_shape[0]  # type: ignore[assignment]

        # First arguemnt is output shape of tconv.
        assert output_shape[0] == input_shape[0]
        assert output_shape[3] == weight_shape[0]
        tconv_output = circle_legalize_dtype_to(output_shape, dtype=torch.int32)

        tconv_output_tensor = self.graph.add_const_tensor(
            tconv_output, source_node=node
        )

        # TConv2D
        tconv2d_operator = self.define_transpose_conv_node(
            pad_decision.conv_padding_type,  # 'SAME'(0) or 'VALID'(1)
            stride,
            [tconv_output_tensor, weight, conv_input, bias],
            [node],
        )

        return tconv2d_operator
