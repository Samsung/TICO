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

from tico.serialize.circle_mapping import extract_circle_dtype, extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.define import define_pad_node
from tico.utils.padding import is_same_padding, is_valid_padding, SAME, VALID
from tico.utils.validate_args_kwargs import Conv2DArgs


@register_node_visitor
class Conv2dVisitor(NodeVisitor):
    """
    NOTE
    - The padding of CircleConv2D has only padding type('VALID', 'SAME') in circle, but the padding of nn.Conv2d has padding type(('valid', 'same')), padding value(int)
    and padding value(tuple->[pad_h, pad_w]).
    ref: https://tensorflow.org/api_docs/python/tf/nn/conv2d

    [1] With valid/same padding: CircleConv2D (only)

        [ATEN IR]
        Input[NHWC] ---- circle_cumstom.conv2d[NHWC] ---- OUTPUT[NHWC]
        Weight[NHWC] ---/
        Bias ----------/

        [CIRCLE IR]
        Input[NHWC] ----  CircleConv2D[NHWC] ---- OUTPUT[NHWC]
        Weight[NHWC] ---/
        Bias ----------/

    [2] With additional padding: CirclePad + CircleConv2D

        [ATEN IR]
        Input[NHWC] ---- circle_cumstom.conv2d[NHWC] ---- OUTPUT[NHWC]
        Weight[NHWC] ---/
        Bias ----------/

        [CIRCLE IR]
        Input[NHWC] ---- CirclePad[NHWC] ---- CircleConv2D[NHWC] ---- OUTPUT[NHWC]
                         Weight[NHWC] ------/
                         Bias -------------/
    """

    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.conv2d,
        torch.ops.circle_custom.conv2d.padding,
    ]

    def define_conv2d_node(
        self, padding: int, stride: List, dilation: List, inputs: List, outputs: List
    ) -> circle.Operator.OperatorT:
        def set_conv2d_option(operator, stride, dilation):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.Conv2DOptions
            )
            option = circle.Conv2DOptions.Conv2DOptionsT()
            option.padding = padding
            option.strideH = stride[0]
            option.strideW = stride[1]
            option.dilationHFactor = dilation[0]
            option.dilationWFactor = dilation[1]
            option.fusedActivationFunction = (
                circle.ActivationFunctionType.ActivationFunctionType.NONE
            )
            operator.builtinOptions = option

        conv2d_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.CONV_2D, self._op_codes
        )
        operator = create_builtin_operator(self.graph, conv2d_op_index, inputs, outputs)
        set_conv2d_option(operator, stride, dilation)
        return operator

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_node(self, node: torch.fx.Node) -> circle.Operator.OperatorT:
        # conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
        # conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, str padding="valid", SymInt[2] dilation=1, SymInt groups=1) -> Tensor
        args = Conv2DArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        input_ = args.input
        weight = args.weight
        bias = args.bias
        stride = args.stride
        padding = args.padding
        dilation = args.dilation
        groups = args.groups

        assert groups == 1, "Only support group 1 conv2d"

        input_dtype: int = extract_circle_dtype(input_)
        input_shape = list(extract_shape(input_))
        assert len(input_shape) == 4, len(input_shape)
        output_shape = extract_shape(node)
        assert len(output_shape) == 4, len(output_shape)

        conv_input: torch.fx.node.Node | circle.Tensor.TensorT = input_
        weight_shape = list(extract_shape(weight))

        if is_valid_padding(padding):
            conv2d_padding_type = VALID
        elif is_same_padding(padding, input_shape, output_shape):
            conv2d_padding_type = SAME
        else:
            assert isinstance(padding, list) and len(padding) == 2

            conv2d_padding_type = VALID

            # Padding is not valid or same, so we use valid padding and add padding operator before conv2d operator.
            # when data_foramt is "NHWC", padding should be [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
            paddings = torch.tensor(
                [
                    [0, 0],
                    [padding[0], padding[0]],
                    [padding[1], padding[1]],
                    [0, 0],
                ],
                dtype=torch.int32,
            )
            pad_output_shape = [
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ]
            # Add (pad_top+pad_bottom) to pad_output_shape_h
            pad_output_shape[1] += padding[0] * 2
            # Add (pad_left+pad_Right) to pad_output_shape_w
            pad_output_shape[2] += padding[1] * 2
            # create padded output tensor
            input_qparam: Optional[QuantParam] = (
                input_.meta[QPARAM_KEY] if QPARAM_KEY in input_.meta else None
            )
            pad_output = self.graph.add_tensor_from_scratch(
                prefix=f"{node.name}_input_pad_output",
                shape=pad_output_shape,
                dtype=input_dtype,
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
            assert len(weight_shape) == 4
            out_channel = weight_shape[0]
            bias = [0.0] * out_channel  # type: ignore[assignment]

        # Conv2D
        conv2d_operator = self.define_conv2d_node(
            conv2d_padding_type,  # 'SAME'(0) or 'VALID'(1)
            stride,
            dilation,
            [conv_input, weight, bias],
            [node],
        )

        return conv2d_operator
