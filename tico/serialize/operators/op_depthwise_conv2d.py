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

from tico.serialize.circle_mapping import extract_circle_dtype, extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.define import define_pad_node
from tico.utils.padding import is_same_padding, is_valid_padding, SAME, VALID
from tico.utils.validate_args_kwargs import Conv2DArgs


@register_node_visitor
class DepthwiseConv2dVisitor(NodeVisitor):
    """
    NOTE
    - The padding of DepthwiseCircleConv2D has only padding type('VALID', 'SAME') in circle, but the padding of nn.Conv2d has padding type(('valid', 'same')), padding value(int)
    and padding value(tuple->[pad_h, pad_w]).
    ref: https://tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d

    [1] With valid/same padding: DepthwiseCircleConv2D (only)

        [ATEN IR]
        Input[NHWC] ---- circle_cumstom.depthwise_conv2d[NHWC] ---- OUTPUT[NHWC]
        Weight[NHWC] ---/
        Bias ----------/

        [CIRCLE IR]
        Input[NHWC] ----  DepthwiseCircleConv2D[NHWC] ---- OUTPUT[NHWC]
        Weight[NHWC] ---/
        Bias ----------/

    [2] With additional padding: CirclePad + DepthwiseCircleConv2D

        [ATEN IR]
        Input[NHWC] ---- circle_cumstom.depthwise_conv2d[NHWC] ---- OUTPUT[NHWC]
        Weight[NHWC] ---/
        Bias ----------/

        [CIRCLE IR]
        Input[NHWC] ---- CirclePad[NHWC] ---- DepthwiseCircleConv2D[NHWC] ---- OUTPUT[NHWC]
                         Weight[NHWC] ------/
                         Bias -------------/
    """

    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.depthwise_conv2d,
        torch.ops.circle_custom.depthwise_conv2d.padding,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_dconv_node(
        self,
        padding: int,
        stride: List[int],
        dilation: List[int],
        depthMultiplier: int,
        inputs: List,
        outputs: List,
    ) -> circle.Operator.OperatorT:
        def set_conv2d_option(operator, stride, dilation):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.DepthwiseConv2DOptions
            )
            option = circle.DepthwiseConv2DOptions.DepthwiseConv2DOptionsT()

            option.padding = padding
            option.strideH = stride[0]
            option.strideW = stride[1]
            option.depthMultiplier = depthMultiplier
            option.dilationHFactor = dilation[0]
            option.dilationWFactor = dilation[1]
            option.fusedActivationFunction = (
                circle.ActivationFunctionType.ActivationFunctionType.NONE
            )
            operator.builtinOptions = option

        conv2d_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.DEPTHWISE_CONV_2D, self._op_codes
        )
        operator = create_builtin_operator(self.graph, conv2d_op_index, inputs, outputs)
        set_conv2d_option(operator, stride, dilation)
        return operator

    def define_node(self, node: torch.fx.Node) -> circle.Operator.OperatorT:
        # Let's get Conv2dArgs because torch Conv2D with group == input_channel maps to CircleDepthwiseConv2D
        args = Conv2DArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input_ = args.input
        weight = args.weight
        bias = args.bias
        stride = args.stride
        padding = args.padding
        dilation = args.dilation
        groups = args.groups

        input_dtype: int = extract_circle_dtype(input_)
        input_shape = list(extract_shape(input_))  # OHWI
        assert len(input_shape) == 4, len(input_shape)

        output_shape = list(extract_shape(node))  # OHWI
        assert len(output_shape) == 4, len(output_shape)

        weight_shape = list(extract_shape(weight))  # 1HWO
        assert (
            weight_shape[3] % groups == 0
        ), "Depthwise convolution requires output channel to be divisible by groups"

        assert weight_shape[0] == 1
        assert weight_shape[3] == output_shape[3]
        assert input_shape[3] == groups

        depthMultiplier = weight_shape[3] // input_shape[3]
        assert weight_shape[3] % input_shape[3] == 0, "depthMultiplier must be integer"

        conv_input: torch.fx.node.Node | circle.Tensor.TensorT = input_

        if is_valid_padding(padding):
            dconv2d_padding_type = VALID
        elif is_same_padding(padding, input_shape, output_shape):
            dconv2d_padding_type = SAME
        else:
            assert isinstance(padding, list) and len(padding) == 2

            dconv2d_padding_type = VALID

            # Padding is not valid or same, so we use valid padding and add padding operator before conv2d operator.
            # when data_format is "NHWC", padding should be [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
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

            pad_output = self.graph.add_tensor_from_scratch(
                prefix=f"{node.name}_input_pad_output",
                shape=pad_output_shape,
                dtype=input_dtype,
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
            out_channel = weight_shape[3]
            bias = [0.0] * out_channel  # type: ignore[assignment]

        # DConv2D
        dconv2d_operator = self.define_dconv_node(
            dconv2d_padding_type,
            stride,
            dilation,
            depthMultiplier,
            [conv_input, weight, bias],
            [node],
        )

        return dconv2d_operator
