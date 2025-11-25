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
    extract_circle_shape,
    to_circle_shape,
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

    def define_slice_node(
        self,
        src_tensor,
        begin_vals: List[int],
        size_vals: List[int],
        dst_tensor,
    ) -> circle.Operator.OperatorT:
        slice_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SLICE, self._op_codes
        )

        # Begin / Size as int32 const tensors
        begin_arr = circle_legalize_dtype_to(begin_vals, dtype=torch.int32)
        size_arr = circle_legalize_dtype_to(size_vals, dtype=torch.int32)

        operator = create_builtin_operator(
            self.graph, slice_op_index, [src_tensor, begin_arr, size_arr], [dst_tensor]
        )
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.SliceOptions
        option = circle.SliceOptions.SliceOptionsT()
        operator.builtinOptions = option
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
        groups = args.groups

        assert groups == 1, "Only support group 1"

        input_shape, input_shape_signature = extract_circle_shape(input_)
        output_shape, _ = extract_circle_shape(node)
        weight_shape, _ = extract_circle_shape(weight)
        assert len(input_shape) == 4, len(input_shape)
        assert len(output_shape) == 4, len(output_shape)
        assert len(weight_shape) == 4, len(weight_shape)

        pad_decision = identify_padding(
            padding, input_shape, output_shape, stride, is_transpose=True
        )

        conv_input: torch.fx.Node | circle.Tensor.TensorT = input_
        if bias is None:
            # luci-interpreter can't run no bias conv. Let's add zero vector for bias.
            bias = [0.0] * weight_shape[0]  # type: ignore[assignment]

        # Compute pre-crop output shape if we need to apply an explicit crop.
        if pad_decision.output_crop_hw is not None:
            pad_h, pad_w = pad_decision.output_crop_hw
            pre_h = int(output_shape[1]) + 2 * pad_h
            pre_w = int(output_shape[2]) + 2 * pad_w
            pre_out_shape = [output_shape[0], pre_h, pre_w, output_shape[3]]  # NHWC
        else:
            pre_out_shape = list(output_shape)

        tconv_output = circle_legalize_dtype_to(pre_out_shape, dtype=torch.int32)

        pre_out_cshape, pre_out_csig = to_circle_shape(pre_out_shape)
        tconv_tmp = node  # type: ignore[assignment]
        if pad_decision.output_crop_hw is not None:
            tconv_tmp = self.graph.add_tensor_from_scratch(  # type: ignore[assignment]
                prefix=f"{node.name}_tconv_out_pre_crop",
                shape=pre_out_cshape,
                shape_signature=pre_out_csig,
                dtype=extract_circle_dtype(node),
                qparam=node.meta.get(QPARAM_KEY),
                source_node=node,
            )

        tconv2d_operator = self.define_transpose_conv_node(
            pad_decision.conv_padding_type,
            stride,
            [tconv_output, weight, conv_input, bias],
            [tconv_tmp],
        )

        # If we need an output crop, insert a SLICE to produce the final tensor.
        if pad_decision.output_crop_hw is not None:
            self.graph.add_operator(tconv2d_operator)
            pad_h, pad_w = pad_decision.output_crop_hw
            begin = [0, pad_h, pad_w, 0]
            size = [
                int(output_shape[0]),
                int(output_shape[1]),
                int(output_shape[2]),
                int(output_shape[3]),
            ]

            slice_op = self.define_slice_node(tconv_tmp, begin, size, node)
            return slice_op

        return tconv2d_operator
