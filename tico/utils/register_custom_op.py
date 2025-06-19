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

from typing import List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.library import custom_op, register_fake

from tico.utils.mx.mx_ops import _quantize_mx

# Note that an operator assumes input tensor has NHWC format.
def CircleResizeNearestNeighbor():
    @custom_op("circle_custom::resize_nearest_neighbor", mutates_args=())
    def resize_nearest_neighbor(input_: torch.Tensor, size: List[int]) -> torch.Tensor:
        input_size = input_.size()
        H = input_size[1]
        W = input_size[2]
        H_scale_factor = size[1] / H
        W_scale_factor = size[2] / W
        if H_scale_factor != W_scale_factor:
            raise RuntimeError("Scale factor of H and W should be same.")
        return torch.nn.functional.interpolate(
            input_, scale_factor=H_scale_factor, mode="nearest"
        )

    @register_fake("circle_custom::resize_nearest_neighbor")
    def _(input_: torch.Tensor, size: List[int]):
        shape = list(input_.size())
        new_shape = [shape[0]] + list(size) + [shape[3]]
        result = torch.empty(new_shape, dtype=input_.dtype)
        return result


def CircleConv2d():
    """
    Note that this op follows the input spec of `aten.conv2d.default` whose number
     of arguments meets (2 <= node.args <= 7) condition.

    [RESTRICTION]
      Therefore, I tried to define a spec of conv2d as conv2d(input, weight, *args).
      But, custom operators in torch do not support positional-only args. So, I set it
       them as None by default.
    """

    @custom_op("circle_custom::conv2d", mutates_args=())
    def conv2d(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        dilation: Optional[List[int]] = None,
        groups: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = [0, 0] if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation
        groups = 1 if groups is None else groups

        if groups != 1:
            raise RuntimeError(
                f"CircleConv2d only supports 1 'groups'. the node's groups: {groups}"
            )

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHWI_to_OIHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        OIHW_weight = torch.ops.aten.permute.default(weight, OHWI_to_OIHW)

        args = [NCHW_input, OIHW_weight, bias, stride, padding, dilation, groups]
        NCHW_output = torch.ops.aten.conv2d.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output

    @register_fake("circle_custom::conv2d")
    def _(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        dilation: Optional[List[int]] = None,
        groups: Optional[int] = None,
    ):
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = [0, 0] if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation
        groups = 1 if groups is None else groups
        if groups != 1:
            raise RuntimeError(
                f"CircleConv2d only supports 1 'groups'. the node's groups: {groups}"
            )

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHWI_to_OIHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        OIHW_weight = torch.ops.aten.permute.default(weight, OHWI_to_OIHW)

        args = [NCHW_input, OIHW_weight, bias, stride, padding, dilation, groups]
        NCHW_output = torch.ops.aten.conv2d.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output


def CircleConv2dPadding():
    """
    Almost same with `CircleConv2d` except padding arugment is a string type.

    Q) Why create another custom op rather than make `CircleConv2d` cover multiple padding type?
    A) `padding` with Optional[Union[List[int], str]] type is not allowed in torch.
    """

    @custom_op("circle_custom::conv2d.padding", mutates_args=())
    def conv2d_padding(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[str] = None,
        dilation: Optional[List[int]] = None,
        groups: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = "valid" if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation
        groups = 1 if groups is None else groups
        if groups != 1:
            raise RuntimeError(
                f"CircleConv2d only supports 1 'groups'. the node's groups: {groups}"
            )

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHWI_to_OIHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        OIHW_weight = torch.ops.aten.permute.default(weight, OHWI_to_OIHW)

        args = [NCHW_input, OIHW_weight, bias, stride, padding, dilation, groups]
        NCHW_output = torch.ops.aten.conv2d.padding(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output

    @register_fake("circle_custom::conv2d.padding")
    def _(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[str] = None,
        dilation: Optional[List[int]] = None,
        groups: Optional[int] = None,
    ):
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = "valid" if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation
        groups = 1 if groups is None else groups
        if groups != 1:
            raise RuntimeError(
                f"CircleConv2d only supports 1 'groups'. the node's groups: {groups}"
            )

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHWI_to_OIHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        OIHW_weight = torch.ops.aten.permute.default(weight, OHWI_to_OIHW)

        args = [NCHW_input, OIHW_weight, bias, stride, padding, dilation, groups]
        NCHW_output = torch.ops.aten.conv2d.padding(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output


def CircleDepthwiseConv2d():
    """
    Note that this op follows the input spec of `aten.conv2d.default` whose number
     of arguments meets (2 <= node.args <= 7) condition.

    [RESTRICTION]
      Therefore, I tried to define a spec of conv2d as conv2d(input, weight, *args).
      But, custom operators in torch do not support positional-only args. So, I set it
       them as None by default.
    """

    @custom_op("circle_custom::depthwise_conv2d", mutates_args=())
    def depthwise_conv2d(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        dilation: Optional[List[int]] = None,
        groups: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = [0, 0] if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation

        assert groups and groups > 1

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHW1_to_1OHW = [3, 0, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        _1OHW_weight = torch.ops.aten.permute.default(weight, OHW1_to_1OHW)

        args = [NCHW_input, _1OHW_weight, bias, stride, padding, dilation, groups]
        NCHW_output = torch.ops.aten.conv2d.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output

    @register_fake("circle_custom::depthwise_conv2d")
    def _(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        dilation: Optional[List[int]] = None,
        groups: Optional[int] = None,
    ):
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = [0, 0] if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation

        assert groups and groups > 1

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHW1_to_1OHW = [3, 0, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        _1OHW_weight = torch.ops.aten.permute.default(weight, OHW1_to_1OHW)

        args = [NCHW_input, _1OHW_weight, bias, stride, padding, dilation, groups]
        NCHW_output = torch.ops.aten.conv2d.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output


def CircleDepthwiseConv2dPadding():
    @custom_op("circle_custom::depthwise_conv2d.padding", mutates_args=())
    def depthwise_conv2d_padding(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[str] = None,
        dilation: Optional[List[int]] = None,
        groups: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = "valid" if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation

        assert groups and groups > 1

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHW1_to_1OHW = [3, 0, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        _1OHW_weight = torch.ops.aten.permute.default(weight, OHW1_to_1OHW)

        args = [NCHW_input, _1OHW_weight, bias, stride, padding, dilation, groups]
        NCHW_output = torch.ops.aten.conv2d.padding(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output

    @register_fake("circle_custom::depthwise_conv2d.padding")
    def _(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[str] = None,
        dilation: Optional[List[int]] = None,
        groups: Optional[int] = None,
    ):
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = "valid" if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation

        assert groups and groups > 1

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHW1_to_1OHW = [3, 0, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        _1OHW_weight = torch.ops.aten.permute.default(weight, OHW1_to_1OHW)

        args = [NCHW_input, _1OHW_weight, bias, stride, padding, dilation, groups]
        NCHW_output = torch.ops.aten.conv2d.padding(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output


def CircleMaxPool2D():
    """
    Note that this op follows the input spec of `aten.max_pool2d_with_indices.default` whose number
     of arguments meets (3 <= node.args <= 6) condition.

    [RESTRICTION]
      Custom operators in torch do not support positional-only args. So, I set it
       them as None by default.
    """

    @custom_op("circle_custom::maxpool2d", mutates_args=())
    def maxpool2d(
        input_: torch.Tensor,
        kernel_size: List[int],
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        dilation: Optional[List[int]] = None,
        ceil_mode: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = kernel_size if not stride else stride
        padding = [0, 0] if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation
        ceil_mode = False if ceil_mode is None else ceil_mode

        NHWC_to_NCHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)

        args = [NCHW_input, kernel_size, stride, padding, dilation, ceil_mode]
        NCHW_output = torch.ops.aten.max_pool2d_with_indices.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        # use first output
        NHWC_output = torch.ops.aten.permute.default(NCHW_output[0], NCHW_to_NHWC)

        return NHWC_output

    @register_fake("circle_custom::maxpool2d")
    def _(
        input_: torch.Tensor,
        kernel_size: List[int],
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        dilation: Optional[List[int]] = None,
        ceil_mode: Optional[bool] = None,
    ):
        """
        Set default values.

        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = kernel_size if not stride else stride
        padding = [0, 0] if padding is None else padding
        dilation = [1, 1] if dilation is None else dilation
        ceil_mode = False if ceil_mode is None else ceil_mode

        NHWC_to_NCHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)

        args = [NCHW_input, kernel_size, stride, padding, dilation, ceil_mode]
        NCHW_output = torch.ops.aten.max_pool2d_with_indices.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        # use first output
        NHWC_output = torch.ops.aten.permute.default(NCHW_output[0], NCHW_to_NHWC)

        return NHWC_output


def CircleAvgPool2D():
    @custom_op("circle_custom::avgpool2d", mutates_args=())
    def avgpool2d(
        input_: torch.Tensor,
        kernel_size: List[int],
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        ceil_mode: Optional[bool] = None,
        count_include_pad: Optional[bool] = None,
        divisor_override: Optional[int] = None,
    ) -> torch.Tensor:
        stride = kernel_size if not stride else stride
        padding = [0, 0] if padding is None else padding
        ceil_mode = False if ceil_mode is None else ceil_mode
        count_include_pad = True if count_include_pad is None else count_include_pad
        divisor_override = None if divisor_override is None else divisor_override

        NHWC_to_NCHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)

        args = [
            NCHW_input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        ]
        NCHW_output = torch.ops.aten.avg_pool2d.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output

    @register_fake("circle_custom::avgpool2d")
    def _(
        input_: torch.Tensor,
        kernel_size: List[int],
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        ceil_mode: Optional[bool] = None,
        count_include_pad: Optional[bool] = None,
        divisor_override: Optional[int] = None,
    ):
        stride = kernel_size if not stride else stride
        padding = [0, 0] if padding is None else padding
        ceil_mode = False if ceil_mode is None else ceil_mode
        count_include_pad = True if count_include_pad is None else count_include_pad
        divisor_override = None if divisor_override is None else divisor_override

        NHWC_to_NCHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)

        args = [
            NCHW_input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        ]
        NCHW_output = torch.ops.aten.avg_pool2d.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output


def CircleInstanceNorm():
    @custom_op("circle_custom::instance_norm", mutates_args=())
    def instance_norm(
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        use_input_stats: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-05,
        cudnn_enabled: bool = False,
    ) -> torch.Tensor:
        NHWC_to_NCHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)

        args = [NCHW_input, weight, bias, None, None, False, momentum, eps, False]
        NCHW_output = torch.ops.aten.instance_norm.default(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output

    @register_fake("circle_custom::instance_norm")
    def _(
        input: FakeTensor,
        weight: Optional[FakeTensor] = None,
        bias: Optional[FakeTensor] = None,
        running_mean: Optional[FakeTensor] = None,
        running_var: Optional[FakeTensor] = None,
        use_input_stats: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-05,
        cudnn_enabled: bool = False,
    ):
        # shape is preserved
        return input.new_empty(input.size())


def CircleQuantizeMX():
    # This operator conducts fake-quantization of microscaling
    # NOTE Why using "quantize"_mx not "fake_quantize"_mx?
    # To align with function name of microxcaling repo.
    # https://github.com/microsoft/microxcaling/blob/v1.1.0/mx/mx_ops.py#L173
    @custom_op("circle_custom::quantize_mx", mutates_args=())
    def quantize_mx(
        input_: torch.Tensor,
        elem_format: str,
        axis: int,
        shared_exp_method: str = "max",
        round: str = "nearest",
    ) -> torch.Tensor:
        if elem_format == "int8":
            scale_bits = 8
            block_size = 32
        else:
            raise RuntimeError(f"Unsupported elem_format in quantize_mx: {elem_format}")

        result = _quantize_mx(
            input_,
            scale_bits=scale_bits,
            elem_format=elem_format,
            axes=[axis],
            block_size=block_size,
            shared_exp_method=shared_exp_method,
            round=round,
        )
        return result

    @register_fake("circle_custom::quantize_mx")
    def _(
        input_: torch.Tensor,
        elem_format: str,
        axis: int,
        shared_exp_method: str = "max",  # Fixed
        round: str = "nearest",  # Fixed
    ) -> torch.Tensor:
        return input_


# Add custom ops to the torch namespace
def RegisterOps():
    CircleResizeNearestNeighbor()
    CircleDepthwiseConv2d()
    CircleDepthwiseConv2dPadding()
    CircleConv2d()
    CircleConv2dPadding()
    CircleMaxPool2D()
    CircleAvgPool2D()
    CircleInstanceNorm()
    CircleQuantizeMX()
