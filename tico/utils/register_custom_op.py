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

from tico.utils.mx.dtypes import normalize_mx_elem_format, SUPPORTED_MX_ELEM_FORMATS
from tico.utils.mx.mx_ops import _quantize_mx


def _same_padding_1d(
    input_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    """Return TensorFlow Lite SAME padding before and after one dimension."""
    output_size = (input_size + stride - 1) // stride
    effective_kernel = (kernel_size - 1) * dilation + 1
    total_padding = max(
        (output_size - 1) * stride + effective_kernel - input_size,
        0,
    )
    padding_before = total_padding // 2
    return padding_before, total_padding - padding_before


def _conv2d_with_string_padding(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: List[int],
    padding: str,
    dilation: List[int],
    groups: int,
) -> torch.Tensor:
    """Run NCHW Conv2d with Circle VALID or SAME padding semantics."""
    padding = padding.lower()
    if padding == "same":
        top, bottom = _same_padding_1d(
            input_.shape[-2],
            weight.shape[-2],
            stride[0],
            dilation[0],
        )
        left, right = _same_padding_1d(
            input_.shape[-1],
            weight.shape[-1],
            stride[1],
            dilation[1],
        )
        if any((left, right, top, bottom)):
            input_ = torch.ops.aten.constant_pad_nd.default(
                input_,
                [left, right, top, bottom],
                0.0,
            )
    elif padding != "valid":
        raise RuntimeError(f"Unsupported Conv2d padding mode: {padding!r}")

    return torch.ops.aten.conv2d.default(
        input_, weight, bias, stride, [0, 0], dilation, groups
    )


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
        permuted = torch.permute(input_, [0, 3, 1, 2])
        resized = torch.nn.functional.interpolate(
            permuted, scale_factor=H_scale_factor, mode="nearest"
        )
        return torch.permute(resized, [0, 2, 3, 1])

    @register_fake("circle_custom::resize_nearest_neighbor")
    def _(input_: torch.Tensor, size: List[int]):
        shape = list(input_.size())
        new_shape = [shape[0]] + list(size) + [shape[3]]
        result = torch.empty(new_shape, dtype=input_.dtype)
        return result


def _normalize_resize_bilinear_size(size: List[int]) -> tuple[int, int]:
    """Validate and return a two-dimensional ResizeBilinear output size."""
    if len(size) != 2:
        raise RuntimeError(
            "ResizeBilinear output size must contain exactly two values, "
            f"but received {size}."
        )

    output_height = int(size[0])
    output_width = int(size[1])
    if output_height <= 0 or output_width <= 0:
        raise RuntimeError(
            "ResizeBilinear output dimensions must be positive, "
            f"but received {(output_height, output_width)}."
        )
    return output_height, output_width


def _validate_resize_bilinear_options(
    *, align_corners: bool, half_pixel_centers: bool
) -> None:
    """Validate the coordinate options accepted by Circle ResizeBilinear."""
    if align_corners and half_pixel_centers:
        raise RuntimeError(
            "ResizeBilinear does not allow align_corners and "
            "half_pixel_centers to be enabled together."
        )


def _resize_bilinear_source_coordinates(
    input_size: int,
    output_size: int,
    *,
    align_corners: bool,
    half_pixel_centers: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return source coordinates for one ResizeBilinear spatial axis."""
    output = torch.arange(output_size, dtype=dtype, device=device)
    if align_corners:
        if output_size == 1:
            return torch.zeros_like(output)
        return output * float(input_size - 1) / float(output_size - 1)
    if half_pixel_centers:
        return (output + 0.5) * float(input_size) / float(output_size) - 0.5
    return output * float(input_size) / float(output_size)


def _resize_bilinear_nhwc_reference(
    input_: torch.Tensor,
    size: List[int],
    *,
    align_corners: bool,
    half_pixel_centers: bool,
) -> torch.Tensor:
    """Apply Circle ResizeBilinear semantics to one floating NHWC tensor."""
    if input_.dim() != 4:
        raise RuntimeError(
            "ResizeBilinear expects a rank-4 NHWC input, "
            f"but received rank {input_.dim()}."
        )
    if not input_.is_floating_point():
        raise RuntimeError(
            "The eager ResizeBilinear reference expects a floating-point "
            f"input, but received {input_.dtype}."
        )

    output_height, output_width = _normalize_resize_bilinear_size(size)
    _validate_resize_bilinear_options(
        align_corners=align_corners,
        half_pixel_centers=half_pixel_centers,
    )

    input_height = int(input_.shape[1])
    input_width = int(input_.shape[2])
    if input_height <= 0 or input_width <= 0:
        raise RuntimeError(
            "ResizeBilinear input spatial dimensions must be positive, "
            f"but received {(input_height, input_width)}."
        )

    compute_dtype = torch.float64 if input_.dtype == torch.float64 else torch.float32
    input_for_compute = input_.to(dtype=compute_dtype)
    source_y = _resize_bilinear_source_coordinates(
        input_height,
        output_height,
        align_corners=align_corners,
        half_pixel_centers=half_pixel_centers,
        device=input_.device,
        dtype=compute_dtype,
    )
    source_x = _resize_bilinear_source_coordinates(
        input_width,
        output_width,
        align_corners=align_corners,
        half_pixel_centers=half_pixel_centers,
        device=input_.device,
        dtype=compute_dtype,
    )

    source_y = source_y.clamp(0.0, float(input_height - 1))
    source_x = source_x.clamp(0.0, float(input_width - 1))
    y0 = torch.floor(source_y).to(torch.int64)
    x0 = torch.floor(source_x).to(torch.int64)
    y1 = torch.clamp(y0 + 1, max=input_height - 1)
    x1 = torch.clamp(x0 + 1, max=input_width - 1)

    y_weight = (source_y - y0.to(compute_dtype)).reshape(1, output_height, 1, 1)
    x_weight = (source_x - x0.to(compute_dtype)).reshape(1, 1, output_width, 1)

    top_left = input_for_compute[:, y0[:, None], x0[None, :], :]
    top_right = input_for_compute[:, y0[:, None], x1[None, :], :]
    bottom_left = input_for_compute[:, y1[:, None], x0[None, :], :]
    bottom_right = input_for_compute[:, y1[:, None], x1[None, :], :]

    top = top_left + (top_right - top_left) * x_weight
    bottom = bottom_left + (bottom_right - bottom_left) * x_weight
    output = top + (bottom - top) * y_weight
    return output.to(dtype=input_.dtype)


def CircleResizeBilinear():
    """Register the internal channel-last Circle ResizeBilinear operator."""

    @custom_op("circle_custom::resize_bilinear", mutates_args=())
    def resize_bilinear(
        input_: torch.Tensor,
        size: List[int],
        align_corners: bool = False,
        half_pixel_centers: bool = False,
    ) -> torch.Tensor:
        """Execute the eager Circle ResizeBilinear reference."""
        return _resize_bilinear_nhwc_reference(
            input_,
            size,
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers,
        )

    @register_fake("circle_custom::resize_bilinear")
    def _(
        input_: torch.Tensor,
        size: List[int],
        align_corners: bool = False,
        half_pixel_centers: bool = False,
    ) -> torch.Tensor:
        """Infer metadata for the internal Circle ResizeBilinear operator."""
        if input_.dim() != 4:
            raise RuntimeError(
                "ResizeBilinear expects a rank-4 NHWC input, "
                f"but received rank {input_.dim()}."
            )
        output_height, output_width = _normalize_resize_bilinear_size(size)
        _validate_resize_bilinear_options(
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers,
        )
        return input_.new_empty(
            (
                input_.shape[0],
                output_height,
                output_width,
                input_.shape[3],
            )
        )


def CirclePReLU():
    """Register a channel-last PReLU operator for Circle lowering.

    PyTorch PReLU treats dimension 1 as the channel dimension, while Circle
    PReLU applies regular right-aligned broadcasting. This internal operator
    makes the channel-last contract explicit after layout legalization.
    """

    @custom_op("circle_custom::prelu", mutates_args=())
    def prelu(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply PReLU with channel-last broadcasting semantics."""
        if input_.dim() == 0:
            raise RuntimeError(
                "CirclePReLU requires an input tensor with rank greater than zero."
            )
        if weight.dim() != 1:
            raise RuntimeError(
                "CirclePReLU requires a rank-1 weight tensor, "
                f"but received rank {weight.dim()}."
            )

        channels = input_.size(-1)
        num_parameters = weight.numel()
        if num_parameters not in (1, channels):
            raise RuntimeError(
                "CirclePReLU weight must contain one value or match the last "
                "input dimension: "
                f"weight={num_parameters}, channels={channels}."
            )

        broadcast_shape = [1] * (input_.dim() - 1) + [num_parameters]
        alpha = weight.reshape(broadcast_shape)
        return torch.where(input_ >= 0, input_, input_ * alpha)

    @register_fake("circle_custom::prelu")
    def _(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Infer metadata for the internal channel-last PReLU operator."""
        if input_.dim() == 0:
            raise RuntimeError(
                "CirclePReLU requires an input tensor with rank greater than zero."
            )
        if weight.dim() != 1:
            raise RuntimeError(
                "CirclePReLU requires a rank-1 weight tensor, "
                f"but received rank {weight.dim()}."
            )

        channels = input_.shape[-1]
        num_parameters = weight.shape[0]
        if isinstance(channels, int) and isinstance(num_parameters, int):
            if num_parameters not in (1, channels):
                raise RuntimeError(
                    "CirclePReLU weight must contain one value or match the last "
                    "input dimension: "
                    f"weight={num_parameters}, channels={channels}."
                )

        return input_.new_empty(input_.size())


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

        NCHW_output = _conv2d_with_string_padding(
            NCHW_input,
            OIHW_weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
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

        NCHW_output = _conv2d_with_string_padding(
            NCHW_input,
            OIHW_weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
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

        NCHW_output = _conv2d_with_string_padding(
            NCHW_input,
            _1OHW_weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
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

        NCHW_output = _conv2d_with_string_padding(
            NCHW_input,
            _1OHW_weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output


def CircleTransposeConv():
    """
    Note that this op follows the input spec of `aten.conv_transpose2d.input` whose number
     of arguments meets (2 <= node.args <= 8) condition.
    [RESTRICTION]
      Therefore, I tried to define a spec of it as transpose_conv(input, weight, *args).
      But, custom operators in torch do not support positional-only args. So, I set it
       them as None by default.
    """

    @custom_op("circle_custom::transpose_conv", mutates_args=())
    def transpose_conv(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        output_padding: Optional[List[int]] = None,
        groups: Optional[int] = None,
        dilation: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Set default values.
        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = [0, 0] if padding is None else padding
        output_padding = [0, 0] if output_padding is None else output_padding
        groups = 1 if groups is None else groups
        dilation = [1, 1] if dilation is None else dilation
        if groups != 1:
            raise RuntimeError(
                f"CircleTransposeConv only supports 1 'groups'. the node's groups: {groups}"
            )

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHWI_to_IOHW = [3, 0, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        OIHW_weight = torch.ops.aten.permute.default(weight, OHWI_to_IOHW)

        args = [
            NCHW_input,
            OIHW_weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        ]
        NCHW_output = torch.ops.aten.conv_transpose2d.input(*args)
        NCHW_to_NHWC = [0, 2, 3, 1]
        NHWC_output = torch.ops.aten.permute.default(NCHW_output, NCHW_to_NHWC)

        return NHWC_output

    @register_fake("circle_custom::transpose_conv")
    def _(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        output_padding: Optional[List[int]] = None,
        groups: Optional[int] = None,
        dilation: Optional[List[int]] = None,
    ):
        """
        Set default values.
        Custom operators have limited types when it comes to default values.
        So, let's set them by None in input specs, and then, set it by default values.
        https://github.com/pytorch/pytorch/blob/6b05aafc/torch/_library/infer_schema.py#L131-L144
        """
        stride = [1, 1] if stride is None else stride
        padding = [0, 0] if padding is None else padding
        output_padding = [0, 0] if output_padding is None else output_padding
        groups = 1 if groups is None else groups
        dilation = [1, 1] if dilation is None else dilation
        if groups != 1:
            raise RuntimeError(
                f"CircleConv2d only supports 1 'groups'. the node's groups: {groups}"
            )

        NHWC_to_NCHW = [0, 3, 1, 2]
        OHWI_to_IOHW = [3, 0, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)
        OIHW_weight = torch.ops.aten.permute.default(weight, OHWI_to_IOHW)

        args = [
            NCHW_input,
            OIHW_weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        ]
        NCHW_output = torch.ops.aten.conv_transpose2d.input(*args)
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
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-05,
        cudnn_enabled: bool = False,
    ) -> torch.Tensor:
        NHWC_to_NCHW = [0, 3, 1, 2]
        NCHW_input = torch.ops.aten.permute.default(input_, NHWC_to_NCHW)

        args = [NCHW_input, weight, bias, None, None, True, momentum, eps, False]
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


def CircleMXFakeQuantize():
    """Register the eager MX fake-quantization custom operator."""

    @custom_op("circle_custom::mx_fake_quantize", mutates_args=())
    def mx_fake_quantize(
        input_: torch.Tensor,
        elem_format: str,
        axis: int,
        shared_exp_method: str = "max",
        round: str = "nearest",
    ) -> torch.Tensor:
        if elem_format not in SUPPORTED_MX_ELEM_FORMATS:
            raise RuntimeError(
                f"Unsupported elem_format in mx_fake_quantize: {elem_format}"
            )
        return _quantize_mx(
            input_,
            scale_bits=8,
            elem_format=normalize_mx_elem_format(elem_format),
            axes=[axis],
            block_size=32,
            shared_exp_method=shared_exp_method,
            round=round,
        )

    @register_fake("circle_custom::mx_fake_quantize")
    def _(
        input_: torch.Tensor,
        elem_format: str,
        axis: int,
        shared_exp_method: str = "max",
        round: str = "nearest",
    ) -> torch.Tensor:
        return input_


def CircleQuantizeMX():
    """Register the internal logical MX quantize custom operator."""

    @custom_op("circle_custom::quantize_mx", mutates_args=())
    def quantize_mx(
        input_: torch.Tensor,
        elem_format: str,
        axis: int,
        shared_exp_method: str = "max",
        round: str = "nearest",
    ) -> torch.Tensor:
        raise RuntimeError(
            "circle_custom::quantize_mx is an internal logical quantize op for "
            "Circle export. Use circle_custom::mx_fake_quantize for eager MX "
            "fake-quantization."
        )

    @register_fake("circle_custom::quantize_mx")
    def _(
        input_: torch.Tensor,
        elem_format: str,
        axis: int,
        shared_exp_method: str = "max",
        round: str = "nearest",
    ) -> torch.Tensor:
        return input_


def CircleDequantizeMX():
    """Register the internal logical MX dequantize custom operator."""

    @custom_op("circle_custom::dequantize_mx", mutates_args=())
    def dequantize_mx(
        input_: torch.Tensor,
        elem_format: str,
        axis: int,
        shared_exp_method: str = "max",
        round: str = "nearest",
    ) -> torch.Tensor:
        raise RuntimeError(
            "circle_custom::dequantize_mx is an internal logical dequantize op "
            "for Circle export and should be folded before eager execution."
        )

    @register_fake("circle_custom::dequantize_mx")
    def _(
        input_: torch.Tensor,
        elem_format: str,
        axis: int,
        shared_exp_method: str = "max",
        round: str = "nearest",
    ) -> torch.Tensor:
        return input_


def CircleRMSNorm():
    @custom_op("circle_custom::rms_norm", mutates_args=())
    def rms_norm(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-06,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states.to(input_dtype)

    @register_fake("circle_custom::rms_norm")
    def _(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-06,
    ) -> torch.Tensor:
        return hidden_states.new_empty(hidden_states.size())


def CircleAttention():
    @custom_op("circle_custom::attention", mutates_args=())
    def attention(
        hidden_states: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        return None

    @register_fake("circle_custom::attention")
    def _(
        hidden_states: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states


def CircleShape():
    """
    Custom operator to extract the shape of a tensor.
    This is similar to TensorFlow's shape operator and is used to preserve
    dynamic shape information in the Circle model.

    Args:
        input_: Input tensor

    Returns:
        A 1D tensor containing the shape of the input tensor
    """

    @custom_op("circle_custom::shape", mutates_args=())
    def shape(input_: torch.Tensor) -> torch.Tensor:
        # Return the shape of the input tensor as a 1D tensor
        shape_val = list(input_.size())
        return torch.tensor(shape_val, dtype=torch.int32)

    @register_fake("circle_custom::shape")
    def _(input_: torch.Tensor) -> torch.Tensor:
        # Return a 1D tensor with symbolic shape
        # The actual value will be determined at runtime
        rank = len(input_.size())
        return torch.empty([rank], dtype=torch.int32)


def RegisterGatherNdOp() -> None:
    """Register the internal logical GatherNd custom operator.

    This custom operator is an explicit TICO IR node for Circle GATHER_ND. It
    exists so that the FX graph does not keep an aten.gather node with changed
    semantics after lowering.
    """

    @custom_op("circle_custom::gather_nd", mutates_args=())
    def gather_nd(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather values or slices from params using full-coordinate indices.

        The output shape follows TensorFlow Lite GatherNd semantics:

            indices.shape[:-1] + params.shape[indices.shape[-1]:]
        """
        if indices.dim() < 1:
            raise RuntimeError("GatherNd indices must have rank greater than zero.")

        indices_nd = indices.size(-1)
        params_rank = params.dim()
        if indices_nd < 1:
            raise RuntimeError("GatherNd indices must contain at least one coordinate.")
        if indices_nd > params_rank:
            raise RuntimeError(
                "The last dimension of GatherNd indices must be less than or "
                "equal to the rank of params."
            )

        flat_indices = indices.reshape(-1, indices_nd).long()
        coordinate_indices = tuple(flat_indices[:, axis] for axis in range(indices_nd))
        gathered = params[coordinate_indices]
        output_shape = list(indices.shape[:-1]) + list(params.shape[indices_nd:])
        return gathered.reshape(output_shape)

    @register_fake("circle_custom::gather_nd")
    def _(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Infer the output shape of the internal GatherNd custom operator."""
        if indices.dim() < 1:
            raise RuntimeError("GatherNd indices must have rank greater than zero.")

        indices_nd = indices.shape[-1]
        if not isinstance(indices_nd, int):
            raise RuntimeError(
                "The last dimension of GatherNd indices must be statically known."
            )

        params_rank = params.dim()
        if indices_nd < 1:
            raise RuntimeError("GatherNd indices must contain at least one coordinate.")
        if indices_nd > params_rank:
            raise RuntimeError(
                "The last dimension of GatherNd indices must be less than or "
                "equal to the rank of params."
            )

        output_shape = list(indices.shape[:-1]) + list(params.shape[indices_nd:])
        return params.new_empty(output_shape)


# Add custom ops to the torch namespace
def RegisterOps():
    CircleResizeNearestNeighbor()
    CircleResizeBilinear()
    CirclePReLU()
    CircleDepthwiseConv2d()
    CircleDepthwiseConv2dPadding()
    CircleConv2d()
    CircleConv2dPadding()
    CircleTransposeConv()
    CircleMaxPool2D()
    CircleAvgPool2D()
    CircleInstanceNorm()
    CircleMXFakeQuantize()
    CircleQuantizeMX()
    CircleDequantizeMX()
    CircleRMSNorm()
    CircleAttention()
    CircleShape()
    RegisterGatherNdOp()
