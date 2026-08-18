# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

"""Circle-compatible SAME-padding Conv2d module facade."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import nn


class SamePaddingConv2d(nn.Conv2d):
    """Apply Circle Conv2D SAME padding to an NCHW tensor.

    PyTorch rejects ``padding="same"`` for strided convolutions. This module
    keeps the normal ``nn.Conv2d`` parameter layout while lowering through the
    Circle custom Conv2D or DepthwiseConv2D operator with SAME padding.

    Gradient-based quantization algorithms need an autograd-enabled execution
    path. When the activation input participates in a gradient graph, the
    module therefore evaluates the same padding and convolution with native
    ATen operators. Normal inference and export keep using the Circle custom
    operators so the serialized graph contract is unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create a regular or depthwise convolution with implicit SAME padding."""
        if padding not in (0, (0, 0)):
            raise ValueError("SamePaddingConv2d manages padding internally.")
        if padding_mode != "zeros":
            raise ValueError("SamePaddingConv2d supports zero padding only.")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )
        if self.groups != 1 and not (
            self.groups == self.in_channels
            and self.out_channels % self.in_channels == 0
        ):
            raise ValueError(
                "SamePaddingConv2d supports regular or depthwise convolution only."
            )

    def _conv_forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run an autograd-safe native path or lower through a Circle custom op."""
        if torch.is_grad_enabled() and input_.requires_grad:
            return _native_same_padding_conv2d(
                input_,
                weight,
                bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
            )

        input_nhwc = torch.ops.aten.permute.default(input_, [0, 2, 3, 1])
        stride = list(self.stride)
        dilation = list(self.dilation)

        if self.groups == 1:
            weight_circle = torch.ops.aten.permute.default(weight, [0, 2, 3, 1])
            output_nhwc = torch.ops.circle_custom.conv2d.padding(
                input_nhwc,
                weight_circle,
                bias,
                stride,
                "same",
                dilation,
                1,
            )
        else:
            weight_circle = torch.ops.aten.permute.default(weight, [1, 2, 3, 0])
            output_nhwc = torch.ops.circle_custom.depthwise_conv2d.padding(
                input_nhwc,
                weight_circle,
                bias,
                stride,
                "same",
                dilation,
                self.groups,
            )

        return torch.ops.aten.permute.default(output_nhwc, [0, 3, 1, 2])


def _native_same_padding_conv2d(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    stride: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    """Evaluate Circle SAME padding with differentiable PyTorch operators."""
    padding = _same_padding_2d(
        input_height=int(input_.shape[-2]),
        input_width=int(input_.shape[-1]),
        kernel_height=int(weight.shape[-2]),
        kernel_width=int(weight.shape[-1]),
        stride=stride,
        dilation=dilation,
    )
    padded = F.pad(input_, padding) if any(padding) else input_
    return F.conv2d(
        padded,
        weight,
        bias,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=groups,
    )


def _same_padding_2d(
    *,
    input_height: int,
    input_width: int,
    kernel_height: int,
    kernel_width: int,
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return ``F.pad`` order for Circle/TensorFlow SAME padding."""
    top, bottom = _same_padding_1d(
        input_height,
        kernel_height,
        stride[0],
        dilation[0],
    )
    left, right = _same_padding_1d(
        input_width,
        kernel_width,
        stride[1],
        dilation[1],
    )
    return left, right, top, bottom


def _same_padding_1d(
    input_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    """Return before/after padding for one SAME-padded spatial dimension."""
    if min(input_size, kernel_size, stride, dilation) <= 0:
        raise ValueError("SAME-padding dimensions must be positive.")
    output_size = (input_size + stride - 1) // stride
    effective_kernel = (kernel_size - 1) * dilation + 1
    total = max((output_size - 1) * stride + effective_kernel - input_size, 0)
    before = total // 2
    return before, total - before
