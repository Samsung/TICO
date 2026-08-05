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
from torch import nn


class SamePaddingConv2d(nn.Conv2d):
    """Apply Circle Conv2D SAME padding to an NCHW tensor.

    PyTorch rejects ``padding="same"`` for strided convolutions. This module
    keeps the normal ``nn.Conv2d`` parameter layout while lowering through the
    Circle custom Conv2D or DepthwiseConv2D operator with SAME padding.
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
        """Lower one NCHW convolution through a channel-last Circle custom op."""
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
