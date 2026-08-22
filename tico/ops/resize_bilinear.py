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

"""Public PyTorch facade for Circle RESIZE_BILINEAR semantics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _normalize_size(size: Sequence[int]) -> tuple[int, int]:
    """Validate and return a fixed two-dimensional output size."""
    if len(size) != 2:
        raise ValueError(
            "ResizeBilinear output size must contain exactly two values, "
            f"but received {list(size)}."
        )

    output_height = int(size[0])
    output_width = int(size[1])
    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            "ResizeBilinear output dimensions must be positive, "
            f"but received {(output_height, output_width)}."
        )
    return output_height, output_width


def _validate_coordinate_options(
    *, align_corners: bool, half_pixel_centers: bool
) -> None:
    """Validate the coordinate options accepted by Circle ResizeBilinear."""
    if align_corners and half_pixel_centers:
        raise ValueError(
            "ResizeBilinear does not allow align_corners and "
            "half_pixel_centers to be enabled together."
        )


class ResizeBilinear2d(nn.Module):
    """Resize an NCHW tensor using Circle RESIZE_BILINEAR semantics.

    The module is a thin frontend facade. The eager custom-op implementation,
    fake-tensor implementation, and custom-op registration live in
    ``tico.utils.register_custom_op`` with the other internal Circle operators.
    """

    def __init__(
        self,
        size: Sequence[int],
        *,
        align_corners: bool = False,
        half_pixel_centers: bool = False,
    ) -> None:
        """Store the fixed output size and Circle coordinate options."""
        super().__init__()
        self.size = _normalize_size(size)
        _validate_coordinate_options(
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers,
        )
        self.align_corners = bool(align_corners)
        self.half_pixel_centers = bool(half_pixel_centers)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Resize one rank-4 NCHW tensor."""
        if input_.dim() != 4:
            raise ValueError(
                "ResizeBilinear2d expects a rank-4 NCHW input, "
                f"but received rank {input_.dim()}."
            )

        input_nhwc = torch.ops.aten.permute.default(input_, [0, 2, 3, 1])
        eager_autograd = (
            torch.is_grad_enabled()
            and input_.requires_grad
            and not torch.compiler.is_compiling()
        )
        if eager_autograd:
            # Calling the differentiable reference directly avoids treating the
            # custom operator as an opaque autograd boundary. Export and
            # non-gradient eager execution keep the Circle custom operator.
            from tico.utils.register_custom_op import _resize_bilinear_nhwc_reference

            output_nhwc = _resize_bilinear_nhwc_reference(
                input_nhwc,
                [self.size[0], self.size[1]],
                align_corners=self.align_corners,
                half_pixel_centers=self.half_pixel_centers,
            )
        else:
            output_nhwc = torch.ops.circle_custom.resize_bilinear.default(
                input_nhwc,
                [self.size[0], self.size[1]],
                self.align_corners,
                self.half_pixel_centers,
            )
        return torch.ops.aten.permute.default(output_nhwc, [0, 3, 1, 2])

    def extra_repr(self) -> str:
        """Return a readable representation of the fixed resize attributes."""
        return (
            f"size={self.size}, align_corners={self.align_corners}, "
            f"half_pixel_centers={self.half_pixel_centers}"
        )
