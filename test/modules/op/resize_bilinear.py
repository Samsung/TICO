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

"""System-test modules for Circle RESIZE_BILINEAR export."""

import torch
from tico.ops import ResizeBilinear2d

from test.modules.base import TestModuleBase


class AsymmetricResizeBilinear(TestModuleBase):
    """Exercise the legacy false/false coordinate mode."""

    def __init__(self):
        """Create a fixed asymmetric bilinear resize module."""
        super().__init__()
        self.resize = ResizeBilinear2d(
            (6, 8),
            align_corners=False,
            half_pixel_centers=False,
        )

    def forward(self, input_):
        """Resize the input tensor."""
        return self.resize(input_)

    def get_example_inputs(self):
        """Return a representative static input."""
        return (torch.randn(1, 2, 3, 4),), {}


class HalfPixelResizeBilinear(TestModuleBase):
    """Exercise the half-pixel coordinate mode used by the hand detector."""

    def __init__(self):
        """Create a fixed half-pixel bilinear resize module."""
        super().__init__()
        self.resize = ResizeBilinear2d(
            (6, 8),
            align_corners=False,
            half_pixel_centers=True,
        )

    def forward(self, input_):
        """Resize the input tensor."""
        return self.resize(input_)

    def get_example_inputs(self):
        """Return a representative static input."""
        return (torch.randn(1, 2, 3, 4),), {}


class AlignCornersResizeBilinear(TestModuleBase):
    """Exercise the align-corners coordinate mode."""

    def __init__(self):
        """Create a fixed align-corners bilinear resize module."""
        super().__init__()
        self.resize = ResizeBilinear2d(
            (6, 8),
            align_corners=True,
            half_pixel_centers=False,
        )

    def forward(self, input_):
        """Resize the input tensor."""
        return self.resize(input_)

    def get_example_inputs(self):
        """Return a representative static input."""
        return (torch.randn(1, 2, 3, 4),), {}
