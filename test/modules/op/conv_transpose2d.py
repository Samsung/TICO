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

import torch
from torch.export import Dim

from test.modules.base import TestModuleBase

from test.utils.tag import skip, use_onert


class SimpleConvTranspose(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.tconv2d = torch.nn.ConvTranspose2d(16, 33, 3, stride=2)

    def forward(self, input):
        return self.tconv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 16, 50, 100),), {}


@skip(
    reason="luci-interpreter does not support dynamic shape yet && onert does not support TransposeConv yet"
)
@use_onert
class SimpleConvTransposeDynamicShape(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.tconv2d = torch.nn.ConvTranspose2d(16, 33, 3, stride=2)

    def forward(self, input):
        return self.tconv2d(input)

    def get_example_inputs(self):
        return (torch.randn(4, 16, 50, 100),), {}

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        dynamic_shapes = {
            "input": {0: batch},
        }
        return dynamic_shapes


class ConvTSamePad(TestModuleBase):
    """
    Basic Transposed-Conv: output spatial size == input size.
    """

    def __init__(self):
        super().__init__()
        self.tconv2d = torch.nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,  # “same” padding
            bias=True,
        )

    def forward(self, x):
        return self.tconv2d(x)

    def get_example_inputs(self):
        return (torch.randn(1, 16, 32, 32),), {}


class ConvTUpsample2x(TestModuleBase):
    """
    Doubles height/width with a kernel 4 x 4 and stride 2.
    """

    def __init__(self):
        super().__init__()
        self.tconv2d = torch.nn.ConvTranspose2d(
            in_channels=8,
            out_channels=8,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        return self.tconv2d(x)

    def get_example_inputs(self):
        # (1, 8, 16, 16) → (1, 8, 32, 32)
        return (torch.randn(1, 8, 16, 16),), {}


class ConvTStride2OutPad1(TestModuleBase):
    """
    Produces an odd output size: H_out = H_in*2 + 1.
    """

    def __init__(self):
        super().__init__()
        self.tconv2d = torch.nn.ConvTranspose2d(
            in_channels=12,
            out_channels=20,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=1,  # makes the final size odd
        )

    def forward(self, x):
        return self.tconv2d(x)

    def get_example_inputs(self):
        # (1, 12, 15, 15) → (1, 20, 31, 31)
        return (torch.randn(1, 12, 15, 15),), {}


class ConvTComplicated(TestModuleBase):
    """
    - in_channels = 16
    - out_channels = 16
    - kernel_size = (1, 3)
    - stride = (1, 2)
    - padding = (0, 1)
    - output_padding = (0, 1)
    """

    def __init__(self):
        super().__init__()
        self.tconv = torch.nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
            output_padding=(0, 1),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tconv(x)

    def get_example_inputs(self):
        return (torch.randn(1, 16, 1, 8),), {}
