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


class SimpleConvTranspose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv2d = torch.nn.ConvTranspose2d(16, 33, 3, stride=2)

    def forward(self, input):
        return self.tconv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 16, 50, 100),)


class ConvTSamePad(torch.nn.Module):
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
        return (torch.randn(1, 16, 32, 32),)


class ConvTUpsample2x(torch.nn.Module):
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
        return (torch.randn(1, 8, 16, 16),)


class ConvTStride2OutPad1(torch.nn.Module):
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
        return (torch.randn(1, 12, 15, 15),)
