# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from test.modules.base import TestModuleBase


class SimpleConv3d(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(2, 3, 8, 16, 16),), {}


class Conv3dWithPadding(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 4, 10, 32, 32),), {}


class Conv3dWithStride(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 16, 64, 64),), {}


class Conv3dWithDilation(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=2,
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 12, 24, 24),), {}


class Conv3dWithGroups(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, groups=2
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 4, 8, 16, 16),), {}


class Conv3dNoBias(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 8, 16, 16),), {}


class Conv3dNonSquareKernel(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 5, 3),
            stride=(1, 2, 1),
            padding=1,
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 16, 64, 32),), {}


class Conv3dAsymmetricStride(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=3, stride=(1, 2, 2), padding=1
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 16, 64, 64),), {}


class Conv3dWithValidPadding(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="valid"
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 8, 16, 16),), {}


class Conv3dWithSamePadding(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="same"
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 8, 16, 16),), {}


class Qwen2VLConv3dBasic(TestModuleBase):
    """
    Conv3D module similar to Qwen2.5-VL video processing components.
    """

    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3,
            out_channels=1280,
            kernel_size=(2, 14, 14),
            stride=(2, 14, 14),
            padding=(0, 0, 0),
            bias=True,
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 4, 28, 28),), {}


class MultiLayerConv3d(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d_1 = torch.nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv3d_2 = torch.nn.Conv3d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = self.conv3d_1(input)
        x = self.relu(x)
        x = self.conv3d_2(x)
        return x

    def get_example_inputs(self):
        return (torch.randn(1, 3, 8, 16, 16),), {}


class Conv3dLargeKernel(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 12, 32, 32),), {}


class Conv3dWithTensorWeightAndBias(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias):
        return torch.nn.functional.conv3d(input, weight, bias)

    def get_example_inputs(self):
        return (
            torch.randn(1, 3, 8, 16, 16),
            torch.randn(16, 3, 3, 3, 3),
            torch.randn(16),
        ), {}


class Conv3dWithTensorWeightNoBias(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight):
        return torch.nn.functional.conv3d(input, weight)

    def get_example_inputs(self):
        return (
            torch.randn(1, 3, 8, 16, 16),
            torch.randn(16, 3, 3, 3, 3),
        ), {}


class SimpleGroupedConv3d(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(
            in_channels=6, out_channels=6, kernel_size=3, groups=2
        )

    def forward(self, input):
        return self.conv3d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 6, 8, 16, 16),), {}


class GroupedConv3dWithTensorWeightBias(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias):
        groups = 3
        return torch.nn.functional.conv3d(
            input, weight, bias, padding="same", groups=groups
        )

    def get_example_inputs(self):
        IC = OC = 9
        groups = 3
        return (
            torch.randn(2, IC, 8, 16, 16),
            torch.randn(OC, IC // groups, 3, 3, 3),
            torch.randn(OC),
        ), {}


class GroupedConv3dWithTensorWeightNoBias(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight):
        groups = 3
        return torch.nn.functional.conv3d(input, weight, padding="same", groups=groups)

    def get_example_inputs(self):
        IC = OC = 9
        groups = 3
        return (
            torch.randn(2, IC, 8, 16, 16),
            torch.randn(OC, IC // groups, 3, 3, 3),
        ), {}
