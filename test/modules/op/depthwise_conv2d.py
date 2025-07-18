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

from test.utils import tag


class SimpleDepthwiseConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, groups=8
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),)


@tag.use_onert
class SimpleDepthwiseConvDynamicShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, groups=8, padding=(2, 2)
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(4, 8, 64, 64),)

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        dynamic_shapes = {
            "input": {0: batch},
        }
        return dynamic_shapes


class SimpleDepthwiseConvWithValidPaddingInStr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=5, out_channels=25, kernel_size=1, groups=5, padding="valid"
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 5, 4, 4),)


class SimpleDepthwiseConvWithValidPaddingInList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=5, out_channels=25, kernel_size=1, groups=5, padding=(0, 0)
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 5, 4, 4),)


class SimpleDepthwiseConvWithSamePaddingInStr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, groups=8, padding="same"
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),)


class SimpleDepthwiseConvWithSamePaddingInList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, groups=8, padding=(1, 1)
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),)


class DepthwiseConvWithTensorWeightBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias):
        groups = 1536
        return torch.nn.functional.conv2d(
            input, weight, bias, padding="same", groups=groups
        )

    def get_example_inputs(self):
        IC = OC = groups = 1536
        return (
            torch.randn(4, IC, 3, 3),
            torch.randn(OC, IC // groups, 1, 1),
            torch.randn(OC),
        )


class DepthwiseConvWithTensorWeightNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight):
        groups = 1536
        return torch.nn.functional.conv2d(input, weight, padding="same", groups=groups)

    def get_example_inputs(self):
        IC = OC = groups = 1536
        return (torch.randn(4, IC, 3, 3), torch.randn(OC, IC // groups, 1, 1))
