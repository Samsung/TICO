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

# With square kernels and equal stride
class SimpleConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16, out_channels=33, kernel_size=3, stride=2
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(20, 16, 50, 100),)


class SimpleQuantizedConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16, out_channels=33, kernel_size=3, stride=2
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 16, 50, 100),)

    def get_calibration_data(self):
        calibration_data = [self.get_example_inputs() for _ in range(100)]
        return calibration_data


class ConvWithNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=3,
            stride=2,
            bias=False,
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(2, 16, 8, 8),)


class ConvWithNoStrideNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=3,
            bias=False,
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(2, 16, 8, 8),)


# With non-square kernels, unequal stride and padding('valid')
class ConvNonSquareKernelValidPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding="valid",
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(20, 16, 50, 100),)


class ConvValidPaddingWithNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding="valid",
            bias=False,
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(20, 16, 50, 100),)


# With non-square kernels, non-stride and padding('same')
class ConvNonSquareKernelSamePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=(3, 5),
            padding="same",
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(20, 16, 50, 100),)


# With non-square kernels, stride and padding(padH, padW)
class ConvStirdePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding=(4, 2),
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(20, 16, 50, 100),)


# With non-square kernels, unequal stride, padding(padH, padW) and dilation
class ConvWithDilation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding=(4, 2),
            dilation=(3, 1),
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(20, 16, 50, 100),)


class TwoConvSameInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding="valid",
        )
        self.conv2d_2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding="valid",
        )

    def forward(self, input):
        return self.conv2d_1(input) + self.conv2d_2(input)

    def get_example_inputs(self):
        return (torch.randn(20, 16, 50, 100),)


# With square kernels, non-stride and padding('same')-> no_padding[0, 0])
class ConvWithSamePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=33,
            kernel_size=(1, 1),
            padding="same",
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(20, 16, 50, 100),)


class ConvWithSamePadding2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 32, 32),)


class ConvWithPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16,
            out_channels=3,
            kernel_size=(1, 1),
            padding=(1, 3),
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 16, 5, 10),)


class ConvWithIntPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=16, out_channels=3, kernel_size=(1, 1), padding=0, stride=2
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 16, 5, 10),)


class ConvWithTensorWeightAndBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias):
        return torch.nn.functional.conv2d(input, weight, bias)

    def get_example_inputs(self):
        return (
            torch.randn(1, 16, 5, 10),
            torch.randn(3, 16, 1, 1),
            torch.randn(3),
        )


class ConvWithTensorWeightNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight):
        return torch.nn.functional.conv2d(input, weight)

    def get_example_inputs(self):
        return (torch.randn(1, 16, 5, 10), torch.randn(3, 16, 1, 1))


class SimpleGroupedConv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            groups=2,
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),)


class SimpleGroupedConv2dWithValidPaddingInStr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            groups=2,
            padding="valid",
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),)


class SimpleGroupedConv2dWithValidPaddingInList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            groups=2,
            padding=(0, 0),
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),)


class SimpleGroupedConv2dWithSamePaddingInStr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            groups=2,
            padding="same",
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),)


class SimpleGroupedConv2dWithSamePaddingInList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            groups=2,
            padding=(1, 1),
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),)


class GroupedConv2dWithTensorWeightBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias):
        groups = 16
        return torch.nn.functional.conv2d(
            input, weight, bias, padding="same", groups=groups
        )

    def get_example_inputs(self):
        IC = OC = 48
        groups = 16
        return (
            torch.randn(4, IC, 32, 32),
            torch.randn(OC, IC // groups, 3, 3),
            torch.randn(OC),
        )


class GroupedConv2dWithTensorWeightNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight):
        groups = 16
        return torch.nn.functional.conv2d(input, weight, padding="same", groups=groups)

    def get_example_inputs(self):
        IC = OC = 48
        groups = 16
        return (
            torch.randn(4, IC, 32, 32),
            torch.randn(OC, IC // groups, 3, 3),
        )


class GroupedConv2dDifferentICAndOCWithTensorWeightBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias):
        groups = 16
        return torch.nn.functional.conv2d(
            input, weight, bias, padding="same", groups=groups
        )

    def get_example_inputs(self):
        IC = 32
        OC = 48
        groups = 16
        return (
            torch.randn(4, IC, 32, 32),
            torch.randn(OC, IC // groups, 3, 3),
            torch.randn(OC),
        )


class GroupedConv2dDifferentICAndOCWithTensorWeightNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight):
        groups = 16
        return torch.nn.functional.conv2d(input, weight, padding="same", groups=groups)

    def get_example_inputs(self):
        IC = 32
        OC = 48
        groups = 16
        return (
            torch.randn(4, IC, 32, 32),
            torch.randn(OC, IC // groups, 3, 3),
        )
