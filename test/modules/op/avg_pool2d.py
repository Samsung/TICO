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


class SimpleAvgPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


class AdaptiveAvgPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 128, 7, 7),)

    # This test should be done after applying NCHW_to_NHWC with onecc.
    # def get_calibration_data(self):
    #     calibration_data = [self.get_example_inputs() for _ in range(100)]
    #     return calibration_data


class AvgPoolWithPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


class AvgPoolWithSamePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


class AvgPoolWithoutStride(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3)

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 17),)


class AvgPoolNonSquareWindow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(3, 2), stride=(2, 1))

    def forward(self, tensor):
        result = self.avgpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)
