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


class SimpleMaxPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2)

    def forward(self, tensor):
        result = self.maxpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


@tag.use_onert
class SimpleMaxPoolDynamicShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2)

    def forward(self, tensor):
        result = self.maxpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        dynamic_shapes = {
            "tensor": {0: batch},
        }
        return dynamic_shapes


class MaxPoolWithPadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, tensor):
        result = self.maxpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


class MaxPoolWithSamePadding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, tensor):
        result = self.maxpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


class MaxPoolNoStride(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(3)

    def forward(self, tensor):
        result = self.maxpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


class MaxPoolFunctionalNoStride(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        result = torch.nn.functional.max_pool2d(tensor, kernel_size=3)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


class MaxPoolNonSquareWindow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d((3, 2), stride=(2, 1))

    def forward(self, tensor):
        result = self.maxpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)


@tag.skip(reason="Not Support Operator")
class MaxPoolReturningIndices(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, return_indices=True)

    def forward(self, tensor):
        result = self.maxpool(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 4, 8, 16),)
