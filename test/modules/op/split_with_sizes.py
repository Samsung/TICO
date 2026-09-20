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

from test.modules.base import TestModuleBase


class SimpleSplitWithSizes(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_, split_size):
        result = torch.split_with_sizes(input_, split_size)
        return result

    def get_example_inputs(self):
        return (
            torch.randn(3, 4),
            (1, 2),
        ), {}


class SimpleSplitWithSizesWithDim1(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_, split_size, dim):
        result = torch.split_with_sizes(input_, split_size, dim)
        return (result[0], result[2])

    def get_example_inputs(self):
        return (
            torch.randn(3, 7, 3),
            (2, 4, 1),
            1,
        ), {}


class SimpleSplitWithSizesCopy(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_, split_size):
        torch.split_with_sizes_copy(input_, split_size)
        return None

    def get_example_inputs(self):
        return (
            torch.randn(3, 4),
            (1, 2),
        ), {}


class SimpleSplitWithSizesCopyWithDim1(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_, split_size, dim):
        torch.split_with_sizes_copy(input_, split_size, dim)
        return None

    def get_example_inputs(self):
        return (
            torch.randn(3, 7, 3),
            (2, 4, 1),
            1,
        ), {}
