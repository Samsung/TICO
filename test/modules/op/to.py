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

from test.utils import tag


@tag.skip(reason="Not Support Operator")
class ToWithMemoryFormat(TestModuleBase):
    """
    TODO
    Not supported operators: aten._to_copy.default
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_channel_last = x.to(dtype=torch.double, memory_format=torch.channels_last)
        return x_channel_last.to(
            dtype=torch.double, memory_format=torch.contiguous_format
        )

    def get_example_inputs(self):
        return (torch.randn(1, 3, 5, 4),), {}


class RedundantDtypeToCopy(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = y.to(torch.float32)
        return x + y

    def get_example_inputs(self):
        return (torch.randn(1, 3, 5, 4), torch.randn(1, 3, 5, 4)), {}


class RedundantDeviceToCopy(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = y.to(device="cpu")
        return x + y

    def get_example_inputs(self):
        return (torch.randn(1, 3, 5, 4), torch.randn(1, 3, 5, 4)), {}


class ToWithIntegerType(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x.to(torch.int32) + 1
        return y

    def get_example_inputs(self):
        return (torch.randn(1, 3),), {}


class ToWithIntegerPlusFloat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x.to(torch.int32) + 1.0
        return y  # float

    def get_example_inputs(self):
        return (torch.randn(1, 3),), {}


class SimpleToForCast(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.int32)

    def get_example_inputs(self):
        return (torch.randn(1, 3),), {}
