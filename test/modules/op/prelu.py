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


class SimplePReLU(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        result = self.prelu(x)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),), {}


class ChannelWisePReLU(TestModuleBase):
    """Exercise PReLU with one learned slope per NCHW channel."""

    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU(num_parameters=2)

    def forward(self, x):
        return self.prelu(x)

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),), {}
