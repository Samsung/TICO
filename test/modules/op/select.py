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


class SimpleSelect(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        dim = 0
        idx = 1
        # Equivalent to `tensor[idx]`
        selected_x = torch.select(x, dim, idx)
        return selected_x

    def get_example_inputs(self):
        return (torch.randn(4),), {}


class SimpleSelect2(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        dim = 2
        idx = 1
        # Equivalent to `tensor[:,:,idx]`
        selected_x = torch.select(x, dim, idx)
        return selected_x

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4),), {}


class SimpleConstIndex(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, discrete_A):
        i = 1
        ssm_state = discrete_A[:, :, i, :]
        return ssm_state

    def get_example_inputs(self):
        return (torch.rand(1, 32, 6, 16),), {}


class NegativeLastSelect(TestModuleBase):
    def forward(self, x: torch.Tensor):
        return torch.select(x, dim=-1, index=-1)

    def get_example_inputs(self):
        return (torch.arange(6, dtype=torch.float32).reshape(2, 3),), {}
