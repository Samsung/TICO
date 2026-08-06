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


class SimpleClampMinOnly(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, -11.0, None)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 20,), {}


class SimpleClampMaxOnly(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, None, 11.0)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 20,), {}


class SimpleClampMinMaxBoth(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, -11.0, 11.0)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 20,), {}


class SimpleClampFrom0to6Float(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, 0.0, 6.0)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 10,), {}


class SimpleClampFrom0to6Int(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, 0, 6)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 10,), {}


class DoubleClampsFrom0to6(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, 0.0)
        x = torch.clamp(x, None, 6.0)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 10,), {}


class DoubleClampsFrom0to6WithBigMax(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, 0.0, 100.0)
        x = torch.clamp(x, None, 6.0)
        return x

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 10,), {}


class SimpleClampTensorMinMax(TestModuleBase):
    """Clamp with scalar tensor buffers as used by Gemma4ClippableLinear."""

    def __init__(self):
        super().__init__()
        self.register_buffer("min_value", torch.tensor(-11.0))
        self.register_buffer("max_value", torch.tensor(11.0))

    def forward(self, x):
        return torch.clamp(x, self.min_value, self.max_value)

    def get_example_inputs(self):
        return (torch.randn(5, 3) * 20,), {}


class ClampIntInputFloatMinMax(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, -100.0, 100.0)
        return x

    def get_example_inputs(self):
        return (torch.randint(-200, 200, (5, 3)),), {}
