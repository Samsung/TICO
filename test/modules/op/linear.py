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


class SimpleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg):
        return self.linear(arg)

    def get_example_inputs(self):
        return (torch.randn(3, 3),)

    # TODO enable this after introducing onert in CI.
    # def get_dynamic_shapes(self):
    #     return {"arg": {0: Dim("batch")}}


class LinearWithDictOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear1(x)
        return {"x": x, "x_minus_1": x - 1}

    def get_example_inputs(self):
        return (torch.randn(2, 10),)


class LinearWithTreeOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear1(x)
        return {"x": x, "x_varients": [x - 1, x + 1]}

    def get_example_inputs(self):
        return (torch.randn(2, 10),)


class LinearWithUnusedInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg, attn_mask):
        return self.linear(arg)

    def get_example_inputs(self):
        return (torch.randn(3, 3), None)
