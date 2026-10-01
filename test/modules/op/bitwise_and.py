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
from test.support import tag


class SimpleBitwiseAnd(TestModuleBase):
    """`bool & bool` traces to `aten.bitwise_and.Tensor`, not `logical_and`."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x & y
        return z

    def get_example_inputs(self):
        torch.manual_seed(0)
        lhs = torch.randn((3, 5)) < 0.5
        rhs = torch.randn((3, 5)) < 0.5
        return (
            lhs,
            rhs,
        ), {}


class BitwiseAndWithBroadcast(TestModuleBase):
    """Mask-style broadcast: (1, 1, Q, K) & (1, 1, 1, K)."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = torch.bitwise_and(x, y)
        return z

    def get_example_inputs(self):
        torch.manual_seed(0)
        lhs = torch.randn((1, 1, 4, 6)) < 0.0
        rhs = torch.randn((1, 1, 1, 6)) < 0.0
        return (
            lhs,
            rhs,
        ), {}


@tag.test_negative(
    expected_err="aten.bitwise_and.Tensor is only supported for bool operands"
)
class BitwiseAndWithIntOperands(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = torch.bitwise_and(x, y)
        return z

    def get_example_inputs(self):
        lhs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        rhs = torch.tensor([[3, 3, 3], [4, 4, 4]], dtype=torch.int32)
        return (
            lhs,
            rhs,
        ), {}
