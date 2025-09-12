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

from tico.config.v1 import CompileConfigV1

from test.modules.base import TestModuleBase
from test.utils.tag import test_negative, use_onert


class SimpleMatmul(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs):
        out = torch.mm(lhs, rhs)
        return out

    def get_example_inputs(self):
        return (torch.randn(3, 4), torch.randn(4, 5)), {}


class SimpleMatmulConstRhs(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 5)

    def forward(self, lhs):
        out = torch.mm(lhs, self.weight)
        return out

    def get_example_inputs(self):
        return (torch.randn(3, 4),), {}


@use_onert
class SimpleMatmulConstRhsOnert(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 5)

    def forward(self, lhs):
        out = torch.mm(lhs, self.weight)
        return out

    def get_example_inputs(self):
        return (torch.randn(3, 4),), {}


@use_onert
@test_negative(expected_err="NNFW_STATUS_ERROR")
class SimpleMatmulConstLhsOnert(TestModuleBase):
    """ """

    def __init__(self):
        super().__init__()
        self.weight = torch.randn(3, 4)

    def forward(self, rhs):
        out = torch.mm(self.weight, rhs)
        return out

    def get_example_inputs(self):
        return (torch.randn(4, 5),), {}


@use_onert
class SimpleMatmulConstLhsOnertWithLinearConversion(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(3, 4)

    def forward(self, rhs):
        out = torch.mm(self.weight, rhs)
        return out

    def get_example_inputs(self):
        return (torch.randn(4, 5),), {}

    def get_compile_config(self):
        return CompileConfigV1(convert_lhs_const_mm_to_fc=True)
