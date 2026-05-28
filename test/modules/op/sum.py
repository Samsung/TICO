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

from test.support.tag import skip


class SimpleSumDimMinus1(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        result = x.sum(dim=-1)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 5),), {}


class SimpleSumDimMinus1With3D(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        result = x.sum(dim=-1)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 5, 2),), {}


class SimpleSumDim2Keepdim(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        result = x.sum(dim=2, keepdim=True)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 5, 2),), {}


# Sum with int type is not supported by luci-interpreter.
# Therefore, the following test case should be skipped.
# It will be enabled once it is supported.
@skip(reason="Not supported yet")
class SimpleSumDim2Int(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        result = x.sum(dim=2)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 5, 2).to(torch.int32),), {}
