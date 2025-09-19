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


class SimpleBatchMatMul(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = torch.bmm(x, y)
        return z

    def get_example_inputs(self):
        return (torch.randn(2, 4, 5), torch.randn(2, 5, 3)), {}


@use_onert
class SimpleSingleBatchLhsConstBmm(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.const_lhs = torch.randn(1, 4, 5)

    def forward(self, rhs):
        z = torch.bmm(self.const_lhs, rhs)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 5, 3),), {}

    def get_compile_config(self):
        return CompileConfigV1(convert_single_batch_lhs_const_bmm_to_fc=True)


@use_onert
@test_negative(expected_err="NNFW_STATUS_ERROR")
class SimpleSingleBatchLhsConstBmm_NEG(TestModuleBase):
    """
    Without CompileConfigV1(convert_single_batch_lhs_const_bmm_to_fc=True), it fails
    """

    def __init__(self):
        super().__init__()
        self.const_lhs = torch.randn(1, 4, 5)

    def forward(self, rhs):
        z = torch.bmm(self.const_lhs, rhs)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 5, 3),), {}
