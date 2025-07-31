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
from tico.passes.lower_pow2_to_mul import LowerPow2ToMul

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class Pow2Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.pow(2)
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 4),), {}


class LowerPow2ToMulTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Pow2Net())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.pow.Tensor_Scalar]), 1
        )

        self.run_value_test(LowerPow2ToMul())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.pow.Tensor_Scalar]), 0
        )
