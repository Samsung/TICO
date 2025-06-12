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
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.fuse_scalar_mul_into_linear import FuseScalarMulIntoLinear
from tico.passes.ops import aten

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class LeadingScalarMulNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)

    def forward(self, x):
        a = self.linear(x)
        b = torch.reshape(a, shape=(16, 4, 8))
        c = torch.permute(b, dims=(1, 0, 2))
        d = c * 0.24
        return d

    def get_example_inputs(self):
        return (torch.randn(16, 1, 32),)


class LeadingScalarMulNetTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(LeadingScalarMulNet())
        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), aten.mul_tensor), 1)
        self.assertEqual(num_of_ops(self.exported_program(), aten.reshape), 1)
        self.assertEqual(num_of_ops(self.exported_program(), aten.permute), 1)
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.linear.default]), 1
        )

        self.run_value_test(FuseScalarMulIntoLinear())
        self.assertEqual(num_of_ops(self.exported_program(), aten.mul_tensor), 0)
        self.assertEqual(num_of_ops(self.exported_program(), aten.reshape), 1)
        self.assertEqual(num_of_ops(self.exported_program(), aten.permute), 1)
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.linear.default]), 1
        )
