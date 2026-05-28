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
from tico.passes import ops
from tico.passes.decompose_grouped_conv2d import DecomposeGroupedConv2d

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class SimpleGroupedConv2dNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            groups=2,
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),), {}


class DecomposeGroupedConv2dTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(SimpleGroupedConv2dNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)

        self.run_value_test(DecomposeGroupedConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 2)


class SimpleGroupedConv2dWithPaddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            groups=2,
            padding="same",
        )

    def forward(self, input):
        return self.conv2d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 64, 64),), {}


class DecomposeGroupedConv2dWithPaddingTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(SimpleGroupedConv2dWithPaddingNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)

        self.run_value_test(DecomposeGroupedConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 2)
