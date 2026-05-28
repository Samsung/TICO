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
from tico.passes.convert_conv1d_to_conv2d import ConvertConv1dToConv2d

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class Conv1dPaddingOne(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(10, 3, 128),), {}


class ConvertConv1dPaddingOneTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv1dPaddingOne())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv1d), 1)

        self.run_value_test(ConvertConv1dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class Conv1dPaddingValid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding="valid"
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(10, 3, 128),), {}


class ConvertConv1dPaddingValidTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(Conv1dPaddingValid())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv1d), 1)

        self.run_value_test(ConvertConv1dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)


class DepthwiseConv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 128, 6),), {}


class ConvertDepthwiseConv1dTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(DepthwiseConv1d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv1d), 1)

        self.run_value_test(ConvertConv1dToConv2d())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.conv2d), 1)
