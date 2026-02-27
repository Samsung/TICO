# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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
from tico.passes.convert_permute_to_reshape import ConvertPermuteToReshape

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class PermuteBasic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (1, 2, 3, 0))

    def get_example_inputs(self):
        return (torch.rand([1, 5, 1, 3]),), {}


class PermuteBasicTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(PermuteBasic())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 1)
        self.run_value_test(ConvertPermuteToReshape(True))
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 0)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)


class PermuteBasicNegative(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (2, 3, 0, 1))

    def get_example_inputs(self):
        return (torch.rand([1, 5, 1, 3]),), {}


class PermuteBasicNegativeTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(PermuteBasicNegative())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 1)
        self.run_value_test(ConvertPermuteToReshape(True))
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)
