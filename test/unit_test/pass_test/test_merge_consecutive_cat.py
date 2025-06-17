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
from tico.passes.merge_consecutive_cat import MergeConsecutiveCat

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class ConsecutiveCatNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w):
        x = torch.cat([w, w], dim=2)
        y = torch.cat([w, w], dim=2)
        z = torch.cat([x, y], dim=2)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3),)


class MergeConsecutiveCatTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(ConsecutiveCatNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 3)

        self.run_value_test(MergeConsecutiveCat())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 1)


class NotMergedConsecutiveCatNet1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w):
        x = torch.cat([w, w], dim=2)
        y = torch.cat([w, w], dim=2)
        xy = torch.add(x, y)
        z = torch.cat([x, y], dim=2)
        repeat_xy = torch.repeat_interleave(xy, 2, dim=2)
        xyz = torch.add(repeat_xy, z)
        return xyz

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3),)


class NotMergedConsecutiveCatTest1(SinglePassValueTest):
    def test_pass_neg(self):
        self.setup(NotMergedConsecutiveCatNet1())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 3)

        self.run_value_test(MergeConsecutiveCat())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 3)


class NotMergedConsecutiveCatNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w):
        x = torch.cat([w, w], dim=2)
        y = torch.cat([w, w], dim=2)
        z = torch.cat([x, y], dim=1)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3),)


class NotMergedConsecutiveCatTest2(SinglePassValueTest):
    def test_pass_neg(self):
        self.setup(NotMergedConsecutiveCatNet2())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 3)

        self.run_value_test(MergeConsecutiveCat())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 3)
