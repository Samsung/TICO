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
from tico.passes.remove_redundant_expand import RemoveRedundantExpand

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class RedundantExpandNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x.expand(x.size())
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 4),), {}


class RedundantExpandWithMinusOneNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Example: x is (3, 4), expand(-1, 4) results in (3, 4)
        z = x.expand(-1, x.size(1))
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 4),), {}


class RemoveRedundantExpandTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantExpandNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.expand), 1)

        self.run_value_test(RemoveRedundantExpand())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.expand), 0)

    def test_pass_with_minus_one(self):
        self.setup(RedundantExpandWithMinusOneNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.expand), 1)

        self.run_value_test(RemoveRedundantExpand())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.expand), 0)
