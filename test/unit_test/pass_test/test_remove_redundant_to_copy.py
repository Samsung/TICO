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

from tico.passes.remove_redundant_to_copy import RemoveRedundantToCopy

from test.modules.op.to import RedundantDeviceToCopy, RedundantDtypeToCopy
from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class RemoveRedundantDtypeToCopyPassTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantDtypeToCopy())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.to_copy), 1)

        self.run_value_test(RemoveRedundantToCopy())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.to_copy), 0)


class RemoveRedundantDeviceToCopyPassTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantDeviceToCopy())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.to_copy), 1)

        self.run_value_test(RemoveRedundantToCopy())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.to_copy), 0)


# Do not support memory format conversion yet.
class NonRedundantToCopyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = y.to(dtype=torch.float32, memory_format=torch.channels_last)
        return x + y

    def get_example_inputs(self):
        return (torch.randn(1, 3, 5, 4), torch.randn(1, 3, 5, 4))


class NonRedundantToCopyTest(SinglePassValueTest):
    def test_pass_neg(self):
        self.setup(NonRedundantToCopyNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.to_copy), 1)

        self.run_value_test(RemoveRedundantToCopy())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.to_copy), 1)
