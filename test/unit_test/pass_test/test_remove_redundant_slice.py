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

import unittest

import torch
from tico.passes import ops
from tico.passes.remove_redundant_slice import RemoveRedundantSlice
from tico.utils.torch_compat import export_produces_slice

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class RedundantSliceNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x[0, :]
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 4),)


class RemoveRedundantSliceTest(SinglePassValueTest):
    @unittest.skipUnless(
        export_produces_slice(),
        "Skip when torch doesn't produce redundant slices.",
    )
    def test_pass(self):
        self.setup(RedundantSliceNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 1)

        self.run_value_test(RemoveRedundantSlice())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 0)
