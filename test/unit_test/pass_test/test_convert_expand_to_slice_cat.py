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
from packaging.version import Version

from tico.passes import ops
from tico.passes.convert_expand_to_slice_cat import ConvertExpandToSliceCat
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.remove_nop import RemoveNop

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class KVCacheExpandNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.reshape(x, [1, 8, 1, 5, 64])
        x = x.expand([1, 8, 4, 5, 64])
        x = torch.reshape(x, [32, 5, 64])
        return x

    def get_example_inputs(self):
        return (torch.rand([1, 8, 5, 64]),), {}


class ConvertKVCacheExpandTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(KVCacheExpandNet())
        self.ep = self.ep.run_decompositions()
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.expand), 1)

        self.run_value_test(ConvertLayoutOpToReshape())
        self.run_pass(RemoveNop())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)

        self.run_value_test(ConvertExpandToSliceCat(True))
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.expand), 0)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 8)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 9)


class NonKVCacheExpandNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.reshape(x, [1, 1, 5, 64])
        x = x.expand([8, 1, 5, 64])
        x = torch.reshape(x, [8, 5, 64])
        return x

    def get_example_inputs(self):
        return (torch.rand([1, 5, 64]),), {}


class ConvertNonKVCacheExpandTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(NonKVCacheExpandNet())
        self.ep = self.ep.run_decompositions()
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.expand), 1)

        self.run_value_test(ConvertLayoutOpToReshape())
        self.run_pass(RemoveNop())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)

        self.run_value_test(ConvertExpandToSliceCat(True))
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 2)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.expand), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.slice), 0)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.cat), 0)
