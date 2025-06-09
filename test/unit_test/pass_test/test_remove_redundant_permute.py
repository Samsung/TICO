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
from tico.passes.remove_redundant_permute import (
    passes as RemoveRedundantPermutePasses,
    RemoveRedundantPermutePattern1,
)

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import MultiPassValueTest, SinglePassValueTest


class RedundantPermuteNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.permute(x, dims=(2, 1, 0))
        z = torch.permute(y, dims=(2, 1, 0))
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 4, 5),)


class RemoveRedundantPermuteTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(RedundantPermuteNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 2)

        self.run_value_test(RemoveRedundantPermutePattern1())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 0)


class RedundantPermuteFusedNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # AxBxC -> CxBxA
        y = torch.permute(x, dims=(2, 1, 0))
        # CxBxA -> CxAxB
        z = torch.permute(y, dims=(0, 2, 1))
        return z

    def get_example_inputs(self):
        return (torch.randn(2, 2, 2),)


class RemoveRedundantPermuteFusedTest(MultiPassValueTest):
    def test_pass(self):
        self.setup(RedundantPermuteFusedNet())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 2)

        self.run_value_test(RemoveRedundantPermutePasses())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.permute), 1)
        for n in self.exported_program().graph.nodes:
            if n.target == torch.ops.aten.permute.default:
                self.assertEqual(n.args[1], [2, 0, 1])
                break
