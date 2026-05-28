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
from tico.serialize.circle_graph import CircleModel, CircleSubgraph, is_const


class CircleGraphTest(unittest.TestCase):
    def test_is_const(self):
        self.assertTrue(is_const(1))
        self.assertTrue(is_const(1.1))
        self.assertTrue(is_const([1, 1]))
        self.assertTrue(is_const([0.1, 0.1]))
        self.assertTrue(is_const(torch.tensor(1)))
        self.assertTrue(is_const([torch.tensor(1)]))
        self.assertTrue(is_const([torch.tensor(1), 1]))
        self.assertTrue(is_const([torch.tensor([1, 1])]))

    def test_duplicate_names(self):
        mod = CircleModel()
        g = CircleSubgraph(mod)
        g.add_tensor_from_scratch(
            prefix="name", shape=[1, 2, 3], shape_signature=None, dtype=0
        )
        g.add_tensor_from_scratch(
            prefix="name", shape=[1, 2, 3], shape_signature=None, dtype=0
        )

        self.assertTrue(g.has_tensor("name"))
        # This result depends on the naming rule of _gen_unique_name_with_prefix
        # Change this if the rule changes
        self.assertTrue(g.has_tensor("name_0"))
