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

import numpy as np
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

    def test_model_initializes_reserved_empty_buffer(self):
        model = CircleModel()

        self.assertEqual(len(model.buffers), 1)
        self.assertIsNone(model.buffers[0].data)

    def test_tensor_without_data_uses_reserved_empty_buffer(self):
        model = CircleModel()
        graph = CircleSubgraph(model)
        exported_program = torch.export.export(
            torch.nn.Identity().eval(),
            (torch.ones(1),),
        )
        input_node = next(
            node for node in exported_program.graph.nodes if node.op == "placeholder"
        )

        graph.add_tensor_from_node(input_node)

        tensor = graph.tensors[graph.name_to_tid[input_node.name]]
        self.assertEqual(tensor.buffer, 0)
        self.assertEqual(len(model.buffers), 1)

    def test_tensor_with_data_uses_dedicated_buffer(self):
        model = CircleModel()
        graph = CircleSubgraph(model)
        exported_program = torch.export.export(
            torch.nn.Identity().eval(),
            (torch.ones(1),),
        )
        input_node = next(
            node for node in exported_program.graph.nodes if node.op == "placeholder"
        )
        data = np.ones((1,), dtype=np.float32)

        graph.add_tensor_from_node(input_node, data)

        tensor = graph.tensors[graph.name_to_tid[input_node.name]]
        self.assertEqual(tensor.buffer, 1)
        self.assertEqual(len(model.buffers), 2)
        self.assertEqual(bytes(model.buffers[1].data), data.tobytes())

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
        self.assertEqual(len(mod.buffers), 1)
        self.assertEqual([tensor.buffer for tensor in g.tensors], [0, 0])
