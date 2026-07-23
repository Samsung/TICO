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

import unittest

from test.unit_test.circle.fixture import make_test_document


class CircleGraphTest(unittest.TestCase):
    def test_producer_and_consumer_indices(self):
        graph = make_test_document().graph(0)

        self.assertEqual(graph.producer(2), 0)
        self.assertEqual(graph.producer(0), None)
        self.assertEqual(graph.consumers(1), (0, 1, 2))
        self.assertEqual(graph.predecessors(1), (0,))
        self.assertEqual(graph.successors(0), (1,))

    def test_constant_tensor_detection_uses_payload(self):
        graph = make_test_document().graph(0)

        self.assertTrue(graph.is_constant(1))
        self.assertFalse(graph.is_constant(0))
        self.assertFalse(graph.is_constant(6))

    def test_region_boundary_promotes_external_values(self):
        graph = make_test_document().graph(0)

        first = graph.region_boundary((0,))
        second = graph.region_boundary((1,))
        chain = graph.region_boundary((0, 1))

        self.assertEqual(first.inputs, (0,))
        self.assertEqual(first.outputs, (2,))
        self.assertEqual(second.inputs, (2,))
        self.assertEqual(second.outputs, (5,))
        self.assertEqual(chain.inputs, (0,))
        self.assertEqual(chain.outputs, (5,))

    def test_variable_tensor_is_not_treated_as_constant(self):
        document = make_test_document()
        document.subgraph(0).tensors[1].isVariable = True

        self.assertFalse(document.graph(0).is_constant(1))

    def test_external_buffer_metadata_marks_a_constant(self):
        document = make_test_document()
        document.model.buffers[1].data = b""
        document.model.buffers[1].size = 16

        self.assertTrue(document.graph(0).is_constant(1))

    def test_forward_and_backward_slices(self):
        graph = make_test_document().graph(0)

        self.assertEqual(graph.forward_operators((0,)), {0, 1})
        self.assertEqual(graph.backward_operators((5,)), {0, 1})
        self.assertEqual(graph.forward_operators((3,)), {2})
