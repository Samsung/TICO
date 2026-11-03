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

from tico.circle.passes.optimization._pattern_utils import operator_is_live

from test.unit_test.circle.fixture import FakeOperator, make_test_document


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


class CircleGraphCachingTest(unittest.TestCase):
    def test_operator_inputs_and_outputs_are_memoized(self):
        graph = make_test_document().graph(0)

        inputs_first = graph.operator_inputs(1)
        self.assertEqual(inputs_first, (2, 1))
        self.assertIs(inputs_first, graph.operator_inputs(1))

        outputs_first = graph.operator_outputs(1)
        self.assertEqual(outputs_first, (5,))
        self.assertIs(outputs_first, graph.operator_outputs(1))

    def test_operator_snapshot_is_stable_after_construction(self):
        document = make_test_document()
        graph = document.graph(0)

        count_before = graph.operator_count
        first_operator = graph._operator(0)

        # Mutating the underlying subgraph after construction must not leak
        # into the revision-pinned snapshot held by this CircleGraph instance.
        document.model.subgraphs[0].operators.append(
            FakeOperator(opcodeIndex=0, inputs=[0], outputs=[4])
        )

        self.assertEqual(graph.operator_count, count_before)
        self.assertIs(graph._operator(0), first_operator)

    def test_live_operators_matches_backward_and_is_cached_immutable(self):
        graph = make_test_document().graph(0)

        live = graph.live_operators()
        self.assertEqual(live, {0, 1})
        self.assertEqual(live, graph.backward_operators(graph.outputs))
        self.assertIsInstance(live, frozenset)
        self.assertIs(live, graph.live_operators())
        with self.assertRaises(AttributeError):
            live.add(2)  # type: ignore[attr-defined]

    def test_operator_is_live_reports_dead_branch(self):
        graph = make_test_document().graph(0)

        self.assertTrue(operator_is_live(graph, 0))
        self.assertTrue(operator_is_live(graph, 1))
        self.assertFalse(operator_is_live(graph, 2))
