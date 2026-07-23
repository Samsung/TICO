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

from tico.circle.errors import CircleSelectionError
from tico.circle.selector import (
    parse_operator_spec,
    resolve_tensor_patterns,
    select_operators_by_tensor_boundaries,
)

from test.unit_test.circle.fixture import make_test_document


class CircleSelectorTest(unittest.TestCase):
    def test_parse_operator_spec(self):
        self.assertEqual(parse_operator_spec("0-2,5,7:8"), (0, 1, 2, 5, 7, 8))

    def test_invalid_range_is_rejected(self):
        with self.assertRaisesRegex(CircleSelectionError, "greater than"):
            parse_operator_spec("3-1")

    def test_tensor_patterns_select_graph_paths(self):
        graph = make_test_document().graph(0)
        starts = resolve_tensor_patterns(graph, ("^x$",))
        ends = resolve_tensor_patterns(graph, ("output$",))

        selected = select_operators_by_tensor_boundaries(
            graph,
            from_tensors=starts,
            to_tensors=ends,
        )

        self.assertEqual(selected, (0, 1))

    def test_forward_only_boundary_selects_all_reachable_operators(self):
        graph = make_test_document().graph(0)

        selected = select_operators_by_tensor_boundaries(graph, from_tensors=(0,))

        self.assertEqual(selected, (0, 1))

    def test_backward_only_boundary_selects_required_operators(self):
        graph = make_test_document().graph(0)

        selected = select_operators_by_tensor_boundaries(graph, to_tensors=(5,))

        self.assertEqual(selected, (0, 1))

    def test_unmatched_pattern_is_rejected(self):
        graph = make_test_document().graph(0)

        with self.assertRaisesRegex(CircleSelectionError, "did not match"):
            resolve_tensor_patterns(graph, ("missing",))
