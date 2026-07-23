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

from tico.circle.inspect import format_document, summarize_document

from test.unit_test.circle.fixture import make_test_document


class CircleInspectTest(unittest.TestCase):
    def test_summary_reports_multi_subgraph_and_payload_counts(self):
        summary = summarize_document(make_test_document())

        self.assertEqual(len(summary.subgraphs), 2)
        self.assertEqual(summary.buffers, 4)
        self.assertEqual(summary.buffers_with_payload, 2)
        self.assertEqual(summary.signatures, 2)

    def test_detailed_formatter_includes_tensor_and_operator_sections(self):
        text = format_document(
            make_test_document(),
            subgraph_index=0,
            include_tensors=True,
            include_operators=True,
        )

        self.assertIn("Subgraph 0 details", text)
        self.assertIn("shared_weight", text)
        self.assertIn("Operators", text)
