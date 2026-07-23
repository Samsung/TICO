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

from tico.circle.errors import CircleRewriteError
from tico.circle.rewrite import compact_model, keep_subgraphs

from test.unit_test.circle.fixture import make_test_document


class CircleRewriteTest(unittest.TestCase):
    def test_compaction_remaps_tensors_signatures_and_shared_buffers(self):
        document = make_test_document()
        document.subgraph(0).operators = document.subgraph(0).operators[:2]

        stats = compact_model(document.model)

        self.assertTrue(stats.modified)
        self.assertEqual(len(document.subgraph(0).tensors), 4)
        self.assertEqual(document.subgraph(0).outputs, [3])
        self.assertEqual(document.model.signatureDefs[0].outputs[0].tensorIndex, 3)
        self.assertEqual(len(document.model.buffers), 2)
        self.assertEqual(document.subgraph(1).tensors[1].buffer, 1)
        self.assertEqual(len(document.model.operatorCodes), 2)
        self.assertTrue(document.verify(raise_on_error=False).ok)

    def test_retained_control_flow_reference_to_dropped_subgraph_is_rejected(self):
        document = make_test_document()

        class FakeIfOptions:
            """Provide a control-flow subgraph reference for rewrite tests."""

            thenSubgraphIndex = 1

        document.subgraph(0).operators[0].builtinOptions = FakeIfOptions()

        with self.assertRaisesRegex(CircleRewriteError, "references subgraph 1"):
            keep_subgraphs(document.model, (0,))

    def test_call_options_subgraph_reference_is_rejected(self):
        document = make_test_document()

        class FakeCallOptions:
            """Provide the scalar field used by Circle CallOptions."""

            subgraph = 1

        document.subgraph(0).operators[0].builtinOptions = FakeCallOptions()

        with self.assertRaisesRegex(CircleRewriteError, "references subgraph 1"):
            keep_subgraphs(document.model, (0,))

    def test_vector_subgraph_references_are_remapped(self):
        document = make_test_document()

        class FakeCaseOptions:
            """Provide the vector field used by StablehloCaseOptions."""

            branchSubgraphIndices = [1, 0]

        options = FakeCaseOptions()
        document.subgraph(0).operators[0].builtinOptions = options

        stats = keep_subgraphs(document.model, (1, 0))

        self.assertTrue(stats.modified)
        self.assertEqual(options.branchSubgraphIndices, [0, 1])

    def test_keep_subgraph_remaps_signature_index(self):
        document = make_test_document()

        stats = keep_subgraphs(document.model, (1,))

        self.assertEqual(stats.removed_subgraphs, 1)
        self.assertEqual(len(document.model.subgraphs), 1)
        self.assertEqual(document.model.signatureDefs[0].subgraphIndex, 0)
        self.assertEqual(document.model.signatureDefs[0].signatureKey, "secondary")
