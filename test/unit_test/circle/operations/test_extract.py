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

from tico.circle.operations import (
    extract_by_operator_indices,
    extract_by_tensor_patterns,
    SignaturePolicy,
)

from test.unit_test.circle.fixture import FakeTensor, make_test_document


class CircleExtractionTest(unittest.TestCase):
    def test_extract_single_operator_rebuilds_boundary_and_compacts(self):
        source = make_test_document()

        result = extract_by_operator_indices(source, (0,), subgraph_index=0)

        extracted = result.document
        self.assertEqual(source.subgraph_count, 2)
        self.assertEqual(extracted.subgraph_count, 1)
        self.assertEqual(result.source_boundary.inputs, (0,))
        self.assertEqual(result.source_boundary.outputs, (2,))
        self.assertEqual(result.boundary.inputs, (0,))
        self.assertEqual(result.boundary.outputs, (2,))
        self.assertEqual(len(extracted.subgraph(0).operators), 1)
        self.assertEqual(extracted.subgraph(0).inputs, [0])
        self.assertEqual(extracted.subgraph(0).outputs, [2])
        self.assertEqual(
            [tensor.name for tensor in extracted.subgraph(0).tensors],
            ["x", "shared_weight", "add_out"],
        )
        self.assertEqual(len(extracted.model.buffers), 2)
        self.assertEqual(len(extracted.model.operatorCodes), 1)
        self.assertEqual(extracted.model.signatureDefs, [])
        self.assertTrue(extracted.verify(raise_on_error=False).ok)

    def test_extract_tensor_patterns_selects_the_full_live_chain(self):
        result = extract_by_tensor_patterns(
            make_test_document(),
            from_patterns=("^x$",),
            to_patterns=("^output$",),
        )

        self.assertEqual(result.selected_operator_indices, (0, 1))
        self.assertEqual(result.source_boundary.outputs, (5,))
        self.assertEqual(result.boundary.outputs, (3,))
        self.assertEqual(len(result.document.subgraph(0).operators), 2)
        self.assertEqual(result.document.subgraph(0).inputs, [0])
        self.assertEqual(result.document.subgraph(0).outputs, [3])

    def test_compatible_signature_can_be_preserved(self):
        result = extract_by_operator_indices(
            make_test_document(),
            (0, 1),
            signature_policy=SignaturePolicy.PRESERVE_COMPATIBLE,
        )

        self.assertEqual(len(result.document.model.signatureDefs), 1)
        signature = result.document.model.signatureDefs[0]
        self.assertEqual(signature.signatureKey, "primary")
        self.assertEqual(signature.inputs[0].tensorIndex, 0)
        self.assertEqual(signature.outputs[0].tensorIndex, 3)

    def test_keep_other_subgraphs_does_not_prune_their_unused_tensors(self):
        source = make_test_document()
        source.subgraph(1).tensors.append(FakeTensor("secondary_orphan"))

        result = extract_by_operator_indices(
            source,
            (0,),
            keep_other_subgraphs=True,
        )

        self.assertEqual(
            [tensor.name for tensor in result.document.subgraph(1).tensors],
            ["y", "shared_weight_secondary", "secondary_output", "secondary_orphan"],
        )

    def test_keep_other_subgraphs_preserves_shared_weight_buffer(self):
        result = extract_by_operator_indices(
            make_test_document(),
            (0,),
            keep_other_subgraphs=True,
        )

        extracted = result.document
        self.assertEqual(extracted.subgraph_count, 2)
        self.assertEqual(extracted.subgraph(0).tensors[1].buffer, 1)
        self.assertEqual(extracted.subgraph(1).tensors[1].buffer, 1)
        self.assertEqual(len(extracted.model.buffers), 2)
        self.assertEqual(
            [signature.signatureKey for signature in extracted.model.signatureDefs],
            ["secondary"],
        )
