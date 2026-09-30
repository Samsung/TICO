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

import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from tico.circle import _buffer
from tico.circle.operations import (
    extract_by_operator_indices,
    extract_by_tensor_patterns,
    PayloadOwnership,
    SignaturePolicy,
)

from test.unit_test.circle.fixture import (
    FakeBuffer,
    FakeMetadata,
    FakeTensor,
    make_test_document,
)


def make_array_backed_document():
    """Return the fixture with NumPy payload views that share one large base."""

    document = make_test_document()
    base = np.arange(64, dtype=np.uint8)
    document.model.buffers[1].data = base[:13]
    document.model.buffers[2].data = base[13:26]
    return document, base


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


class CircleExtractionPayloadOwnershipTest(unittest.TestCase):
    """Cover metadata-first cloning and retained-payload detachment."""

    def test_default_result_is_detached_and_source_is_unchanged(self):
        source, base = make_array_backed_document()
        source_buffers = list(source.model.buffers)
        source_tensors = list(source.subgraph(0).tensors)
        source_operators = list(source.subgraph(0).operators)
        source_signatures = list(source.model.signatureDefs)

        result = extract_by_operator_indices(source, (0,))

        self.assertIs(result.payload_ownership, PayloadOwnership.DETACHED)
        self.assertEqual(source.subgraph_count, 2)
        self.assertEqual(source.model.buffers, source_buffers)
        self.assertEqual(source.subgraph(0).tensors, source_tensors)
        self.assertEqual(source.subgraph(0).operators, source_operators)
        self.assertEqual(source.model.signatureDefs, source_signatures)
        np.testing.assert_array_equal(base, np.arange(64, dtype=np.uint8))

        retained = result.document.model.buffers[1].data
        self.assertEqual(retained.tobytes(), bytes(range(13)))
        self.assertIsNot(retained, source.model.buffers[1].data)
        self.assertIsNone(retained.base)
        self.assertTrue(retained.flags.writeable)

        # Editing either side never leaks into the other.
        retained[0] = 200
        result.document.subgraph(0).tensors[0].name = "renamed"
        result.document.model.signatureDefs.append("bogus")
        self.assertEqual(base[0], 0)
        self.assertEqual(source.subgraph(0).tensors[0].name, "x")
        self.assertEqual(len(source.model.signatureDefs), 2)
        base[1] = 201
        source.subgraph(0).tensors[1].name = "mutated"
        self.assertEqual(retained[1], 1)
        self.assertEqual(result.document.subgraph(0).tensors[1].name, "shared_weight")

    def test_tensor_pattern_extraction_is_detached_too(self):
        source, base = make_array_backed_document()

        result = extract_by_tensor_patterns(
            source, from_patterns=("^x$",), to_patterns=("^output$",)
        )

        retained = result.document.model.buffers[1].data
        self.assertIsNone(retained.base)
        base[2] = 202
        self.assertEqual(retained[2], 2)

    def test_discarded_payloads_are_never_copied(self):
        source, _base = make_array_backed_document()
        dead_payload = source.model.buffers[2].data

        with patch.object(
            _buffer, "owned_payload_copy", wraps=_buffer.owned_payload_copy
        ) as copy_payload:
            result = extract_by_operator_indices(source, (0,))

        copied = [call.args[0] for call in copy_payload.call_args_list]
        self.assertEqual(len(copied), 1)
        self.assertIs(copied[0], source.model.buffers[1].data)
        self.assertNotIn(id(dead_payload), {id(value) for value in copied})
        self.assertEqual(len(result.document.model.buffers), 2)

    def test_small_result_does_not_pin_the_source_backing_store(self):
        source, base = make_array_backed_document()
        base_ref = weakref.ref(base)

        result = extract_by_operator_indices(source, (0,))
        del source, base
        gc.collect()

        self.assertIsNone(base_ref())
        self.assertEqual(
            result.document.model.buffers[1].data.tobytes(), bytes(range(13))
        )

    def test_borrowed_result_shares_only_retained_storage(self):
        source, base = make_array_backed_document()

        with patch.object(_buffer, "owned_payload_copy") as copy_payload:
            result = extract_by_operator_indices(
                source, (0,), payload_ownership=PayloadOwnership.BORROWED
            )

        copy_payload.assert_not_called()
        self.assertIs(result.payload_ownership, PayloadOwnership.BORROWED)
        self.assertIs(
            result.document.model.buffers[1].data, source.model.buffers[1].data
        )
        # The buffer table and graph metadata are still independent copies.
        self.assertIsNot(result.document.model.buffers[1], source.model.buffers[1])
        self.assertIsNot(result.document.model.buffers[0], source.model.buffers[0])
        result.document.subgraph(0).tensors[0].name = "renamed"
        self.assertEqual(source.subgraph(0).tensors[0].name, "x")
        base[3] = 203
        self.assertEqual(result.document.model.buffers[1].data[3], 203)

    def test_shared_buffer_payload_is_copied_once_across_subgraphs(self):
        source, _base = make_array_backed_document()

        result = extract_by_operator_indices(source, (0,), keep_other_subgraphs=True)

        extracted = result.document
        self.assertEqual(extracted.subgraph(0).tensors[1].buffer, 1)
        self.assertEqual(extracted.subgraph(1).tensors[1].buffer, 1)
        self.assertEqual(len(extracted.model.buffers), 2)
        self.assertIsNone(extracted.model.buffers[1].data.base)

    def test_metadata_buffer_is_retained_and_detached(self):
        source, base = make_array_backed_document()
        source.model.buffers.append(FakeBuffer(data=base[40:50]))
        source.model.metadata = [FakeMetadata("note", 4)]

        result = extract_by_operator_indices(source, (0,))

        metadata = result.document.model.metadata[0]
        retained = result.document.model.buffers[metadata.buffer].data
        self.assertEqual(retained.tobytes(), bytes(range(40, 50)))
        self.assertIsNone(retained.base)
        self.assertTrue(result.document.verify(raise_on_error=False).ok)

    def test_bytes_payloads_remain_valid_and_source_untouched(self):
        source = make_test_document()

        result = extract_by_operator_indices(source, (0,))

        self.assertEqual(result.document.model.buffers[1].data, b"shared-weight")
        self.assertEqual(source.model.buffers[2].data, b"dead-constant")
        self.assertEqual(len(source.model.buffers), 4)

    def test_invalid_ownership_value_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_by_operator_indices(
                make_test_document(), (0,), payload_ownership="shared"  # type: ignore[arg-type]
            )
