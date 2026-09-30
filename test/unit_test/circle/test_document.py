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

import copy
import unittest
from unittest.mock import Mock

import numpy as np

from tico.circle.document import CircleDocument
from tico.circle.errors import CircleIOError

from test.unit_test.circle.fixture import make_test_document


class CircleDocumentTest(unittest.TestCase):
    def test_clone_is_independent(self):
        document = make_test_document()

        clone = document.clone()
        clone.subgraph(0).name = "changed"

        self.assertEqual(document.subgraph(0).name, "primary")
        self.assertEqual(clone.subgraph(0).name, "changed")

    def test_clone_copies_payload_views_and_drops_the_mapping(self):
        document = make_test_document()
        base = np.arange(32, dtype=np.uint8)
        document.model.buffers[1].data = base[:8]
        mapping = Mock()
        mapped = CircleDocument(document.model, payload_mapping=mapping)

        clone = mapped.clone()
        deep = copy.deepcopy(mapped)

        for copied in (clone, deep):
            self.assertIsNone(copied.payload_mapping)
            payload = copied.model.buffers[1].data
            self.assertIsNot(payload, base)
            self.assertIsNone(payload.base)
            self.assertEqual(payload.tobytes(), bytes(range(8)))
        base[0] = 99
        self.assertEqual(clone.model.buffers[1].data[0], 0)
        mapping.release.assert_not_called()

    def test_release_payloads_and_context_manager_release_once(self):
        mapping = Mock()
        document = CircleDocument(make_test_document().model, payload_mapping=mapping)
        self.assertIs(document.payload_mapping, mapping)

        with document as entered:
            self.assertIs(entered, document)
        mapping.release.assert_called_once()
        self.assertIsNone(document.payload_mapping)
        document.release_payloads()
        mapping.release.assert_called_once()

        plain = make_test_document()
        plain.release_payloads()  # No mapping: a harmless no-op.
        self.assertIsNone(plain.payload_mapping)

    def test_non_atomic_save_over_mapped_source_is_refused_before_writing(self):
        mapping = Mock()
        mapping.path = "model.circle"
        mapping.same_file.side_effect = lambda path: str(path) == "model.circle"
        document = CircleDocument(make_test_document().model, payload_mapping=mapping)

        with unittest.mock.patch("tico.circle.document.save_model") as save:
            with self.assertRaisesRegex(CircleIOError, "Refusing to overwrite"):
                document.save("model.circle", atomic=False)
            save.assert_not_called()
            document.save("model.circle")
            document.save("other.circle", atomic=False)
            document.save("-", atomic=False)
        self.assertEqual(save.call_count, 3)

    def test_subgraph_bounds_are_checked(self):
        document = make_test_document()

        with self.assertRaisesRegex(IndexError, "Subgraph index 2"):
            document.subgraph(2)

    def test_model_without_subgraphs_field_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "subgraphs"):
            CircleDocument(object())
