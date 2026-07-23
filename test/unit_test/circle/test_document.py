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

from tico.circle.document import CircleDocument

from test.unit_test.circle.fixture import make_test_document


class CircleDocumentTest(unittest.TestCase):
    def test_clone_is_independent(self):
        document = make_test_document()

        clone = document.clone()
        clone.subgraph(0).name = "changed"

        self.assertEqual(document.subgraph(0).name, "primary")
        self.assertEqual(clone.subgraph(0).name, "changed")

    def test_subgraph_bounds_are_checked(self):
        document = make_test_document()

        with self.assertRaisesRegex(IndexError, "Subgraph index 2"):
            document.subgraph(2)

    def test_model_without_subgraphs_field_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "subgraphs"):
            CircleDocument(object())
