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

from tico.circle.verify import CircleVerificationError, verify_document

from test.unit_test.circle.fixture import make_test_document


class CircleVerificationTest(unittest.TestCase):
    def test_valid_fixture_has_only_cleanup_warnings(self):
        report = verify_document(make_test_document(), raise_on_error=False)

        self.assertTrue(report.ok)
        self.assertTrue(report.warnings)

    def test_invalid_buffer_reference_is_reported(self):
        document = make_test_document()
        document.subgraph(0).tensors[0].buffer = 99

        report = verify_document(document, raise_on_error=False)

        self.assertFalse(report.ok)
        self.assertIn("INVALID_INDEX", {issue.code for issue in report.errors})
        with self.assertRaises(CircleVerificationError):
            verify_document(document)

    def test_multiple_producers_are_rejected(self):
        document = make_test_document()
        document.subgraph(0).operators[2].outputs = [2]

        report = verify_document(document, raise_on_error=False)

        self.assertIn("MULTIPLE_PRODUCERS", {issue.code for issue in report.errors})

    def test_undefined_operator_input_is_rejected(self):
        document = make_test_document()
        document.subgraph(0).operators[0].inputs = [6, 1]

        report = verify_document(document, raise_on_error=False)

        self.assertIn("UNDEFINED_INPUT", {issue.code for issue in report.errors})

    def test_buffer_zero_payload_is_rejected(self):
        document = make_test_document()
        document.model.buffers[0].data = b"reserved"

        report = verify_document(document, raise_on_error=False)

        self.assertIn("BUFFER_ZERO_NOT_EMPTY", {issue.code for issue in report.errors})

    def test_signature_mapping_must_reference_graph_io(self):
        document = make_test_document()
        document.model.signatureDefs[0].inputs[0].tensorIndex = 2

        report = verify_document(document, raise_on_error=False)

        self.assertIn("SIGNATURE_IO_MISMATCH", {issue.code for issue in report.errors})
