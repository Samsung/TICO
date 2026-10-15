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

import numpy as np
from circle_schema import circle

from tico.circle.document import CircleDocument
from tico.circle.inspect import format_document, summarize_document
from tico.circle.runtime import CircleReferenceRuntime

from test.support.circle.builder import CircleModelBuilder
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

    def test_builtin_operator_names_match_runtime_after_roundtrip(self):
        """Inspect legacy and modern codes as the operator actually executed."""

        mul = circle.BuiltinOperator.BuiltinOperator.MUL
        placeholder = (
            circle.BuiltinOperator.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
        )
        for builtin, deprecated in ((0, mul), (placeholder, mul), (mul, 0)):
            with self.subTest(builtin=builtin, deprecated=deprecated):
                builder = CircleModelBuilder()
                x = builder.input("x", [2])
                y = builder.input("y", [2])
                output = builder.mul(x, y, name="product")
                builder.set_outputs(output)
                code = builder.model.operatorCodes[0]
                code.builtinCode = builtin
                code.deprecatedBuiltinCode = deprecated
                document = CircleDocument.from_bytes(builder.build().to_bytes())
                self.assertTrue(document.verify(raise_on_error=False).ok)

                runtime = CircleReferenceRuntime(document)
                self.assertEqual(runtime.program.operators[0].name, "MUL")
                result = runtime.run(
                    (
                        np.array([2.0, -3.0], dtype=np.float32),
                        np.array([3.0, 4.0], dtype=np.float32),
                    )
                )
                np.testing.assert_array_equal(
                    result.outputs[0], np.array([6.0, -12.0], dtype=np.float32)
                )
                self.assertEqual(result.outputs[0].dtype, np.float32)
                text = format_document(document, include_operators=True)
                self.assertIn("[0] MUL (opcode=0): inputs=[0, 1], outputs=[2]", text)

    def test_custom_operator_name_takes_precedence_over_builtin_code(self):
        document = make_test_document()
        document.model.operatorCodes[0].customCode = b"vendor.custom"

        text = format_document(document, subgraph_index=0, include_operators=True)

        self.assertIn("[0] vendor.custom (opcode=0)", text)

    def test_invalid_opcode_indices_remain_printable(self):
        document = make_test_document()
        for opcode_index in (-1, len(document.model.operatorCodes)):
            with self.subTest(opcode_index=opcode_index):
                document.subgraph().operators[0].opcodeIndex = opcode_index

                text = format_document(
                    document, subgraph_index=0, include_operators=True
                )

                self.assertIn(
                    f"[0] <invalid:{opcode_index}> (opcode={opcode_index})", text
                )

    def test_unknown_builtin_codes_remain_numeric(self):
        document = make_test_document()
        code = document.model.operatorCodes[0]
        code.deprecatedBuiltinCode = circle.BuiltinOperator.BuiltinOperator.MUL
        for builtin_code in (-1, 999):
            with self.subTest(builtin_code=builtin_code):
                code.builtinCode = builtin_code

                text = format_document(
                    document, subgraph_index=0, include_operators=True
                )

                self.assertIn(f"[0] {builtin_code} (opcode=0)", text)
