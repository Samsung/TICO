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

import numpy as np

from tico.circle._schema import decode_text
from tico.circle.operations import extract_by_operator_indices

from test.support.circle.builder import CircleModelBuilder
from test.support.circle.value_test import CircleValueTestCase


class CircleExtractionValueTest(CircleValueTestCase):
    """Check extracted graph values against source intermediate tensors."""

    def test_middle_operator_reproduces_source_boundary_value(self):
        """Feed a captured source intermediate into an extracted middle region."""

        builder = CircleModelBuilder(description="extract-value-test")
        x = builder.input("x", [3])
        one = builder.const_f32("one", 1.0)
        added = builder.add(x, one, name="added")
        two = builder.const_f32("two", 2.0)
        multiplied = builder.mul(added, two, name="multiplied")
        three = builder.const_f32("three", 3.0)
        output = builder.add(multiplied, three, name="output")
        builder.set_outputs(output)
        source = builder.build()

        input_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected_source = (input_value + np.float32(1.0)) * np.float32(
            2.0
        ) + np.float32(3.0)
        result = self.assert_extraction_preserves_value(
            source,
            (input_value,),
            lambda document: extract_by_operator_indices(document, (1,)),
            expected_source_outputs=(expected_source,),
        )

        extracted = result.document
        self.assertEqual(result.selected_operator_indices, (1,))
        self.assertEqual(result.source_boundary.inputs, (added,))
        self.assertEqual(result.source_boundary.outputs, (multiplied,))
        self.assertEqual(len(extracted.subgraph().operators), 1)
        self.assertEqual(len(extracted.subgraph().inputs), 1)
        self.assertEqual(len(extracted.subgraph().outputs), 1)
        self.assertEqual(len(extracted.model.buffers), 2)
        input_tensor = extracted.subgraph().tensors[extracted.subgraph().inputs[0]]
        constant_tensor = next(
            tensor
            for tensor in extracted.subgraph().tensors
            if decode_text(tensor.name) == "two"
        )
        self.assertEqual(decode_text(input_tensor.name), "added")
        self.assertEqual(constant_tensor.buffer, 1)
        constant_index = next(
            index
            for index, tensor in enumerate(extracted.subgraph().tensors)
            if tensor is constant_tensor
        )
        self.assertNotIn(constant_index, list(extracted.subgraph().inputs))

    def test_multi_output_region_matches_two_source_intermediates(self):
        """Preserve output order for an extracted fan-out region."""

        builder = CircleModelBuilder(description="extract-multi-output-value-test")
        x = builder.input("x", [3])
        one = builder.const_f32("one", 1.0)
        shared = builder.add(x, one, name="shared")
        two = builder.const_f32("two", 2.0)
        doubled = builder.mul(shared, two, name="doubled")
        four = builder.const_f32("four", 4.0)
        shifted = builder.add(shared, four, name="shifted")
        builder.set_outputs(doubled, shifted)
        source = builder.build()

        input_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.assert_extraction_preserves_value(
            source,
            (input_value,),
            lambda document: extract_by_operator_indices(document, (1, 2)),
            expected_source_outputs=(
                (input_value + np.float32(1.0)) * np.float32(2.0),
                input_value + np.float32(5.0),
            ),
        )

        self.assertEqual(result.source_boundary.inputs, (shared,))
        self.assertEqual(result.source_boundary.outputs, (doubled, shifted))
        self.assertEqual(len(result.document.subgraph().outputs), 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
