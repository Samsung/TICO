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

from tico.circle.passes import CirclePassContext, CirclePassManager, SimplifyViewOpsPass
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass

from test.support.circle.builder import CircleModelBuilder
from test.support.circle.value_test import CircleValueTestCase


class CircleLayoutOpsValueTest(CircleValueTestCase):
    """Check numerical equivalence of redundant layout rewrites."""

    def test_inverse_three_cycle_transposes_are_removed(self):
        """Remove inverse transposes whose permutations are not self-inverse."""

        builder = CircleModelBuilder(description="transpose-value-test")
        x = builder.input("x", [2, 3, 4])
        first = builder.transpose(x, [1, 2, 0], name="first_transpose")
        output = builder.transpose(first, [2, 0, 1], name="output")
        builder.set_outputs(output)
        source = builder.build()

        pipeline = CirclePassManager(
            [
                SimplifyViewOpsPass(),
                DeadCodeEliminationPass(),
                CompactIndicesPass(),
            ]
        )
        input_value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: pipeline.run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(input_value,),
        )

        transformed = result.document
        self.assertEqual(transformed.subgraph().operators, [])
        self.assertEqual(transformed.subgraph().inputs, [0])
        self.assertEqual(transformed.subgraph().outputs, [0])
        self.assertEqual(len(transformed.subgraph().tensors), 1)
        self.assertEqual(len(transformed.model.buffers), 1)
        self.assertEqual(len(transformed.model.operatorCodes), 0)

    def test_consecutive_reshapes_are_reduced_to_the_last_shape(self):
        """Bypass the first reshape while preserving the final tensor value."""

        builder = CircleModelBuilder(description="reshape-value-test")
        x = builder.input("x", [2, 3, 4])
        first = builder.reshape(x, [6, 4], name="first_reshape")
        output = builder.reshape(first, [4, 6], name="output")
        builder.set_outputs(output)
        source = builder.build()

        pipeline = CirclePassManager(
            [
                SimplifyViewOpsPass(),
                DeadCodeEliminationPass(),
                CompactIndicesPass(),
            ]
        )
        input_value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        expected = input_value.reshape(4, 6)
        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: pipeline.run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(expected,),
        )

        transformed = result.document
        self.assertEqual(len(transformed.subgraph().operators), 1)
        output_tensor = transformed.subgraph().tensors[
            transformed.subgraph().outputs[0]
        ]
        self.assertEqual(list(output_tensor.shape), [4, 6])


if __name__ == "__main__":
    import unittest

    unittest.main()
