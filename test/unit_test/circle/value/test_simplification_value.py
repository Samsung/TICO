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

from tico.circle.passes import (
    CanonicalizeEquivalentOpsPass,
    CirclePassContext,
    CirclePassManager,
    RemoveNoOpOperatorsPass,
    SimplifyViewOpsPass,
)
from tico.circle.passes.cleanup import DeadCodeEliminationPass

from test.support.circle.builder import CircleModelBuilder
from test.support.circle.value_test import CircleValueTestCase


class CircleSimplificationValueTest(CircleValueTestCase):
    """Check generated Circle round trips and values after PR 3 rewrites."""

    def test_identity_reshape_is_removed_without_changing_values(self):
        """Rewire an identity RESHAPE and remove it with external DCE."""

        builder = CircleModelBuilder(description="identity-reshape-value-test")
        source_tensor = builder.input("input", [2, 3])
        output = builder.reshape(source_tensor, [2, 3], name="output")
        builder.set_outputs(output)
        source = builder.build()
        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)

        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: CirclePassManager(
                [SimplifyViewOpsPass(), DeadCodeEliminationPass()]
            ).run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(input_value,),
        )

        self.assertEqual(len(result.document.subgraph().operators), 0)

    def test_inverse_transposes_are_removed_without_changing_values(self):
        """Rewire inverse TRANSPOSE operations and remove them with external DCE."""

        builder = CircleModelBuilder(description="inverse-transpose-value-test")
        source_tensor = builder.input("input", [2, 3])
        transposed = builder.transpose(
            source_tensor,
            [1, 0],
            name="transposed",
        )
        output = builder.transpose(transposed, [1, 0], name="output")
        builder.set_outputs(output)
        source = builder.build()
        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)

        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: CirclePassManager(
                [SimplifyViewOpsPass(), DeadCodeEliminationPass()]
            ).run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(input_value,),
        )

        self.assertEqual(len(result.document.subgraph().operators), 0)

    def test_add_zero_is_removed_without_changing_values(self):
        """Remove an activation-free ADD with a scalar zero constant."""

        builder = CircleModelBuilder(description="add-zero-value-test")
        source_tensor = builder.input("input", [2, 3])
        zero = builder.const_f32("zero", np.float32(0.0))
        output = builder.add(source_tensor, zero, name="output")
        builder.set_outputs(output)
        source = builder.build()
        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)

        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: RemoveNoOpOperatorsPass().run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(input_value,),
        )

        self.assertEqual(len(result.document.subgraph().operators), 0)

    def test_reshape_moves_after_scalar_mul_without_changing_values(self):
        """Commute RESHAPE through scalar MUL and preserve numerical outputs."""

        builder = CircleModelBuilder(description="reshape-mul-value-test")
        source_tensor = builder.input("input", [2, 3])
        reshaped = builder.reshape(source_tensor, [3, 2], name="reshaped")
        scalar = builder.const_f32("scalar", np.float32(2.0))
        output = builder.mul(reshaped, scalar, name="output")
        builder.set_outputs(output)
        source = builder.build()
        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)
        expected = input_value.reshape(3, 2) * np.float32(2.0)

        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: SimplifyViewOpsPass().run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(expected,),
        )

        transformed = result.document.subgraph().operators
        self.assertEqual(len(transformed), 2)
        self.assertEqual(transformed[0].inputs[0], source_tensor)

    def test_unit_axis_transpose_canonicalizes_to_reshape(self):
        """Replace a unit-axis-only TRANSPOSE with an equivalent RESHAPE."""

        builder = CircleModelBuilder(description="view-transpose-value-test")
        source_tensor = builder.input("input", [1, 2, 3])
        output = builder.transpose(source_tensor, [1, 2, 0], name="output")
        builder.set_outputs(output)
        source = builder.build()
        input_value = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        expected = input_value.reshape(2, 3, 1)

        result = self.assert_pass_preserves_value(
            source,
            (input_value,),
            lambda document: CanonicalizeEquivalentOpsPass().run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(expected,),
        )

        transformed = result.document.subgraph().operators
        self.assertEqual(len(transformed), 1)
        self.assertEqual(len(transformed[0].inputs), 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
