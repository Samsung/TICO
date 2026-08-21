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

from tico.circle.passes import CirclePassContext, FoldConstantsPass

from test.support.circle.builder import CircleModelBuilder
from test.support.circle.value_test import CircleValueTestCase


class CircleConstantFoldValueTest(CircleValueTestCase):
    """Check generated Circle round trips and values after constant folding."""

    def test_add_mul_chain_is_replaced_by_one_constant_output(self):
        """Fold a generated arithmetic chain without changing output semantics."""

        builder = CircleModelBuilder(description="constant-fold-value-test")
        lhs = builder.const_f32("lhs", np.array([1.0, 2.0], np.float32))
        rhs = builder.const_f32("rhs", np.array([3.0, 4.0], np.float32))
        added = builder.add(lhs, rhs, name="added")
        scale = builder.const_f32("scale", np.float32(2.0))
        output = builder.mul(added, scale, name="output")
        builder.set_outputs(output)
        source = builder.build()
        expected = np.array([8.0, 12.0], np.float32)

        result = self.assert_pass_preserves_value(
            source,
            (),
            lambda document: FoldConstantsPass().run(
                document,
                CirclePassContext(verify_after_each_pass=True),
            ),
            expected_outputs=(expected,),
        )

        self.assertEqual(len(result.document.subgraph().operators), 0)
        output_tensor = result.document.subgraph().tensors[
            result.document.subgraph().outputs[0]
        ]
        self.assertGreater(int(output_tensor.buffer), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
