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
from tico.circle.passes import CirclePassContext, CirclePassManager
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass

from test.support.circle.builder import CircleModelBuilder
from test.support.circle.value_test import CircleValueTestCase


class CircleCleanupValueTest(CircleValueTestCase):
    """Check value preservation across dead-code elimination and compaction."""

    def test_dce_and_compaction_preserve_values_with_interleaved_dead_objects(self):
        """Remove an interleaved dead branch without changing live outputs."""

        builder = CircleModelBuilder(description="cleanup-value-test")
        x = builder.input("x", [3])
        live_addend = builder.const_f32("live_addend", 2.0)
        added = builder.add(x, live_addend, name="added")

        dead_addend = builder.const_f32("dead_addend", 100.0)
        builder.sub(x, dead_addend, name="dead_output")

        live_multiplier = builder.const_f32("live_multiplier", 3.0)
        output = builder.mul(added, live_multiplier, name="output")
        builder.set_outputs(output)
        source = builder.build()
        self.assertEqual(len(source.model.operatorCodes), 3)

        pipeline = CirclePassManager(
            [
                DeadCodeEliminationPass(),
                CompactIndicesPass(),
            ]
        )
        input_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = (input_value + np.float32(2.0)) * np.float32(3.0)

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
        self.assertEqual(len(transformed.subgraph().operators), 2)
        self.assertEqual(len(transformed.subgraph().tensors), 5)
        self.assertEqual(len(transformed.model.buffers), 3)
        self.assertEqual(len(transformed.model.operatorCodes), 2)
        self.assertNotIn(
            "dead_output",
            [decode_text(tensor.name) for tensor in transformed.subgraph().tensors],
        )
        self.assertNotIn(
            "dead_addend",
            [decode_text(tensor.name) for tensor in transformed.subgraph().tensors],
        )
        self.assertTrue(result.transform_result.modified)
        self.assertGreater(result.transform_result.changes, 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
