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

from test.support.circle.builder import CircleModelBuilder
from test.support.circle.value_test import CircleValueTestCase


class CircleIOValueTest(CircleValueTestCase):
    """Check that Circle serialization preserves executable values."""

    def test_object_api_round_trip_preserves_output_and_interface(self):
        """Compare an Object API fixture with its serialized round trip."""

        builder = CircleModelBuilder(description="io-value-test")
        x = builder.input("x", [2, 3])
        two = builder.const_f32("two", 2.0)
        output = builder.add(x, two, name="output")
        builder.set_outputs(output)
        source = builder.build()

        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)
        expected = input_value + np.float32(2.0)
        source_result = self.evaluator.evaluate(source, (input_value,))
        restored = self.round_trip(source)
        restored_result = self.evaluator.evaluate(restored, (input_value,))

        self.assert_outputs_equal((expected,), source_result.outputs)
        self.assert_outputs_equal((expected,), restored_result.outputs)
        self.assert_outputs_equal(source_result.outputs, restored_result.outputs)
        self.assert_interfaces_equal(source, restored, check_tensor_names=True)


if __name__ == "__main__":
    import unittest

    unittest.main()
