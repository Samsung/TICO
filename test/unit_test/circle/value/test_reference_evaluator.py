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

from test.support.circle.builder import CircleModelBuilder
from test.support.circle.evaluator import CircleReferenceEvaluator


class CircleReferenceEvaluatorTest(unittest.TestCase):
    """Test the NumPy reference semantics independently of Circle passes."""

    def setUp(self):
        """Create a fresh evaluator for each test."""

        self.evaluator = CircleReferenceEvaluator()

    def test_add_supports_broadcasting_and_records_intermediates(self):
        """Evaluate broadcast ADD and expose its intermediate tensor value."""

        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3])
        bias = builder.const_f32("bias", [1.0, 2.0, 3.0])
        output = builder.add(x, bias, name="output")
        builder.set_outputs(output)
        document = builder.build()

        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)
        expected = input_value + np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.evaluator.evaluate(document, (input_value,))

        np.testing.assert_array_equal(result.outputs[0], expected)
        np.testing.assert_array_equal(result.tensor_values[output], expected)

    def test_sub_matches_numpy(self):
        """Evaluate SUB with a broadcast constant."""

        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3])
        bias = builder.const_f32("bias", [1.0, 2.0, 3.0])
        output = builder.sub(x, bias, name="output")
        builder.set_outputs(output)
        document = builder.build()

        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)
        expected = input_value - np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.evaluator.evaluate(document, (input_value,))

        np.testing.assert_array_equal(result.outputs[0], expected)

    def test_reshape_and_transpose_match_numpy(self):
        """Evaluate a RESHAPE followed by a TRANSPOSE."""

        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3, 4])
        reshaped = builder.reshape(x, [6, 4], name="reshaped")
        output = builder.transpose(reshaped, [1, 0], name="output")
        builder.set_outputs(output)
        document = builder.build()

        input_value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        expected = input_value.reshape(6, 4).transpose(1, 0)
        result = self.evaluator.evaluate(document, (input_value,))

        np.testing.assert_array_equal(result.outputs[0], expected)

    def test_multiple_outputs_preserve_declared_order(self):
        """Return multiple graph outputs in the declared interface order."""

        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        one = builder.const_f32("one", 1.0)
        two = builder.const_f32("two", 2.0)
        added = builder.add(x, one, name="added")
        multiplied = builder.mul(added, two, name="multiplied")
        builder.set_outputs(multiplied, added)
        document = builder.build()

        input_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.evaluator.evaluate(document, (input_value,))

        np.testing.assert_array_equal(result.outputs[0], (input_value + 1.0) * 2.0)
        np.testing.assert_array_equal(result.outputs[1], input_value + 1.0)

    def test_input_shape_mismatch_is_rejected(self):
        """Reject inputs that do not match the Circle tensor shape."""

        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3])
        builder.set_outputs(x)
        document = builder.build()

        with self.assertRaisesRegex(ValueError, "shape"):
            self.evaluator.evaluate(
                document,
                (np.zeros((3, 2), dtype=np.float32),),
            )

    def test_input_dtype_mismatch_is_rejected(self):
        """Reject inputs that do not match the Circle tensor dtype."""

        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3])
        builder.set_outputs(x)
        document = builder.build()

        with self.assertRaisesRegex(TypeError, "dtype"):
            self.evaluator.evaluate(
                document,
                (np.zeros((2, 3), dtype=np.float64),),
            )

    def test_unsupported_operator_is_rejected(self):
        """Reject builtin operators outside the explicit evaluator subset."""

        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        one = builder.const_f32("one", 1.0)
        output = builder.add(x, one, name="output")
        builder.set_outputs(output)
        document = builder.build()
        operator_code = document.model.operatorCodes[0]
        operator_code.builtinCode = circle.BuiltinOperator.BuiltinOperator.ABS
        operator_code.deprecatedBuiltinCode = circle.BuiltinOperator.BuiltinOperator.ABS

        with self.assertRaisesRegex(NotImplementedError, "builtin operator"):
            self.evaluator.evaluate(
                document,
                (np.ones(3, dtype=np.float32),),
            )

    def test_fused_activation_is_rejected(self):
        """Reject arithmetic operators whose fused activation is not NONE."""

        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        one = builder.const_f32("one", 1.0)
        output = builder.add(x, one, name="output")
        builder.set_outputs(output)
        document = builder.build()
        document.subgraph().operators[
            0
        ].builtinOptions.fusedActivationFunction = (
            circle.ActivationFunctionType.ActivationFunctionType.RELU
        )

        with self.assertRaisesRegex(NotImplementedError, "activation"):
            self.evaluator.evaluate(
                document,
                (np.ones(3, dtype=np.float32),),
            )

    def test_optional_input_is_rejected(self):
        """Reject an optional operator input instead of guessing semantics."""

        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        one = builder.const_f32("one", 1.0)
        output = builder.add(x, one, name="output")
        builder.set_outputs(output)
        document = builder.build()
        document.subgraph().operators[0].inputs[1] = -1

        with self.assertRaisesRegex(NotImplementedError, "optional inputs"):
            self.evaluator.evaluate(
                document,
                (np.ones(3, dtype=np.float32),),
            )

    def test_external_buffer_is_rejected(self):
        """Reject constants backed by an external buffer."""

        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        one = builder.const_f32("one", 1.0)
        output = builder.add(x, one, name="output")
        builder.set_outputs(output)
        document = builder.build()
        constant_tensor = document.subgraph().tensors[one]
        external_buffer = document.model.buffers[int(constant_tensor.buffer)]
        external_buffer.offset = 16
        external_buffer.size = 4

        with self.assertRaisesRegex(NotImplementedError, "external buffers"):
            self.evaluator.evaluate(
                document,
                (np.ones(3, dtype=np.float32),),
            )


if __name__ == "__main__":
    unittest.main()
