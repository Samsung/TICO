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

"""Graph-level contracts of the Circle reference runtime.

Expected values are written by hand from small fixtures; the runtime under test
never sees them.
"""

from __future__ import annotations

import unittest

import numpy as np
from circle_schema import circle
from tico.circle import CircleDocument
from tico.circle.runtime import (
    CircleReferenceRuntime,
    CircleRuntimeValidationError,
    ExecutionMode,
    run_circle,
    UnsupportedCircleOperatorError,
)

from test.support.circle.builder import CircleModelBuilder


def _add_options() -> circle.AddOptions.AddOptionsT:
    options = circle.AddOptions.AddOptionsT()
    options.fusedActivationFunction = 0
    return options


class InterfaceBindingTest(unittest.TestCase):
    def test_multiple_outputs_keep_declared_order_and_graph_input_passthrough(self):
        """A graph input reused as an output is returned as a copy in interface order."""

        builder = CircleModelBuilder()
        x = builder.input("x", [2])
        one = builder.const_f32("one", [1.0, 1.0])
        added = builder.add(x, one, name="added")
        builder.set_outputs(added, x, one)
        document = builder.build()

        value = np.array([3.0, 4.0], dtype=np.float32)
        result = run_circle(document.to_bytes(), (value,))

        self.assertEqual(len(result.outputs), 3)
        np.testing.assert_array_equal(result.outputs[0], [4.0, 5.0])
        np.testing.assert_array_equal(result.outputs[1], [3.0, 4.0])
        np.testing.assert_array_equal(result.outputs[2], [1.0, 1.0])
        result.outputs[1][0] = 100.0
        self.assertEqual(value[0], 3.0)

    def test_constant_only_graph(self):
        """A graph without operators or inputs returns its constant outputs."""

        builder = CircleModelBuilder()
        const = builder.const_i32("answer", [[42]])
        builder.set_outputs(const)
        document = builder.build()

        result = run_circle(document.to_bytes(), ())
        np.testing.assert_array_equal(
            result.outputs[0], np.array([[42]], dtype=np.int32)
        )
        self.assertEqual(result.outputs[0].dtype, np.int32)

    def test_scalar_and_zero_sized_constants_keep_their_rank(self):
        """Scalar constants stay 0-d and zero-sized constants stay empty."""

        builder = CircleModelBuilder()
        scalar = builder.const_f32("scalar", 2.5)
        empty = builder.constant("empty", np.zeros((0, 3), dtype=np.float32))
        builder.set_outputs(scalar, empty)
        document = CircleDocument.from_bytes(builder.build().to_bytes())

        result = run_circle(document, ())
        self.assertEqual(result.outputs[0].shape, ())
        self.assertEqual(float(result.outputs[0]), 2.5)
        self.assertEqual(result.outputs[1].shape, (0, 3))

    def test_input_count_shape_and_dtype_are_validated(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3])
        builder.set_outputs(x)
        runtime = CircleReferenceRuntime(builder.build())

        with self.assertRaisesRegex(CircleRuntimeValidationError, "expects 1 inputs"):
            runtime.run(())
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "requires dimension 1 to be 3"
        ):
            runtime.run((np.zeros((2, 4), dtype=np.float32),))
        with self.assertRaisesRegex(CircleRuntimeValidationError, "requires rank 2"):
            runtime.run((np.zeros((6,), dtype=np.float32),))
        with self.assertRaisesRegex(CircleRuntimeValidationError, "dtype float64"):
            runtime.run((np.zeros((2, 3), dtype=np.float64),))

    def test_inputs_are_not_mutated_and_runs_are_independent(self):
        """Repeated runs on one runtime do not leak state or modify inputs."""

        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        bias = builder.const_f32("bias", [1.0, 2.0, 3.0])
        output = builder.add(x, bias, name="output")
        builder.set_outputs(output)
        runtime = CircleReferenceRuntime(builder.build())

        first = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        second = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        out_first = runtime.run((first,)).outputs[0]
        out_second = runtime.run((second,)).outputs[0]
        out_first_again = runtime.run((first,)).outputs[0]

        np.testing.assert_array_equal(first, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(out_first, [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(out_second, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(out_first_again, out_first)

    def test_torch_tensor_inputs_are_accepted(self):
        import torch

        builder = CircleModelBuilder()
        x = builder.input("x", [2])
        y = builder.input("y", [2])
        output = builder.mul(x, y, name="output")
        builder.set_outputs(output)
        runtime = CircleReferenceRuntime(builder.build())

        result = runtime.run((torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0])))
        np.testing.assert_array_equal(result.outputs[0], [8.0, 15.0])

    def test_trace_keeps_intermediate_values(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [2])
        one = builder.const_f32("one", 1.0)
        added = builder.add(x, one, name="added")
        doubled = builder.add(added, added, name="doubled")
        builder.set_outputs(doubled)
        runtime = CircleReferenceRuntime(builder.build())

        value = np.array([1.0, 2.0], dtype=np.float32)
        plain = runtime.run((value,))
        traced = runtime.run((value,), trace=True)

        self.assertIsNone(plain.tensor_values)
        assert traced.tensor_values is not None
        np.testing.assert_array_equal(traced.tensor_values[added], [2.0, 3.0])
        np.testing.assert_array_equal(traced.tensor_values[doubled], [4.0, 6.0])
        self.assertIn(one, traced.tensor_values)


class DynamicShapeTest(unittest.TestCase):
    def _dynamic_model(self) -> CircleDocument:
        """x[-1, 3] -> RESHAPE(SHAPE-derived) -> ADD bias, output [-1, 3]."""

        builder = CircleModelBuilder()
        x = builder.input("x", [1, 3], shape_signature=[-1, 3])
        bias = builder.const_f32("bias", [10.0, 20.0, 30.0])
        summed = builder.activation("summed", [1, 3], shape_signature=[-1, 3])
        builder.operator("ADD", [x, bias], [summed], options=_add_options())
        # SHAPE -> the output shape follows the runtime batch size.
        shape = builder.activation("shape", [2], dtype=np.int32)
        shape_options = circle.ShapeOptions.ShapeOptionsT()
        shape_options.outType = circle.TensorType.TensorType.INT32
        builder.operator("SHAPE", [summed], [shape], options=shape_options)
        reshaped = builder.activation("reshaped", [1, 3], shape_signature=[-1, 3])
        reshape_options = circle.ReshapeOptions.ReshapeOptionsT()
        reshape_options.newShape = [1, 3]
        builder.operator(
            "RESHAPE", [summed, shape], [reshaped], options=reshape_options
        )
        builder.set_outputs(reshaped, shape)
        return CircleDocument.from_bytes(builder.build().to_bytes())

    def test_same_model_runs_with_several_batch_sizes(self):
        runtime = CircleReferenceRuntime(self._dynamic_model())
        for batch in (1, 4, 2):
            value = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)
            result = runtime.run((value,))
            np.testing.assert_array_equal(
                result.outputs[0],
                value + np.array([10.0, 20.0, 30.0], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                result.outputs[1], np.array([batch, 3], dtype=np.int32)
            )
            self.assertEqual(result.outputs[1].dtype, np.int32)

    def test_static_dimension_of_dynamic_input_is_still_enforced(self):
        runtime = CircleReferenceRuntime(self._dynamic_model())
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "dimension 1 to be 3"
        ):
            runtime.run((np.zeros((2, 4), dtype=np.float32),))

    def test_probe_rejects_dynamic_inputs(self):
        runtime = CircleReferenceRuntime(self._dynamic_model())
        with self.assertRaisesRegex(CircleRuntimeValidationError, "dynamic inputs"):
            runtime.probe_static_contracts()

    def test_declared_static_output_shape_is_checked_against_computed_shape(self):
        """A wrong serialized output shape is reported, not silently reshaped."""

        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3])
        wrong = builder.activation("wrong", [3, 2])
        perm = builder.const_i32("perm", [0, 1])
        builder.operator(
            "TRANSPOSE",
            [x, perm],
            [wrong],
            options=circle.TransposeOptions.TransposeOptionsT(),
        )
        builder.set_outputs(wrong)
        runtime = CircleReferenceRuntime(builder.build())

        with self.assertRaisesRegex(
            CircleRuntimeValidationError,
            r"operator 0 \(TRANSPOSE\) output 0 has shape \[2, 3\].*dimension 0 to be 3",
        ):
            runtime.run((np.zeros((2, 3), dtype=np.float32),))
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "dimension 0 to be 3"
        ):
            runtime.probe_static_contracts()

    def test_declared_dtype_is_checked_against_computed_dtype(self):
        """An INT32 output produced by a FLOAT32 ADD is rejected without casting."""

        builder = CircleModelBuilder()
        x = builder.input("x", [2])
        one = builder.const_f32("one", 1.0)
        wrong = builder.activation("wrong", [2], dtype=np.int32)
        builder.operator("ADD", [x, one], [wrong], options=_add_options())
        builder.set_outputs(wrong)
        runtime = CircleReferenceRuntime(builder.build())

        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "dtype float32.*requires INT32"
        ):
            runtime.run((np.zeros((2,), dtype=np.float32),))


class PrepareValidationTest(unittest.TestCase):
    def test_unsupported_operator_reports_opcode_version_and_operands(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        out = builder.activation("out", [3])
        builder.operator("UNIQUE", [x], [out], version=3)
        builder.set_outputs(out)

        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError,
            r"UNIQUE[\s\S]*operators\[0\][\s\S]*version 3[\s\S]*FLOAT32\[3\]",
        ):
            CircleReferenceRuntime(builder.build())

    def test_custom_operator_is_rejected(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        out = builder.activation("out", [3])
        builder.operator("CUSTOM", [x], [out])
        builder.model.operatorCodes[0].customCode = "MyCustomOp"
        builder.set_outputs(out)

        with self.assertRaisesRegex(UnsupportedCircleOperatorError, "Custom operators"):
            CircleReferenceRuntime(builder.build())

    def test_unproduced_tensor_is_reported_by_structural_verification(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        phantom = builder.activation("phantom", [3])
        out = builder.activation("out", [3])
        builder.operator("ADD", [x, phantom], [out], options=_add_options())
        builder.set_outputs(out)
        document = builder.build()

        from tico.circle.verify import CircleVerificationError

        with self.assertRaisesRegex(CircleVerificationError, "UNDEFINED_INPUT"):
            CircleReferenceRuntime(document)
        # Without structural verification the executor reports the same defect.
        with self.assertRaisesRegex(CircleRuntimeValidationError, "has no value"):
            CircleReferenceRuntime(document, verify=False).run(
                (np.zeros((3,), dtype=np.float32),)
            )

    def test_constant_with_wrong_payload_size_is_rejected(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        bias = builder.const_f32("bias", [1.0, 2.0, 3.0])
        out = builder.add(x, bias, name="out")
        builder.set_outputs(out)
        document = builder.build()
        buffer = document.model.buffers[document.subgraph().tensors[bias].buffer]
        buffer.data = buffer.data[:-4]

        with self.assertRaisesRegex(
            CircleRuntimeValidationError,
            r"tensors\[1\] constant payload.*requires 12 bytes",
        ):
            CircleReferenceRuntime(document, verify=False)

    def test_operator_arity_is_validated(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [3])
        out = builder.activation("out", [3])
        builder.operator("ADD", [x], [out], options=_add_options())
        builder.set_outputs(out)
        runtime = CircleReferenceRuntime(builder.build())

        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "ADD expects 2 inputs, found 1"
        ):
            runtime.run((np.zeros((3,), dtype=np.float32),))

    def test_invalid_option_is_rejected(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3])
        out = builder.activation("out", [3, 2])
        perm = builder.const_i32("perm", [1, 1])
        builder.operator(
            "TRANSPOSE",
            [x, perm],
            [out],
            options=circle.TransposeOptions.TransposeOptionsT(),
        )
        builder.set_outputs(out)
        runtime = CircleReferenceRuntime(builder.build())

        with self.assertRaisesRegex(
            CircleRuntimeValidationError, r"permutation \[1, 1\] is invalid"
        ):
            runtime.run((np.zeros((2, 3), dtype=np.float32),))

    def test_unknown_fused_activation_is_rejected(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [2])
        one = builder.const_f32("one", 1.0)
        out = builder.add(x, one, name="out")
        builder.set_outputs(out)
        document = builder.build()
        document.subgraph().operators[0].builtinOptions.fusedActivationFunction = 5

        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "fused activation function 5"
        ):
            run_circle(document, (np.zeros((2,), dtype=np.float32),))

    def test_static_probe_accepts_consistent_model(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [2, 3])
        reshaped = builder.reshape(x, [3, 2], name="reshaped")
        builder.set_outputs(reshaped)
        runtime = CircleReferenceRuntime(builder.build())
        result = runtime.probe_static_contracts()
        self.assertEqual(result.outputs[0].shape, (3, 2))


class NativeQuantizedPolicyTest(unittest.TestCase):
    def test_value_preserving_operator_moves_quantized_integers_exactly(self):
        builder = CircleModelBuilder()
        quant = builder.quantization([0.5], [3])
        x = builder.activation("x", [2, 2], dtype=np.uint8, quantization=quant)
        builder.subgraph.inputs.append(x)
        out = builder.activation("out", [2, 2], dtype=np.uint8, quantization=quant)
        perm = builder.const_i32("perm", [1, 0])
        builder.operator(
            "TRANSPOSE",
            [x, perm],
            [out],
            options=circle.TransposeOptions.TransposeOptionsT(),
        )
        builder.set_outputs(out)
        runtime = CircleReferenceRuntime(builder.build())

        value = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = runtime.run((value,))
        np.testing.assert_array_equal(result.outputs[0], [[1, 3], [2, 4]])
        self.assertEqual(result.outputs[0].dtype, np.uint8)

    def test_arithmetic_on_quantized_integers_is_rejected_in_native_mode(self):
        builder = CircleModelBuilder()
        quant = builder.quantization([0.5], [3])
        x = builder.activation("x", [2], dtype=np.uint8, quantization=quant)
        builder.subgraph.inputs.append(x)
        out = builder.activation("out", [2], dtype=np.uint8, quantization=quant)
        builder.operator("ADD", [x, x], [out], options=_add_options())
        builder.set_outputs(out)
        runtime = CircleReferenceRuntime(builder.build())

        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "ADD on integer quantized"
        ):
            runtime.run((np.array([1, 2], dtype=np.uint8),))
        # The same graph executes with dequantized semantics: (1-3)*0.5 + (1-3)*0.5 = -2
        # requantized on the uint8 grid: -2/0.5 + 3 = -1 -> clamped to 0 -> -1.5.
        result = runtime.run(
            (np.array([-1.0, 4.0], dtype=np.float32),), mode=ExecutionMode.FAKE_QUANTIZE
        )
        np.testing.assert_array_equal(result.outputs[0], [-1.5, 8.0])


if __name__ == "__main__":
    unittest.main()
