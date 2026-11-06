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

"""Serialized native selected-row lookups, numerical and allocation contracts."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from circle_schema import circle
from tico.circle.document import CircleDocument
from tico.circle.runtime import (
    CircleReferenceRuntime,
    CircleRuntimeValidationError,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)

from test.support.circle.builder import CircleModelBuilder


def _options(name, **kwargs):
    option = getattr(getattr(circle, name), f"{name}T")()
    for key, value in kwargs.items():
        setattr(option, key, value)
    return option


def rowwise_model(*, shared_dq=False, shared_weight=False, indices_dtype=np.int64):
    """W -> DQ -> Gather -> Q_A -> DQ_A -> Mul(DQ_scale) -> Q_B."""

    builder = CircleModelBuilder()
    ids = builder.input("ids", [1, 1], dtype=indices_dtype, shape_signature=[1, -1])
    codes = np.array(
        [[0, 1, 2, 255], [50, 51, 52, 0], [0, 127, 128, 255], [253, 254, 255, 0]],
        dtype=np.uint8,
    )
    weight = builder.constant("weight", codes)
    builder.subgraph.tensors[weight].quantization = builder.quantization(
        [0.5, 0.25, 100.0, 0.125], [1, 51, 128, 254], quantized_dimension=0
    )
    full = builder.activation("full", codes.shape)
    builder.operator(
        "DEQUANTIZE", [weight], [full], options=_options("DequantizeOptions")
    )
    gathered = builder.activation("selected", [1, 1, 4], shape_signature=[1, -1, 4])
    builder.operator(
        "GATHER",
        [full, ids],
        [gathered],
        options=_options("GatherOptions", axis=0, batchDims=0),
    )
    qa = builder.activation(
        "qa",
        [1, 1, 4],
        dtype=np.int8,
        shape_signature=[1, -1, 4],
        quantization=builder.quantization([1.0], [0]),
    )
    builder.operator("QUANTIZE", [gathered], [qa], options=_options("QuantizeOptions"))
    dqa = builder.activation("dqa", [1, 1, 4], shape_signature=[1, -1, 4])
    builder.operator("DEQUANTIZE", [qa], [dqa], options=_options("DequantizeOptions"))
    # Fake-quantized MUL operand: code 3, zp 1, scale .75 => 1.5, not raw 1.6.
    scale_q = builder.constant("scale_q", np.array(3, dtype=np.uint8))
    builder.subgraph.tensors[scale_q].quantization = builder.quantization([0.75], [1])
    scale = builder.activation("scale", [])
    builder.operator(
        "DEQUANTIZE", [scale_q], [scale], options=_options("DequantizeOptions")
    )
    mul = builder.activation("mul", [1, 1, 4], shape_signature=[1, -1, 4])
    builder.operator(
        "MUL",
        [dqa, scale],
        [mul],
        options=_options("MulOptions", fusedActivationFunction=0),
    )
    out = builder.activation(
        "out",
        [1, 1, 4],
        dtype=np.int16,
        shape_signature=[1, -1, 4],
        quantization=builder.quantization([0.001], [0]),
    )
    builder.operator("QUANTIZE", [mul], [out], options=_options("QuantizeOptions"))
    outputs = [out]
    if shared_dq:
        outputs.append(full)
    if shared_weight:
        # Another consumer of raw shared storage must remain untouched.
        raw = builder.activation(
            "raw",
            codes.shape,
            dtype=np.uint8,
            quantization=builder.subgraph.tensors[weight].quantization,
        )
        builder.operator(
            "RESHAPE",
            [weight],
            [raw],
            options=_options("ReshapeOptions", newShape=list(codes.shape)),
        )
        outputs.append(raw)
    builder.set_outputs(*outputs)
    return builder, weight, full, gathered


class RowwiseGatherTest(unittest.TestCase):
    def _runtime(self, **kwargs):
        builder, weight, full, gathered = rowwise_model(**kwargs)
        data = builder.build().to_bytes()
        doc = CircleDocument.from_bytes(data)
        self.assertTrue(doc.verify().ok)
        return CircleReferenceRuntime(doc), data, weight, full, gathered

    def test_decode_prefill_roundtrip_exact_native_and_fake(self):
        runtime, data, weight, full, gathered = self._runtime()
        oracle = CircleReferenceRuntime(data, enable_rowwise_gather=False)
        self.assertEqual(len(runtime.rowwise_gather_plans), 1)
        self.assertEqual(runtime.program.tensor(weight).constant.dtype, np.uint8)
        for tokens in ([[0]], [[3]], [[2, 0, 3, 1, 2, 0]]):
            ids = np.asarray(tokens, dtype=np.int64)
            for mode in (ExecutionMode.NATIVE, ExecutionMode.FAKE_QUANTIZE):
                actual = runtime.run([ids], mode=mode, trace=True)
                expected = oracle.run([ids], mode=mode, trace=True)
                np.testing.assert_array_equal(actual.outputs[0], expected.outputs[0])
                assert actual.tensor_values is not None
                assert expected.tensor_values is not None
                np.testing.assert_array_equal(
                    actual.tensor_values[gathered], expected.tensor_values[gathered]
                )
                self.assertNotIn(full, actual.tensor_values)
                self.assertEqual(
                    actual.outputs[0].dtype,
                    np.int16 if mode is ExecutionMode.NATIVE else np.float32,
                )
        self.assertNotIn(weight, runtime._dequantized_constants)
        # Independently calculated endpoints/ties after Q_A and scale operand DQ.
        expected_values = np.array(
            [[[-1500, 0, 1500, 32767], [-32768, -32768, 0, 32767]]], dtype=np.int16
        )
        np.testing.assert_array_equal(
            runtime.run([np.array([[0, 2]], dtype=np.int64)]).outputs[0],
            expected_values,
        )

    def test_bounds_errors_are_preserved(self):
        runtime, _, _, _, _ = self._runtime()
        for ids in ([-1], [4], [1, 0, 4]):
            with self.assertRaisesRegex(
                CircleRuntimeValidationError, r"indices contain values outside \[0, 4\)"
            ):
                runtime.run([np.array([ids], dtype=np.int64)])

    def test_loading_and_native_execution_never_dequantize_full_table(self):
        builder, _, _, _ = rowwise_model()
        data = builder.build().to_bytes()
        from tico.circle.runtime import executor, rowwise_gather
        from tico.circle.runtime.kernels import quantize

        original = executor.dequantize_array
        restored = []

        def observe(value, tensor):
            restored.append(value.shape)
            self.assertNotEqual(value.shape, (4, 4))
            return original(value, tensor)

        original_take = rowwise_gather.np.take
        selected = []

        def observe_take(value, indices, *args, **kwargs):
            result = original_take(value, indices, *args, **kwargs)
            if value.shape == (4, 4):
                self.assertEqual(value.dtype, np.uint8)
                selected.append(result.shape)
            return result

        with (
            patch.object(executor, "dequantize_array", side_effect=observe),
            patch.object(quantize, "dequantize_array", side_effect=observe),
            patch.object(rowwise_gather.np, "take", side_effect=observe_take),
        ):
            runtime = CircleReferenceRuntime(data)
            self.assertEqual(restored, [])
            runtime.run([np.array([[3]], dtype=np.int64)])
            runtime.run([np.array([[3, 1, 0]], dtype=np.int64)])
        self.assertEqual(selected, [(1, 1, 4), (1, 3, 4)])
        self.assertEqual(runtime._dequantized_constants, {})
        self.assertEqual(restored, [(1, 1, 4), (), (1, 3, 4), ()])

    def test_shared_integer_weight_unchanged(self):
        runtime, data, weight, _, _ = self._runtime(shared_weight=True)
        before = runtime.program.tensor(weight).constant.copy()
        qparams = runtime.program.tensor(weight).quantization
        result = runtime.run([np.array([[3, 0]], dtype=np.int64)])
        np.testing.assert_array_equal(result.outputs[1], before)
        np.testing.assert_array_equal(runtime.program.tensor(weight).constant, before)
        self.assertEqual(runtime.program.tensor(weight).quantization, qparams)
        np.testing.assert_array_equal(
            result.outputs[0],
            CircleReferenceRuntime(data, enable_rowwise_gather=False)
            .run([np.array([[3, 0]], dtype=np.int64)])
            .outputs[0],
        )
        self.assertFalse(
            any(
                value.dtype == np.float32 and value.shape == before.shape
                for value in runtime._dequantized_constants.values()
            )
        )

    def test_shared_full_dq_rejected_without_hidden_fallback(self):
        builder, _, _, _ = rowwise_model(shared_dq=True)
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "DQ result is a graph output"
        ):
            CircleReferenceRuntime(builder.build().to_bytes())

    def test_invalid_row_qparams_rejected(self):
        for scales, zero_points, axis in (
            ([0.5] * 4, [0] * 4, 1),
            ([0.5, 0, 1, 2], [0] * 4, 0),
            ([0.5] * 4, [0, 0, 0, 256], 0),
        ):
            builder, weight, _, _ = rowwise_model()
            builder.subgraph.tensors[weight].quantization = builder.quantization(
                scales, zero_points, quantized_dimension=axis
            )
            with self.assertRaises(
                (UnsupportedCircleOperatorError, CircleRuntimeValidationError)
            ):
                CircleReferenceRuntime(builder.build().to_bytes(), verify=False)

    def test_per_tensor_non_target_retains_ordinary_execution(self):
        builder, weight, full, _ = rowwise_model()
        builder.subgraph.tensors[weight].quantization = builder.quantization([0.5], [1])
        runtime = CircleReferenceRuntime(builder.build().to_bytes())
        self.assertEqual(runtime.rowwise_gather_plans, {})
        result = runtime.run([np.array([[0]], dtype=np.int64)], trace=True)
        assert result.tensor_values is not None
        self.assertIn(full, result.tensor_values)

    def test_shared_dq_multiple_gathers_restore_each_selection(self):
        builder, _, full, _ = rowwise_model()
        ids = builder.subgraph.inputs[0]
        second = builder.activation("second", [1, 1, 4], shape_signature=[1, -1, 4])
        builder.operator(
            "GATHER",
            [full, ids],
            [second],
            options=_options("GatherOptions", axis=0, batchDims=0),
        )
        builder.subgraph.outputs.append(second)
        data = builder.build().to_bytes()
        runtime = CircleReferenceRuntime(data)
        self.assertEqual(len(runtime.rowwise_gather_plans), 2)
        ids_value = np.array([[3, 0, 1]], dtype=np.int64)
        actual = runtime.run([ids_value])
        expected = CircleReferenceRuntime(data, enable_rowwise_gather=False).run(
            [ids_value]
        )
        for left, right in zip(actual.outputs, expected.outputs):
            np.testing.assert_array_equal(left, right)

    def test_shared_dq_non_gather_consumer_is_rejected(self):
        builder, _, full, _ = rowwise_model()
        builder.reshape(full, [4, 4], name="other_consumer")
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "consumer other than"
        ):
            CircleReferenceRuntime(builder.build().to_bytes())

    def test_quantized_indices_and_nonzero_batch_dims_rejected(self):
        for case in ("indices", "batch_dims"):
            builder, _, _, _ = rowwise_model()
            if case == "indices":
                builder.subgraph.tensors[builder.subgraph.inputs[0]].quantization = (
                    builder.quantization([1.0], [0])
                )
            else:
                builder.subgraph.operators[1].builtinOptions.batchDims = 1
            with self.assertRaises(UnsupportedCircleOperatorError):
                CircleReferenceRuntime(builder.build().to_bytes())

    def test_int32_indices_and_empty_prefill(self):
        runtime, data, _, _, _ = self._runtime(indices_dtype=np.int32)
        for ids in (
            np.array([[3, 1, 3, 0]], dtype=np.int32),
            np.empty((1, 0), dtype=np.int32),
        ):
            expected = (
                CircleReferenceRuntime(data, enable_rowwise_gather=False)
                .run([ids])
                .outputs[0]
            )
            np.testing.assert_array_equal(runtime.run([ids]).outputs[0], expected)


if __name__ == "__main__":
    unittest.main()
