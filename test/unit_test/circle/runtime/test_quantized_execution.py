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

"""QUANTIZE/DEQUANTIZE kernels, packed constants, and fake-quantize semantics."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
from circle_schema import circle
from tico.circle.runtime import (
    CircleReferenceRuntime,
    CircleRuntimeValidationError,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)

from test.support.circle.builder import CircleModelBuilder

UINT8 = circle.TensorType.TensorType.UINT8
INT8 = circle.TensorType.TensorType.INT8
INT16 = circle.TensorType.TensorType.INT16
INT32 = circle.TensorType.TensorType.INT32
UINT4 = circle.TensorType.TensorType.UINT4
INT4 = circle.TensorType.TensorType.INT4


def _options(name: str, **fields: Any) -> Any:
    options = getattr(getattr(circle, name), f"{name}T")()
    for key, value in fields.items():
        setattr(options, key, value)
    return options


class QuantizeDequantizeKernelTest(unittest.TestCase):
    def _quantize_model(
        self, tensor_type: int, scale: float, zero_point: int, count: int
    ):
        builder = CircleModelBuilder()
        x = builder.input("x", [count])
        out = builder.activation("q", [count], dtype=np.uint8)
        tensor = builder.subgraph.tensors[out]
        tensor.type = tensor_type
        tensor.quantization = builder.quantization([scale], [zero_point])
        builder.operator("QUANTIZE", [x], [out], options=_options("QuantizeOptions"))
        builder.set_outputs(out)
        return CircleReferenceRuntime(builder.build())

    def test_quantize_rounds_half_away_from_zero_and_clamps(self):
        runtime = self._quantize_model(UINT8, scale=0.5, zero_point=128, count=6)
        value = np.array([0.25, -0.25, 0.75, 1.25, 100.0, -100.0], dtype=np.float32)
        result = runtime.run((value,)).outputs[0]
        # 0.25/0.5 = 0.5 -> 1, -0.5 -> -1, 1.5 -> 2, 2.5 -> 3, clamps at 255 / 0
        np.testing.assert_array_equal(
            result, np.array([129, 127, 130, 131, 255, 0], dtype=np.uint8)
        )
        self.assertEqual(result.dtype, np.uint8)

    def test_quantize_to_int16_symmetric(self):
        runtime = self._quantize_model(INT16, scale=0.001, zero_point=0, count=3)
        value = np.array([1.0, -1.0, 40.0], dtype=np.float32)
        result = runtime.run((value,)).outputs[0]
        np.testing.assert_array_equal(
            result, np.array([1000, -1000, 32767], dtype=np.int16)
        )

    def test_dequantize_per_tensor_and_per_channel(self):
        builder = CircleModelBuilder()
        per_tensor = builder.quantized_constant(
            "w8",
            [[0, 128], [255, 130]],
            tensor_type=UINT8,
            scale=[0.5],
            zero_point=[128],
        )
        per_channel = builder.quantized_constant(
            "w8c",
            [[0, 128], [255, 130]],
            tensor_type=UINT8,
            scale=[0.5, 0.1],
            zero_point=[128, 130],
            quantized_dimension=1,
        )
        out_a = builder.activation("a", [2, 2])
        out_b = builder.activation("b", [2, 2])
        builder.operator(
            "DEQUANTIZE", [per_tensor], [out_a], options=_options("DequantizeOptions")
        )
        builder.operator(
            "DEQUANTIZE", [per_channel], [out_b], options=_options("DequantizeOptions")
        )
        builder.set_outputs(out_a, out_b)
        result = CircleReferenceRuntime(builder.build()).run(())
        np.testing.assert_allclose(
            result.outputs[0], [[-64.0, 0.0], [63.5, 1.0]], rtol=1e-6
        )
        np.testing.assert_allclose(
            result.outputs[1], [[-64.0, -0.2], [63.5, 0.0]], rtol=1e-5, atol=1e-6
        )

    def test_per_channel_axis_mismatch_is_rejected(self):
        builder = CircleModelBuilder()
        weights = builder.quantized_constant(
            "w",
            [[0, 128, 1]],
            tensor_type=UINT8,
            scale=[0.5, 0.1],
            zero_point=[128, 130],
            quantized_dimension=1,
        )
        out = builder.activation("out", [1, 3])
        builder.operator(
            "DEQUANTIZE", [weights], [out], options=_options("DequantizeOptions")
        )
        builder.set_outputs(out)
        with self.assertRaisesRegex(
            CircleRuntimeValidationError,
            "2 per-channel scales on quantized dimension 1",
        ):
            CircleReferenceRuntime(builder.build()).run(())

    def test_non_positive_scale_is_rejected(self):
        builder = CircleModelBuilder()
        weights = builder.quantized_constant(
            "w", [1, 2], tensor_type=UINT8, scale=[0.0], zero_point=[0]
        )
        out = builder.activation("out", [2])
        builder.operator(
            "DEQUANTIZE", [weights], [out], options=_options("DequantizeOptions")
        )
        builder.set_outputs(out)
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "non-positive or non-finite"
        ):
            CircleReferenceRuntime(builder.build()).run(())

    def test_dequantize_without_qparams_is_rejected(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [2], dtype=np.int32)
        out = builder.activation("out", [2])
        builder.operator(
            "DEQUANTIZE", [x], [out], options=_options("DequantizeOptions")
        )
        builder.set_outputs(out)
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "carries no quantization parameters"
        ):
            CircleReferenceRuntime(builder.build()).run(
                (np.array([1, 2], dtype=np.int32),)
            )


class PackedFourBitConstantTest(unittest.TestCase):
    def _packed(self, tensor_type: int, payload: bytes, count: int) -> np.ndarray:
        builder = CircleModelBuilder()
        buffer = circle.Buffer.BufferT()
        buffer.data = np.frombuffer(payload, dtype=np.uint8).copy()
        builder.model.buffers.append(buffer)
        dtype = np.uint8 if tensor_type == UINT4 else np.int8
        index = builder._add_tensor(
            "packed", [count], dtype=dtype, buffer_index=len(builder.model.buffers) - 1
        )
        tensor = builder.subgraph.tensors[index]
        tensor.type = tensor_type
        tensor.quantization = builder.quantization([1.0], [0])
        out = builder.activation("out", [count])
        builder.operator(
            "DEQUANTIZE", [index], [out], options=_options("DequantizeOptions")
        )
        builder.set_outputs(out)
        return CircleReferenceRuntime(builder.build()).run(()).outputs[0]

    def test_uint4_low_nibble_first_with_odd_count(self):
        # bytes: 0x21 -> [1, 2], 0x03 -> [3] (upper nibble of the last byte unused)
        result = self._packed(UINT4, bytes([0x21, 0x03]), 3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_int4_sign_extension(self):
        # 0xF8 -> low nibble 8 -> -8, high nibble F -> -1; 0x07 -> 7
        result = self._packed(INT4, bytes([0xF8, 0x07]), 3)
        np.testing.assert_array_equal(result, [-8.0, -1.0, 7.0])

    def test_payload_length_is_validated(self):
        with self.assertRaisesRegex(CircleRuntimeValidationError, "requires 2 bytes"):
            self._packed(UINT4, bytes([0x21]), 3)


class FakeQuantizeModeTest(unittest.TestCase):
    """Compare fake-quantize execution against a hand-written float model."""

    def _quantized_fc(self):
        """x(u8) -> FULLY_CONNECTED(u8 weights, i32 bias) -> y(u8), plus RESHAPE."""

        builder = CircleModelBuilder()
        x = builder.activation(
            "x", [1, 2], dtype=np.uint8, quantization=builder.quantization([0.5], [128])
        )
        builder.subgraph.inputs.append(x)
        # weights [[1, -1], [2, 0]] stored as uint8 with zero point 128 and scale 1.0
        weights = builder.quantized_constant(
            "w",
            [[129, 127], [130, 128]],
            tensor_type=UINT8,
            scale=[1.0],
            zero_point=[128],
        )
        # bias [0.5, -1.0] stored as int32 with scale 0.5 (input scale * weight scale)
        bias = builder.quantized_constant(
            "b", [1, -2], tensor_type=INT32, scale=[0.5], zero_point=[0]
        )
        y = builder.activation(
            "y", [1, 2], dtype=np.uint8, quantization=builder.quantization([0.25], [0])
        )
        builder.operator(
            "FULLY_CONNECTED",
            [x, weights, bias],
            [y],
            options=_options("FullyConnectedOptions", keepNumDims=True),
        )
        shape = builder.const_i32("shape", [2])
        out = builder.activation(
            "out", [2], dtype=np.uint8, quantization=builder.quantization([0.25], [0])
        )
        builder.operator(
            "RESHAPE",
            [y, shape],
            [out],
            options=_options("ReshapeOptions", newShape=[2]),
        )
        builder.set_outputs(out)
        return CircleReferenceRuntime(builder.build())

    def test_fake_quantized_execution_matches_hand_computation(self):
        runtime = self._quantized_fc()
        value = np.array([[1.3, -0.4]], dtype=np.float32)
        result = runtime.run((value,), mode=ExecutionMode.FAKE_QUANTIZE).outputs[0]

        # input fake-quant on grid 0.5: 1.3 -> 1.5, -0.4 -> -0.5
        # FC: [1.5*1 + (-0.5)*(-1) + 0.5, 1.5*2 + (-0.5)*0 - 1.0] = [2.5, 2.0]
        # output grid 0.25 with zero point 0: unchanged; RESHAPE passes through.
        np.testing.assert_array_equal(result, np.array([2.5, 2.0], dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)

    def test_output_is_clamped_to_the_output_grid(self):
        runtime = self._quantized_fc()
        value = np.array([[70.0, 0.0]], dtype=np.float32)
        result = runtime.run((value,), mode=ExecutionMode.FAKE_QUANTIZE).outputs[0]
        # input clamps to (255-128)*0.5 = 63.5; FC -> [64.0, 126.0]; output max = 255*0.25 = 63.75
        np.testing.assert_array_equal(
            result, np.array([63.75, 63.75], dtype=np.float32)
        )

        value = np.array([[60.0, 0.0]], dtype=np.float32)
        result = runtime.run((value,), mode=ExecutionMode.FAKE_QUANTIZE).outputs[0]
        # 60.0 is representable; FC -> [60.5, 119.0]; only the second value clamps.
        np.testing.assert_array_equal(result, np.array([60.5, 63.75], dtype=np.float32))

    def test_fake_quantize_mode_requires_float_inputs_for_quantized_tensors(self):
        runtime = self._quantized_fc()
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, r"FLOAT32 \(dequantized UINT8\)"
        ):
            runtime.run(
                (np.array([[1, 2]], dtype=np.uint8),), mode=ExecutionMode.FAKE_QUANTIZE
            )

    def test_native_mode_rejects_quantized_fc_but_runs_quantize_dequantize(self):
        runtime = self._quantized_fc()
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "FULLY_CONNECTED on integer quantized"
        ):
            runtime.run((np.array([[1, 2]], dtype=np.uint8),))

        builder = CircleModelBuilder()
        x = builder.input("x", [2])
        q = builder.activation(
            "q", [2], dtype=np.uint8, quantization=builder.quantization([0.5], [128])
        )
        builder.operator("QUANTIZE", [x], [q], options=_options("QuantizeOptions"))
        dq = builder.activation("dq", [2])
        builder.operator("DEQUANTIZE", [q], [dq], options=_options("DequantizeOptions"))
        builder.set_outputs(q, dq)
        result = CircleReferenceRuntime(builder.build()).run(
            (np.array([0.3, -70.0], dtype=np.float32),)
        )
        np.testing.assert_array_equal(
            result.outputs[0], np.array([129, 0], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            result.outputs[1], np.array([0.5, -64.0], dtype=np.float32)
        )

    def test_probe_uses_fake_quantize_mode_for_quantized_models(self):
        runtime = self._quantized_fc()
        self.assertTrue(runtime.has_quantized_activations())
        result = runtime.probe_static_contracts()
        self.assertEqual(result.outputs[0].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
