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

"""Real Qwen3-VL token stages produce native INT16, with genuine SpinQuant."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from tico import CompileConfigV1
from tico.circle import CircleDocument
from tico.circle.runtime import (
    CircleReferenceRuntime,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)
from tico.circle.runtime.kernels import quantize
from tico.circle.runtime.quantization import dequantize_array, quantize_array
from tico.quantization.recipes.debug.wrapper_smoke.cases.qwen3_vl import _has_qwen3_vl
from tico.quantization.recipes.export.circle import convert_and_save_circle
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    make_token_embedding_dynamic_shapes,
    register_fake_quant_meta_kernels_for_dynamic_export,
)
from tico.utils.model import CircleModel

from test.support.quantization.qwen_embedding import build_quantized_qwen


@unittest.skipUnless(_has_qwen3_vl().available, "optional Qwen3-VL modules unavailable")
class QwenTokenEmbeddingIntegrationTest(unittest.TestCase):
    fixtures: dict[bool, Any]

    @classmethod
    def setUpClass(cls):
        cls.fixtures = {
            rotation: build_quantized_qwen(rotation=rotation)
            for rotation in (False, True)
        }

    def _export_token(self, fixture, path):
        from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
            Qwen3VLTextEmbeddingExportAdapter,
        )

        adapter = Qwen3VLTextEmbeddingExportAdapter(fixture.owner).eval()
        register_fake_quant_meta_kernels_for_dynamic_export()
        convert_and_save_circle(
            adapter,
            (torch.tensor([[0, 15, 7, 3]]),),
            path,
            config=CompileConfigV1(preserve_embedding_quantization=True),
            dynamic_shapes=make_token_embedding_dynamic_shapes(16),
            strict=False,
        )
        document = CircleDocument.load(path)
        self.assertTrue(document.verify().ok)
        original = quantize.dequantize_array
        restored = []

        def observe(value, tensor):
            restored.append(value.shape)
            self.assertNotEqual(value.shape, (16, 8))
            return original(value, tensor)

        with patch.object(quantize, "dequantize_array", side_effect=observe):
            runtime = CircleReferenceRuntime(document)
        self.assertEqual(restored, [])
        self.assertEqual(runtime._dequantized_constants, {})
        return runtime, adapter

    def test_real_lifecycle_spinquant_and_token_stage_native_roundtrip(self):
        for rotation, fixture in self.fixtures.items():
            with self.subTest(rotation=rotation), TemporaryDirectory() as directory:
                self.assertEqual(fixture.weight_codes.dtype, np.uint8)
                self.assertEqual(fixture.weight_grid.axis, 0)
                self.assertEqual(fixture.weight_grid.scale.shape, (16,))
                self.assertGreater(np.unique(fixture.weight_grid.scale).size, 1)
                self.assertGreater(np.unique(fixture.weight_grid.zero_point).size, 1)
                self.assertEqual(fixture.embedding_grid.dtype, np.dtype(np.int16))
                original_weight = fixture.embedding.module.weight.detach().clone()
                lm = fixture.owner.lm_head.wrapped
                self.assertEqual(
                    lm.module.weight.data_ptr(),
                    fixture.embedding.module.weight.data_ptr(),
                )
                lm_grid = lm.obs_weight.require_frozen()
                if rotation:
                    self.assertFalse(
                        torch.equal(fixture.r1, torch.eye(8, dtype=fixture.r1.dtype))
                    )
                    torch.testing.assert_close(
                        fixture.r1.T @ fixture.r1,
                        torch.eye(8, dtype=torch.float64),
                        atol=1e-14,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        fixture.rotation.module.weight,
                        fixture.r1.T.float(),
                        atol=0,
                        rtol=0,
                    )
                    self.assertNotEqual(
                        float(fixture.embedding_grid.scale),
                        float(fixture.rotation_input_grid.scale),
                    )
                path = Path(directory) / "token_embedding.q.circle"
                runtime, adapter = self._export_token(fixture, path)
                self.assertTrue(runtime.rowwise_gather_plans)
                table = next(
                    t
                    for t in runtime.program.tensors
                    if t.is_constant and t.shape == (16, 8)
                )
                self.assertEqual(table.dtype, np.dtype(np.uint8))
                np.testing.assert_array_equal(table.constant, fixture.weight_codes)
                self.assertFalse(
                    any(
                        t.is_constant
                        and t.shape == (16, 8)
                        and t.dtype != np.dtype(np.uint8)
                        for t in runtime.program.tensors
                    )
                )
                output = runtime.output_tensors[0]
                self.assertEqual(output.dtype, np.dtype(np.int16))
                np.testing.assert_array_equal(
                    output.quantization.scale, fixture.output_grid.scale.reshape(-1)
                )
                np.testing.assert_array_equal(
                    output.quantization.zero_point,
                    fixture.output_grid.zero_point.reshape(-1),
                )
                grids = [fixture.embedding_grid]
                if rotation:
                    grids.extend(
                        (fixture.rotation_input_grid, fixture.rotation_output_grid)
                    )
                    linear = next(
                        op
                        for op in runtime.program.operators
                        if op.name == "FULLY_CONNECTED"
                    )
                    for index in (*linear.inputs[:2], *linear.outputs):
                        tensor = runtime.program.tensor(index)
                        self.assertEqual(tensor.dtype, np.dtype(np.float32))
                        self.assertFalse(tensor.is_quantized)
                quantizers = [
                    op for op in runtime.program.operators if op.name == "QUANTIZE"
                ]
                self.assertEqual(len(quantizers), len(grids))
                for op, grid in zip(quantizers, grids):
                    tensor = runtime.program.tensor(op.outputs[0])
                    np.testing.assert_array_equal(
                        tensor.quantization.scale, grid.scale.reshape(-1)
                    )
                    np.testing.assert_array_equal(
                        tensor.quantization.zero_point, grid.zero_point.reshape(-1)
                    )
                for tokens in ([[15]], [[0, 15, 7, 3, 0, 7]]):
                    ids = np.array(tokens, dtype=np.int64)
                    # Oracle snapshots predate export and use each original grid.
                    expected = fixture.rotate(fixture.text_values(ids))
                    restored = []
                    original_dequantize = quantize.dequantize_array

                    def observe(value, tensor):
                        restored.append(value.shape)
                        self.assertNotEqual(value.shape, (16, 8))
                        return original_dequantize(value, tensor)

                    with patch.object(
                        quantize, "dequantize_array", side_effect=observe
                    ):
                        result = runtime.run([ids], mode=ExecutionMode.NATIVE).outputs[
                            0
                        ]
                    np.testing.assert_array_equal(result, expected)
                    self.assertEqual(result.dtype, np.int16)
                    self.assertEqual(result.shape, (*ids.shape, 8))
                    oracle = CircleReferenceRuntime(
                        runtime.program, enable_rowwise_gather=False
                    )
                    np.testing.assert_array_equal(
                        result, oracle.run([ids], mode=ExecutionMode.NATIVE).outputs[0]
                    )
                    # CircleModel returns native codes; it does not automatically DQ.
                    raw = CircleModel.load(str(path), runtime="reference")(
                        torch.from_numpy(ids)
                    )
                    np.testing.assert_array_equal(np.asarray(raw), expected)
                    self.assertEqual(runtime._dequantized_constants, {})
                    # Independent WrapQ arithmetic uses ties-to-even at each grid.
                    expected_wrap = fixture.output_grid.dequantize(
                        fixture.rotate(
                            fixture.text_values(ids, ties_even=True), ties_even=True
                        )
                    )
                    actual_wrap = adapter(torch.from_numpy(ids)).detach().numpy()
                    np.testing.assert_array_equal(actual_wrap, expected_wrap)
                torch.testing.assert_close(
                    fixture.embedding.module.weight, original_weight, atol=0, rtol=0
                )
                after = lm.obs_weight.require_frozen()
                torch.testing.assert_close(after.scale, lm_grid.scale, atol=0, rtol=0)
                torch.testing.assert_close(
                    after.zero_point, lm_grid.zero_point, atol=0, rtol=0
                )

    def test_decoder_handoff_preserves_independent_frozen_input_contract(self):
        import tico

        for rotation, fixture in self.fixtures.items():
            with self.subTest(rotation=rotation), TemporaryDirectory() as directory:
                runtime, _ = self._export_token(
                    fixture, Path(directory) / "token_embedding.q.circle"
                )
                ids = np.array([[15]], dtype=np.int64)
                producer = runtime.run([ids], mode=ExecutionMode.NATIVE).outputs[0]
                decoder = fixture.decoder.as_export_module(
                    "decode", return_kv=False
                ).eval()
                hidden = torch.zeros(1, 1, 8)
                mask = torch.zeros(1, 1, 1, 2)
                cos = torch.ones(1, 1, 8)
                sin = torch.zeros(1, 1, 8)
                past = (torch.zeros(1, 1, 1, 8), torch.zeros(1, 1, 1, 8))
                converted = tico.convert(
                    decoder,
                    (hidden,),
                    kwargs=dict(
                        attention_mask=mask,
                        position_embeddings=(cos, sin),
                        past_key_values=past,
                    ),
                    strict=False,
                    config=CompileConfigV1(preserve_embedding_quantization=True),
                )
                path = Path(directory) / "decoder_layer_decode_0.q.circle"
                converted.save(path)
                consumer = CircleReferenceRuntime(CircleDocument.load(path))
                input_tensor = consumer.input_tensors[0]
                assert input_tensor.quantization is not None
                self.assertEqual(input_tensor.dtype, np.dtype(np.int16))
                self.assertEqual(input_tensor.shape, (1, 1, 8))
                np.testing.assert_array_equal(
                    input_tensor.quantization.scale,
                    fixture.decoder_input_grid.scale.reshape(-1),
                )
                self.assertNotEqual(
                    runtime.output_tensors[0].quantization.scale,
                    input_tensor.quantization.scale,
                )
                real = dequantize_array(producer, runtime.output_tensors[0])
                consumer_codes = quantize_array(real, input_tensor)
                independent = fixture.decoder_input_grid.quantize(
                    fixture.output_grid.dequantize(producer)
                )
                np.testing.assert_array_equal(consumer_codes, independent)
                self.assertFalse(np.array_equal(consumer_codes, producer))
                bridge = dequantize_array(consumer_codes, input_tensor)
                inputs = [
                    bridge,
                    mask.numpy(),
                    cos.numpy(),
                    sin.numpy(),
                    past[0].numpy(),
                    past[1].numpy(),
                ]
                # The decoder remains the existing quantized arithmetic graph.
                # Its execution here is explicitly a fake-quant semantic check.
                native_inputs = [
                    (
                        quantize_array(value, tensor)
                        if tensor.is_quantized_integer
                        else value
                    )
                    for value, tensor in zip(inputs, consumer.input_tensors)
                ]
                with self.assertRaises(UnsupportedCircleOperatorError):
                    consumer.run(native_inputs, mode=ExecutionMode.NATIVE)
                result = consumer.run(inputs, mode=ExecutionMode.FAKE_QUANTIZE).outputs[
                    0
                ]
                self.assertEqual(result.shape, (1, 1, 8))
                self.assertEqual(result.dtype, np.float32)
                eager = (
                    decoder(
                        torch.from_numpy(bridge),
                        attention_mask=mask,
                        position_embeddings=(cos, sin),
                        past_key_values=past,
                    )
                    .detach()
                    .numpy()
                )
                np.testing.assert_array_equal(result, eager)


if __name__ == "__main__":
    unittest.main()
