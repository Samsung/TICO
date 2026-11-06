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

"""Real Qwen fixed-span multimodal embedding export and native CPU execution.

The existing adapter accepts every visual row, with no valid-count/mask input.
Its unrotated result is FLOAT32; only the production rotation owns a final Q.
Decoder integer input conversion is a separate, explicitly tested host handoff.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import (
    CircleReferenceRuntime,
    CircleRuntimeValidationError,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)
from tico.circle.runtime.kernels import quantize as quantize_kernels
from tico.circle.runtime.quantization import dequantize_array, quantize_array
from tico.quantization.recipes.debug.wrapper_smoke.cases.qwen3_vl import (
    _text_static_decoder_sample,
)
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLMultimodalEmbeddingExportAdapter,
    Qwen3VLVisionPrefillExportAdapter,
)

from test.support.quantization.qwen_embedding import AffineGrid, build_quantized_qwen


@unittest.skipUnless(has_transformers_for("qwen3-vl"), "Qwen3-VL unavailable")
class QwenMultimodalEmbeddingIntegrationTest(unittest.TestCase):
    def _export(self, fixture, ids, images, directory, name="multimodal"):
        adapter = Qwen3VLMultimodalEmbeddingExportAdapter(
            fixture.owner, visual_start_idx=2
        ).eval()
        path = Path(directory) / f"{name}.circle"
        tico.convert_to_file(
            adapter,
            (ids, images),
            path,
            config=tico.CompileConfigV1(preserve_embedding_quantization=True),
        )
        document = CircleDocument.load(path)
        self.assertTrue(document.verify().ok)
        restored_during_load = []
        original = quantize_kernels.dequantize_array

        def observe(value, tensor):
            restored_during_load.append(tuple(value.shape))
            return original(value, tensor)

        with mock.patch.object(quantize_kernels, "dequantize_array", observe):
            runtime = CircleReferenceRuntime(document)
        self.assertNotIn((16, 8), restored_during_load)
        self.assertFalse(runtime._dequantized_constants)
        return adapter, runtime

    def _assert_structure(self, runtime, fixture, ids, images):
        self.assertEqual(runtime.input_tensors[0].shape, tuple(ids.shape))
        self.assertEqual(runtime.input_tensors[0].dtype, np.dtype("int64"))
        self.assertFalse(runtime.input_tensors[0].is_quantized)
        image_input = runtime.input_tensors[1]
        self.assertEqual(image_input.shape, tuple(images.shape))
        self.assertEqual(image_input.dtype, np.dtype("float32"))
        self.assertFalse(image_input.is_quantized)
        self.assertEqual(len(runtime.rowwise_gather_plans), 1)
        plan = next(iter(runtime.rowwise_gather_plans.values()))
        weight = runtime.program.tensor(plan.weight_index)
        self.assertEqual(weight.dtype, np.dtype("uint8"))
        self.assertEqual(weight.shape, (16, 8))
        self.assertEqual(weight.quantization.quantized_dimension, 0)
        np.testing.assert_array_equal(weight.constant, fixture.weight_codes)
        np.testing.assert_array_equal(
            weight.quantization.scale, fixture.weight_grid.scale
        )
        np.testing.assert_array_equal(
            weight.quantization.zero_point, fixture.weight_grid.zero_point
        )
        tables = [
            tensor
            for tensor in runtime.program.tensors
            if tensor.is_constant and tensor.shape == (16, 8)
        ]
        self.assertEqual([tensor.index for tensor in tables], [weight.index])
        # Text QA remains before visual insertion, including the DQ that
        # exposes its rounded/clipped values to FLOAT32 layout operations.
        embedding_quantizers = [
            op
            for op in runtime.program.operators
            if op.name == "QUANTIZE"
            and runtime.program.tensor(op.outputs[0]).quantization.scale
            == tuple(fixture.embedding_grid.scale.reshape(-1))
        ]
        self.assertTrue(embedding_quantizers)
        self.assertTrue(
            any(
                dq.name == "DEQUANTIZE" and dq.inputs == quantizer.outputs
                for quantizer in embedding_quantizers
                for dq in runtime.program.operators
            )
        )
        output = runtime.output_tensors[0]
        self.assertEqual(output.shape, (1, ids.shape[1], 8))
        if fixture.rotation is None:
            self.assertEqual(output.dtype, np.dtype("float32"))
            self.assertFalse(output.is_quantized)
        else:
            self.assertEqual(output.dtype, np.dtype("int16"))
            np.testing.assert_array_equal(
                output.quantization.scale, fixture.rotation_output_grid.scale
            )
            np.testing.assert_array_equal(
                output.quantization.zero_point,
                fixture.rotation_output_grid.zero_point,
            )
        return weight

    def _run_checked(self, runtime, ids, images):
        restored = []
        saved_ids, saved_images = ids.clone(), images.clone()
        original = quantize_kernels.dequantize_array

        def observe(value, tensor):
            restored.append(tuple(value.shape))
            return original(value, tensor)

        with mock.patch.object(quantize_kernels, "dequantize_array", observe):
            actual = runtime.run(
                (ids.numpy(), images.numpy()), mode=ExecutionMode.NATIVE
            ).outputs[0]
        self.assertNotIn((16, 8), restored)
        self.assertFalse(runtime._dequantized_constants)
        oracle = (
            CircleReferenceRuntime(runtime.program, enable_rowwise_gather=False)
            .run((ids.numpy(), images.numpy()), mode=ExecutionMode.NATIVE)
            .outputs[0]
        )
        np.testing.assert_array_equal(actual, oracle)
        torch.testing.assert_close(ids, saved_ids, atol=0, rtol=0)
        torch.testing.assert_close(images, saved_images, atol=0, rtol=0)
        return actual

    @torch.no_grad()
    def test_mixed_text_image_order_tail_padding_and_real_spin_rotation(self):
        # Four ordered visual slots occupy one contiguous span. Tail token 0
        # is ordinary padding data at this stage: no token-validity mask exists.
        ids = torch.tensor([[15, 0, 14, 14, 14, 14, 5, 0]], dtype=torch.int64)
        images = torch.linspace(-1.137, 1.793, 32).reshape(4, 8)
        for rotation in (False, True):
            with self.subTest(rotation=rotation), tempfile.TemporaryDirectory() as tmp:
                fixture = build_quantized_qwen(rotation=rotation)
                self.assertIs(
                    fixture.embedding.module.weight,
                    fixture.owner.lm_head.wrapped.module.weight,
                )
                before_weight = fixture.embedding.module.weight.detach().clone()
                before_grid = fixture.embedding.weight_obs.require_frozen()
                adapter, runtime = self._export(fixture, ids, images, tmp)
                weight = self._assert_structure(runtime, fixture, ids, images)
                codes = weight.constant.copy()
                actual = self._run_checked(runtime, ids, images)
                text = fixture.text_values(ids.numpy())
                merged = text.copy()
                merged[:, 2:6, :] = images.numpy()
                expected = fixture.rotate(merged) if rotation else merged
                np.testing.assert_array_equal(actual, expected)
                # Grid-rounded text remains on both sides of the visual span;
                # no image-row permutation, common scale, or full-table lookup.
                if not rotation:
                    np.testing.assert_array_equal(actual[:, :2], text[:, :2])
                    np.testing.assert_array_equal(actual[:, 6:], text[:, 6:])
                    np.testing.assert_array_equal(actual[0, 2:6], images.numpy())
                    self.assertTrue(
                        np.any(
                            actual[0, 2:6]
                            != fixture.embedding_grid.fake_quantize(images.numpy())
                        )
                    )
                else:
                    self.assertFalse(torch.equal(fixture.r1, torch.eye(8)))
                    self.assertTrue(
                        np.any(actual != fixture.rotation_output_grid.quantize(merged))
                    )
                eager = adapter(ids, images).numpy()
                eager_codes = (
                    fixture.rotation_output_grid.quantize(eager, ties_even=True)
                    if rotation
                    else eager
                )
                # WrapQ uses ties-even, whereas Circle uses half-away. Compare
                # each arithmetic contract to its own independent frozen oracle;
                # text token 0 deliberately exercises a halfway value in QA.
                merged_even = fixture.text_values(ids.numpy(), ties_even=True)
                merged_even[:, 2:6, :] = images.numpy()
                eager_expected = (
                    fixture.rotate(merged_even, ties_even=True)
                    if rotation
                    else merged_even
                )
                np.testing.assert_array_equal(eager_codes, eager_expected)
                torch.testing.assert_close(
                    fixture.embedding.module.weight, before_weight, atol=0, rtol=0
                )
                after_grid = fixture.embedding.weight_obs.require_frozen()
                torch.testing.assert_close(
                    after_grid.scale, before_grid.scale, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    after_grid.zero_point, before_grid.zero_point, atol=0, rtol=0
                )
                np.testing.assert_array_equal(weight.constant, codes)

    @torch.no_grad()
    def test_fixed_visual_capacity_consumes_sentinels_and_rejects_new_shape(self):
        fixture = build_quantized_qwen()
        ids = torch.tensor([[1, 15, 14, 14, 14, 14, 2, 0]])
        images = torch.full((4, 8), 17.375)
        images[:2] = torch.tensor([0.13, -0.29])[:, None]
        with tempfile.TemporaryDirectory() as tmp:
            _, runtime = self._export(fixture, ids, images, tmp)
            actual = self._run_checked(runtime, ids, images)
            # There is no valid-count or image mask in this ABI. Padded visual
            # capacity is consumed verbatim, so masking it requires a new ABI.
            self.assertEqual(len(runtime.input_tensors), 2)
            np.testing.assert_array_equal(actual[0, 4:6], images.numpy()[2:])
            with self.assertRaises(CircleRuntimeValidationError):
                runtime.run((ids.numpy(), images.numpy()[:2]))
            with self.assertRaises(CircleRuntimeValidationError):
                runtime.run((ids.numpy(), np.empty((0, 8), np.float32)))
            with self.assertRaises(CircleRuntimeValidationError):
                runtime.run((ids.numpy(), images.numpy().astype(np.int16)))
            for invalid in (-1, 16):
                invalid_ids = ids.numpy().copy()
                invalid_ids[0, 0] = invalid
                with self.assertRaisesRegex(
                    CircleRuntimeValidationError, "outside.*0, 16"
                ):
                    runtime.run((invalid_ids, images.numpy()))

    @torch.no_grad()
    def test_separately_exported_zero_visual_rows_preserve_all_text(self):
        fixture = build_quantized_qwen()
        ids = torch.tensor([[15, 0, 2, 1, 7, 7, 0, 3]])
        images = torch.empty((0, 8), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            adapter, runtime = self._export(fixture, ids, images, tmp, "zero_visual")
            np.testing.assert_array_equal(
                adapter(ids, images).numpy(),
                fixture.text_values(ids.numpy(), ties_even=True),
            )
            actual = self._run_checked(runtime, ids, images)
            np.testing.assert_array_equal(actual, fixture.text_values(ids.numpy()))

    @torch.no_grad()
    def test_vision_output_grid_is_dequantized_for_float_image_input(self):
        fixture = build_quantized_qwen()
        qvision = fixture.owner.model.wrapped.visual.wrapped
        adapter = Qwen3VLVisionPrefillExportAdapter(qvision, grid_thw=(1, 4, 4)).eval()
        pixels = torch.linspace(-1, 1, 16 * 12).reshape(1, 16, 12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vision.circle"
            tico.convert_to_file(
                adapter,
                (pixels,),
                path,
                config=tico.CompileConfigV1(preserve_embedding_quantization=True),
            )
            vision = CircleReferenceRuntime(CircleDocument.load(path))
            output = vision.output_tensors[0]
            self.assertEqual(output.dtype, np.dtype("int16"))
            self.assertEqual(output.shape, (4, 8))
            assert output.quantization is not None
            grid = AffineGrid.from_observer(
                qvision.merger.wrapped.linear_fc2.wrapped.obs_act_out
            )
            np.testing.assert_array_equal(output.quantization.scale, grid.scale)
            np.testing.assert_array_equal(
                output.quantization.zero_point, grid.zero_point
            )
            # Vision's quantized arithmetic is tested here only in fake mode.
            # Materialize its integer ABI using its own output grid, then use
            # the production DQ helper for the CPU multimodal FLOAT32 input.
            vision_float = vision.run(
                (pixels.numpy(),), mode=ExecutionMode.FAKE_QUANTIZE
            ).outputs[0]
            vision_codes = quantize_array(vision_float, output)
            image_float = dequantize_array(vision_codes, output)
            np.testing.assert_array_equal(image_float, grid.dequantize(vision_codes))
            ids = torch.tensor([[15, 0, 14, 14, 14, 14, 2, 0]])
            images = torch.from_numpy(image_float)
            _, runtime = self._export(fixture, ids, images, tmp)
            self.assertEqual(runtime.input_tensors[1].dtype, np.dtype("float32"))
            actual = self._run_checked(runtime, ids, images)
            np.testing.assert_array_equal(actual[0, 2:6], image_float)
            with self.assertRaises(CircleRuntimeValidationError):
                runtime.run((ids.numpy(), vision_codes))

    @torch.no_grad()
    def test_native_multimodal_output_to_actual_prefill_input_grid(self):
        ids = torch.tensor([[15, 0, 14, 14, 14, 14, 2, 0]])
        images = torch.linspace(-0.73, 0.91, 32).reshape(4, 8)
        for rotation in (False, True):
            with self.subTest(rotation=rotation), tempfile.TemporaryDirectory() as tmp:
                fixture = build_quantized_qwen(rotation=rotation)
                _, runtime = self._export(fixture, ids, images, tmp)
                native = self._run_checked(runtime, ids, images)
                decoded = (
                    dequantize_array(native, runtime.output_tensors[0])
                    if rotation
                    else native
                )
                sample = _text_static_decoder_sample(
                    fixture.decoder, batch_size=1, query_len=8, key_len=8, mask_rank=4
                )
                prefill = fixture.decoder.as_export_module(
                    "prefill", return_kv=False
                ).eval()
                path = Path(tmp) / "decoder_prefill.circle"
                tico.convert_to_file(
                    prefill,
                    (
                        torch.from_numpy(decoded),
                        sample.kwargs["attention_mask"],
                        sample.kwargs["position_embeddings"],
                    ),
                    path,
                    config=tico.CompileConfigV1(preserve_embedding_quantization=True),
                )
                consumer = CircleReferenceRuntime(CircleDocument.load(path))
                hidden = consumer.input_tensors[0]
                self.assertEqual(hidden.dtype, np.dtype("int16"))
                self.assertEqual(hidden.shape, (1, 8, 8))
                assert hidden.quantization is not None
                np.testing.assert_array_equal(
                    hidden.quantization.scale, fixture.decoder_input_grid.scale
                )
                np.testing.assert_array_equal(
                    hidden.quantization.zero_point,
                    fixture.decoder_input_grid.zero_point,
                )
                self.assertNotEqual(
                    tuple(fixture.decoder_input_grid.scale.reshape(-1)),
                    tuple(fixture.output_grid.scale.reshape(-1)),
                )
                # Explicit host conversion uses the actual consumer contract.
                codes = quantize_array(decoded, hidden)
                np.testing.assert_array_equal(
                    codes, fixture.decoder_input_grid.quantize(decoded)
                )
                consumer_float = dequantize_array(codes, hidden)
                inputs = [consumer_float, sample.kwargs["attention_mask"].numpy()]
                inputs.extend(
                    value.numpy() for value in sample.kwargs["position_embeddings"]
                )
                # The decoder remains a quantized arithmetic graph. Its fake
                # mode verifies the separate consumer path, not native NPU math.
                result = consumer.run(inputs, mode=ExecutionMode.FAKE_QUANTIZE).outputs[
                    0
                ]
                self.assertEqual(result.shape, (1, 8, 8))
                self.assertTrue(np.all(np.isfinite(result)))
                eager_consumer = prefill(
                    torch.from_numpy(consumer_float),
                    sample.kwargs["attention_mask"],
                    sample.kwargs["position_embeddings"],
                ).numpy()
                np.testing.assert_array_equal(result, eager_consumer)
                with self.assertRaises(UnsupportedCircleOperatorError):
                    consumer.run(
                        (
                            codes,
                            *[
                                (
                                    quantize_array(value, tensor)
                                    if tensor.is_quantized_integer
                                    else value
                                )
                                for tensor, value in zip(
                                    consumer.input_tensors[1:], inputs[1:]
                                )
                            ],
                        ),
                        mode=ExecutionMode.NATIVE,
                    )


if __name__ == "__main__":
    unittest.main()
