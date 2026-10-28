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

"""Real full-model exports protect temporary preparation and the serialized ABI."""

import tempfile
import unittest
from pathlib import Path
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
        Qwen3VLModelPrefillExportAdapter,
    )

import numpy as np
import torch

from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.debug.wrapper_smoke import runner
from tico.quantization.recipes.debug.wrapper_smoke.case import ForwardInput
from tico.quantization.recipes.debug.wrapper_smoke.cases.qwen3_vl import (
    _has_qwen3_vl,
    QwenForConditionalGenerationCase,
    QwenModelCase,
    QwenTextModelCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.result import WrapperSmokeResult
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from torch.utils._pytree import tree_map


def _floating_input_dtype(sample, dtype):
    def convert(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(dtype=dtype)
        return value

    args, kwargs = tree_map(convert, (sample.args, dict(sample.kwargs)))
    return ForwardInput(args, kwargs)


def _prepare(case, *, dtype=None):
    model, _ = case.build({})
    if dtype is not None:
        model.to(dtype=dtype)
    prepared = case.prepare_model(model, {})
    case.after_prepare(prepared, {})
    for calibration in case.calibration_inputs(prepared, {}):
        if dtype is not None:
            calibration = _floating_input_dtype(calibration, dtype)
        case.forward(prepared, calibration)
    sample = case.eval_input(prepared, {})
    if dtype is not None:
        sample = _floating_input_dtype(sample, dtype)
    if dtype is torch.bfloat16:
        # BF16 activations with FP32 per-channel scales are supported on CPU.
        # Import only real calibrated candidates before conversion; BF16 scales
        # themselves retain the existing CPU execution rejection contract.
        for observer in prepared.modules():
            if (
                isinstance(observer, AffineObserverBase)
                and observer.channel_axis is not None
            ):
                candidate = observer.derive_candidate()
                assert candidate.scale is not None and candidate.zero_point is not None
                observer.load_qparams(
                    candidate.scale.float(), candidate.zero_point, lock=True
                )
    return case.convert_model(prepared, {}), sample


@unittest.skipUnless(_has_qwen3_vl().available, "Qwen3-VL unavailable")
class TestQwen3VLWrapperSmokeFullExport(unittest.TestCase):
    """Retain vision, DeepStack, text, and optional LM head with frozen source state."""

    @torch.no_grad()
    def test_bfloat16_full_prefill_matches_eager_and_preserves_static_inputs(self):
        for case_type in (QwenModelCase, QwenForConditionalGenerationCase):
            with self.subTest(case=case_type.__name__):
                case = case_type()
                quantized, sample = _prepare(case, dtype=torch.bfloat16)
                quantized.eval()
                before = _state_contract(quantized)
                modes = {
                    name: (module.training, getattr(module, "_mode", None))
                    for name, module in quantized.named_modules()
                }
                self.assertEqual(sample.kwargs["input_ids"].dtype, torch.int64)
                self.assertEqual(sample.kwargs["image_grid_thw"].dtype, torch.int64)
                self.assertEqual(sample.kwargs["pixel_values"].dtype, torch.bfloat16)
                for observer in quantized.modules():
                    if (
                        isinstance(observer, AffineObserverBase)
                        and observer.channel_axis is not None
                    ):
                        scale = observer.require_frozen().scale
                        assert scale is not None
                        self.assertEqual(scale.dtype, torch.float32)
                expected = case.output_tensor(case.forward(quantized, sample))
                self.assertEqual(expected.dtype, torch.bfloat16)

                def check_static_inputs(invocation):
                    adapter = cast(
                        "Qwen3VLModelPrefillExportAdapter", invocation.module
                    )
                    values = {
                        name: (value, value.detach().clone())
                        for name, value in adapter.static_inputs.named_buffers()
                    }
                    self.assertEqual(values["position_ids"][0].dtype, torch.int64)
                    for name in ("attention_mask", "cos", "sin"):
                        self.assertEqual(values[name][0].dtype, torch.float32)
                        self.assertEqual(values[name][0].device.type, "cpu")
                    actual = case.output_tensor(
                        invocation.module(*invocation.inputs.args)
                    )
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.device, expected.device)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for name, (original, saved) in values.items():
                        value = getattr(adapter.static_inputs, name)
                        self.assertIs(value, original)
                        self.assertEqual(value.dtype, saved.dtype)
                        self.assertEqual(value.device, saved.device)
                        torch.testing.assert_close(value, saved, atol=0, rtol=0)

                with case.export_context(quantized, sample, {}):
                    check_static_inputs(case.export_invocation(quantized, sample, {}))
                self.assertFalse(
                    any(
                        hasattr(module, "_static_export_inputs")
                        for module in quantized.modules()
                    )
                )
                self.assertEqual(before, _state_contract(quantized))
                self.assertEqual(
                    modes,
                    {
                        name: (module.training, getattr(module, "_mode", None))
                        for name, module in quantized.named_modules()
                    },
                )
                with self.assertRaisesRegex(RuntimeError, "BF16 context probe"):
                    with case.export_context(quantized, sample, {}):
                        check_static_inputs(
                            case.export_invocation(quantized, sample, {})
                        )
                        raise RuntimeError("BF16 context probe")
                self.assertFalse(
                    any(
                        hasattr(module, "_static_export_inputs")
                        for module in quantized.modules()
                    )
                )
                self.assertEqual(before, _state_contract(quantized))
                self.assertEqual(
                    modes,
                    {
                        name: (module.training, getattr(module, "_mode", None))
                        for name, module in quantized.named_modules()
                    },
                )

    @torch.no_grad()
    def test_full_cases_export_through_runner_and_replay_saved_files(self):
        for case_type in (
            QwenTextModelCase,
            QwenModelCase,
            QwenForConditionalGenerationCase,
        ):
            with (
                self.subTest(case=case_type.__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                case = case_type()
                quantized, sample = _prepare(case)
                before = _state_contract(quantized)
                expected = case.output_tensor(case.forward(quantized, sample))
                result = WrapperSmokeResult(case=case.name, passed=True)
                runner._export_circle(
                    case, quantized, sample, {}, Path(directory), result
                )
                self.assertTrue(result.passed, result.messages)
                with case.export_context(quantized, sample, {}):
                    invocation = case.export_invocation(quantized, sample, {})
                    actual = case.output_tensor(
                        invocation.module(
                            *invocation.inputs.args, **invocation.inputs.kwargs
                        )
                    )
                    # The fixed-image adapter preserves every eager quantization
                    # boundary; this assertion is independent of Circle reference
                    # GATHER's documented omission of an activation grid round trip.
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    if isinstance(case, QwenModelCase):
                        adapter = cast(
                            "Qwen3VLModelPrefillExportAdapter", invocation.module
                        )
                        # 4x4 merged image at offset zero, followed by text.
                        image_positions = torch.stack(
                            (
                                torch.zeros(16, dtype=torch.long),
                                torch.arange(4).repeat_interleave(4),
                                torch.arange(4).repeat(4),
                            )
                        )
                        tail = torch.arange(4, 38).expand(3, -1)
                        positions = torch.cat((image_positions, tail), dim=1).unsqueeze(
                            1
                        )
                        torch.testing.assert_close(
                            adapter.static_inputs.position_ids,
                            positions,
                            atol=0,
                            rtol=0,
                        )
                        original_kwargs = dict(
                            sample.kwargs,
                            position_ids=adapter.static_inputs.position_ids,
                        )
                        original = case.output_tensor(quantized(**original_kwargs))
                        torch.testing.assert_close(actual, original, atol=0, rtol=0)
                        self.assertEqual(
                            len(adapter.layers),
                            case.qwen_cfg.text_config.num_hidden_layers,
                        )
                        altered = invocation.module(
                            invocation.inputs.args[0],
                            torch.zeros_like(invocation.inputs.args[1]),
                        )
                        self.assertFalse(torch.equal(altered, actual))
                        self.assertEqual(
                            adapter.vision.wrapped.deepstack_visual_indexes, [0, 1]
                        )
                self.assertFalse(
                    any(
                        hasattr(module, "_static_export_inputs")
                        for module in quantized.modules()
                    )
                )
                path = Path(result.artifacts["circle"])
                self.assertGreater(path.stat().st_size, 0)
                document = CircleDocument.load(path)
                document.verify(raise_on_error=True)
                runtime = CircleReferenceRuntime(document)
                inputs = [sample.kwargs["input_ids"].numpy()]
                self.assertEqual(runtime.input_tensors[0].dtype, np.int64)
                if isinstance(case, QwenModelCase):
                    inputs.append(sample.kwargs["pixel_values"].numpy())
                else:
                    # Scalar kwargs are serialized graph inputs too; replay the
                    # exact cache-free tuple-output ABI rather than dropping them.
                    self.assertEqual(
                        [tensor.name for tensor in runtime.input_tensors[1:]],
                        ["tico::use_cache", "tico::return_dict"],
                    )
                    self.assertTrue(
                        all(
                            tensor.dtype == np.bool_
                            for tensor in runtime.input_tensors[1:]
                        )
                    )
                    inputs.extend((np.asarray(False), np.asarray(False)))
                self.assertEqual(len(runtime.input_tensors), len(inputs))
                (output,) = runtime.run(
                    inputs, mode=ExecutionMode.FAKE_QUANTIZE
                ).outputs
                self.assertEqual(output.shape, tuple(expected.shape))
                self.assertEqual(output.dtype, np.float32)
                self.assertTrue(np.isfinite(output).all())
                self.assertTrue(np.any(output != 0))
                self.assertEqual(before, _state_contract(quantized))
                self.assertFalse(
                    any(
                        hasattr(module, "_static_export_inputs")
                        for module in quantized.modules()
                    )
                )

    @torch.no_grad()
    def test_context_restores_source_on_failure_and_adapter_rejects_mode_drift(self):
        from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
            Qwen3VLModelPrefillExportAdapter,
        )

        case = QwenModelCase()
        quantized, sample = _prepare(case)
        before = _state_contract(quantized)
        with self.assertRaisesRegex(ValueError, "prepared_static_export"):
            Qwen3VLModelPrefillExportAdapter(quantized.wrapped, grid_thw=case.thw)
        with self.assertRaisesRegex(ValueError, "input length"):
            with case.export_context(quantized, sample, {}):
                invocation = case.export_invocation(quantized, sample, {})
                ids, pixels = invocation.inputs.args
                invocation.module(ids[:, :-1], pixels)
        self.assertFalse(
            any(
                hasattr(module, "_static_export_inputs")
                for module in quantized.modules()
            )
        )
        self.assertEqual(before, _state_contract(quantized))
        with case.export_context(quantized, sample, {}):
            with self.assertRaisesRegex(ValueError, "prepared text mRoPE grid"):
                Qwen3VLModelPrefillExportAdapter(quantized.wrapped, grid_thw=(1, 4, 4))
            invocation = case.export_invocation(quantized, sample, {})
            owner = cast(
                "Qwen3VLModelPrefillExportAdapter", invocation.module
            ).text_model
            previous = owner._mode
            try:
                owner._mode = Mode.NO_QUANT
                with self.assertRaisesRegex(RuntimeError, "mode does not match"):
                    invocation.module(*invocation.inputs.args)
            finally:
                owner._mode = previous
        self.assertEqual(before, _state_contract(quantized))


if __name__ == "__main__":
    unittest.main()
