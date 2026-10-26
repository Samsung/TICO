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

"""Guard the host-owned static decoder ABI before Dynamo traces a smoke case."""

import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from tico.quantization.recipes.debug.wrapper_smoke import runner
from tico.quantization.recipes.debug.wrapper_smoke.case import ForwardInput
from tico.quantization.recipes.debug.wrapper_smoke.cases.llama import (
    _has_llama,
    LlamaAttentionDecodeCase,
    LlamaAttentionPrefillCase,
    LlamaDecoderLayerDecodeCase,
    LlamaDecoderLayerPrefillCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.cases.qwen3_vl import (
    _has_qwen3_vl,
    QwenTextAttentionDecodeCase,
    QwenTextAttentionPrefillCase,
    QwenTextDecoderLayerDecodeCase,
    QwenTextDecoderLayerPrefillCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.result import WrapperSmokeResult

_LLAMA_CASES = (
    LlamaAttentionPrefillCase,
    LlamaAttentionDecodeCase,
    LlamaDecoderLayerPrefillCase,
    LlamaDecoderLayerDecodeCase,
)
_QWEN_CASES = (
    QwenTextAttentionPrefillCase,
    QwenTextAttentionDecodeCase,
    QwenTextDecoderLayerPrefillCase,
    QwenTextDecoderLayerDecodeCase,
)
_LLAMA = _has_llama()
_QWEN = _has_qwen3_vl()


def _prepare_case(case):
    """Exercise the real tiny prepare/calibrate/convert lifecycle."""
    cfg: dict[str, Any] = {}
    case.validate_config(cfg)
    model, reference = case.build(cfg)
    prepared = case.prepare_model(model, cfg)
    case.after_prepare(prepared, cfg)
    with torch.no_grad():
        calibration = case.calibration_inputs(prepared, cfg)
        case.forward(prepared, calibration[0])
        sample = case.eval_input(prepared, cfg)
        if case.compare_reference_source == "reference":
            case.reference_forward(reference, sample)
        case.forward(prepared, sample)
        quantized = case.convert_model(prepared, cfg)
        case.forward(quantized, sample)
    return quantized, sample, calibration


def _hidden(sample):
    return sample.args[0] if sample.args else sample.kwargs["hidden_states"]


class _ContractAssertions(unittest.TestCase):
    """Assertions shared by the two decoder families without input builders."""

    def assert_causal_mask(self, sample, fill_value):
        hidden = _hidden(sample)
        mask = sample.kwargs["attention_mask"]
        self.assertIsInstance(mask, torch.Tensor)
        self.assertTrue(mask.is_floating_point())
        self.assertTrue(torch.isfinite(mask).all())
        batch, query = hidden.shape[:2]
        self.assertEqual(mask.shape[0], batch)
        self.assertEqual(tuple(mask.shape[-2:]), (query, query))
        planes = mask.reshape(-1, query, query)
        visible = torch.ones(query, query, dtype=torch.bool).tril()
        self.assertTrue(torch.all(planes[:, visible] == 0))
        self.assertTrue(torch.all(planes[:, ~visible] == fill_value))

    def assert_invocation(self, case, invocation):
        invocation.validate_static_decoder_inputs()
        sample = invocation.inputs
        self.assertEqual(len(sample.args), 1)
        self.assertIn("attention_mask", sample.kwargs)
        self.assertIn("position_embeddings", sample.kwargs)
        bound = inspect.signature(invocation.module.forward).bind(
            *sample.args, **dict(sample.kwargs)
        )
        hidden = bound.arguments["hidden_states"]
        mask = bound.arguments["attention_mask"]
        cos, sin = bound.arguments["position_embeddings"]
        config = case.text_cfg if case.name.startswith("qwen") else case.config
        self.assertEqual(hidden.shape[0], 1)
        self.assertEqual(hidden.shape[-1], config.hidden_size)
        batch, query = hidden.shape[:2]
        self.assertEqual(tuple(cos.shape), (batch, query, config.head_dim))
        self.assertEqual(tuple(sin.shape), tuple(cos.shape))
        self.assertEqual(mask.shape[0], batch)
        self.assertEqual(mask.shape[-2], query)
        if "decode" in case.tags:
            cache_name = (
                "past_key_values" if case.name.startswith("qwen") else "past_key_value"
            )
            self.assertIn(cache_name, sample.kwargs)
            key, value = bound.arguments[cache_name]
            self.assertEqual(tuple(key.shape), tuple(value.shape))
            self.assertEqual(tuple(key.shape[:2]), (batch, config.num_key_value_heads))
            self.assertEqual(key.shape[-1], config.head_dim)
            self.assertEqual(mask.shape[-1], key.shape[-2] + query)
            self.assertTrue(torch.all(mask[..., -1] == 0))
        else:
            self.assert_causal_mask(sample, -120.0)
        with torch.no_grad():
            output = invocation.module(*sample.args, **dict(sample.kwargs))
        self.assertEqual(tuple(output[0].shape), tuple(hidden.shape))

    def assert_adapter_rejects_missing_inputs(self, invocation):
        args, kwargs = invocation.inputs.args, dict(invocation.inputs.kwargs)
        with self.assertRaisesRegex(ValueError, "explicit attention_mask"):
            invocation.module(*args, **dict(kwargs, attention_mask=None))
        cos, _ = kwargs["position_embeddings"]
        for malformed in (None, (cos,), (cos, "sin")):
            with self.subTest(position_embeddings=type(malformed).__name__):
                with self.assertRaisesRegex(
                    ValueError, r"explicit \(cos, sin\) tensors"
                ):
                    invocation.module(
                        *args, **dict(kwargs, position_embeddings=malformed)
                    )

    def assert_preflight_rejects_shape_drift(self, invocation):
        kwargs = dict(invocation.inputs.kwargs)
        mask = kwargs["attention_mask"]
        bad_mask = dict(kwargs, attention_mask=mask[..., :-1])
        malformed = replace(
            invocation, inputs=ForwardInput(invocation.inputs.args, bad_mask)
        )
        with self.assertRaises(ValueError):
            malformed.validate_static_decoder_inputs()
        cos, sin = kwargs["position_embeddings"]
        bad_rope = dict(kwargs, position_embeddings=(cos[..., :-1], sin))
        malformed = replace(
            invocation, inputs=ForwardInput(invocation.inputs.args, bad_rope)
        )
        with self.assertRaises(ValueError):
            malformed.validate_static_decoder_inputs()
        bad_rope = dict(kwargs, position_embeddings=(cos[..., :-1], sin[..., :-1]))
        malformed = replace(
            invocation, inputs=ForwardInput(invocation.inputs.args, bad_rope)
        )
        with self.assertRaises(ValueError):
            malformed.validate_static_decoder_inputs()
        cache_name = next(
            (name for name in ("past_key_value", "past_key_values") if name in kwargs),
            None,
        )
        if cache_name is not None:
            bad_cache = dict(kwargs, **{cache_name: None})
            malformed = replace(
                invocation, inputs=ForwardInput(invocation.inputs.args, bad_cache)
            )
            with self.assertRaises(ValueError):
                malformed.validate_static_decoder_inputs()
            key, value = kwargs[cache_name]
            bad_cache = dict(kwargs, **{cache_name: (key[:, :0], value[:, :0])})
            malformed = replace(
                invocation, inputs=ForwardInput(invocation.inputs.args, bad_cache)
            )
            with self.assertRaises(ValueError):
                malformed.validate_static_decoder_inputs()


@unittest.skipUnless(_LLAMA.available, _LLAMA.reason or "Llama is unavailable")
class TestLlamaWrapperSmokeExportContracts(_ContractAssertions):
    """Every Llama decoder smoke export supplies the guarded host ABI."""

    def test_all_static_decoder_cases(self):
        for case_type in _LLAMA_CASES:
            with self.subTest(case=case_type.__name__):
                case = case_type()
                quantized, sample, calibration = _prepare_case(case)
                if "prefill" in case.tags:
                    for eager_sample in (*calibration, sample):
                        self.assert_causal_mask(eager_sample, -120.0)
                invocation = case.export_invocation(quantized, sample, {})
                self.assert_invocation(case, invocation)
                self.assert_adapter_rejects_missing_inputs(invocation)
                self.assert_preflight_rejects_shape_drift(invocation)

    def test_prefill_uses_actual_qconfig_fill_value(self):
        class CustomAttentionFillCase(LlamaAttentionPrefillCase):
            def ptq_config(self, cfg):
                return replace(super().ptq_config(cfg), attention_mask_fill_value=-77.0)

        class CustomLayerFillCase(LlamaDecoderLayerPrefillCase):
            def ptq_config(self, cfg):
                return replace(super().ptq_config(cfg), attention_mask_fill_value=-77.0)

        for case_type in (CustomAttentionFillCase, CustomLayerFillCase):
            with self.subTest(case=case_type.__name__):
                case = case_type()
                quantized, sample, calibration = _prepare_case(case)
                for eager_sample in (*calibration, sample):
                    self.assert_causal_mask(eager_sample, -77.0)
                invocation = case.export_invocation(quantized, sample, {})
                invocation.validate_static_decoder_inputs()
                self.assert_causal_mask(invocation.inputs, -77.0)

    def test_standalone_rope_obeys_actual_attention_convention(self):
        class NpuAttentionCase(LlamaAttentionPrefillCase):
            def ptq_config(self, cfg):
                return replace(
                    super().ptq_config(cfg), model_args={"profile": "npu_export"}
                )

        pairs = []
        for case_type in (LlamaAttentionPrefillCase, NpuAttentionCase):
            case = case_type()
            model, _ = case.build({})
            prepared = case.prepare_model(model, {})
            torch.manual_seed(79)
            sample = case.eval_input(prepared, {})
            invocation = case.export_invocation(prepared, sample, {})
            invocation.validate_static_decoder_inputs()
            pairs.append(invocation.inputs.kwargs["position_embeddings"])
        (hf_cos, hf_sin), (npu_cos, npu_sin) = pairs
        half = hf_sin.shape[-1] // 2
        torch.testing.assert_close(hf_cos, npu_cos, rtol=0, atol=0)
        torch.testing.assert_close(
            hf_sin[..., :half], -npu_sin[..., :half], rtol=0, atol=0
        )
        torch.testing.assert_close(
            hf_sin[..., half:], npu_sin[..., half:], rtol=0, atol=0
        )

    def test_runner_preserves_one_canonical_invocation_at_conversion(self):
        class RecordingCase(LlamaAttentionPrefillCase):
            def __init__(self):
                super().__init__()
                self.invocations = []

            def export_invocation(self, quantized, eval_sample, cfg):
                invocation = super().export_invocation(quantized, eval_sample, cfg)
                self.invocations.append(invocation)
                return invocation

        class SerializedCircle:
            def save(self, path):
                Path(path).write_bytes(b"mock Circle serialization boundary")

        case = RecordingCase()
        quantized, sample, _ = _prepare_case(case)
        received = []

        def convert(module, args, kwargs):
            self.assertEqual(len(case.invocations), 1)
            invocation = case.invocations[0]
            invocation.validate_static_decoder_inputs()
            self.assertIs(module, invocation.module)
            self.assertIs(args[0], invocation.inputs.args[0])
            for name in ("attention_mask", "position_embeddings"):
                self.assertIs(kwargs[name], invocation.inputs.kwargs[name])
            received.append(kwargs)
            return SerializedCircle()

        result = WrapperSmokeResult(case=case.name, passed=True)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("tico.convert", side_effect=convert),
        ):
            runner._export_circle(case, quantized, sample, {}, Path(temporary), result)
            self.assertTrue(Path(result.artifacts["circle"]).is_file())
        self.assertTrue(result.passed, result.messages)
        self.assertEqual(len(received), 1)
        self.assert_causal_mask(case.invocations[0].inputs, -120.0)


@unittest.skipUnless(_QWEN.available, _QWEN.reason or "Qwen3-VL is unavailable")
class TestQwenWrapperSmokeExportContracts(_ContractAssertions):
    """Qwen text adapters retain the family-specific mask, RoPE and cache ABI."""

    def test_all_static_decoder_cases(self):
        for case_type in _QWEN_CASES:
            with self.subTest(case=case_type.__name__):
                case = case_type()
                quantized, sample, calibration = _prepare_case(case)
                if "prefill" in case.tags:
                    for eager_sample in (*calibration, sample):
                        self.assert_causal_mask(eager_sample, -120.0)
                invocation = case.export_invocation(quantized, sample, {})
                self.assert_invocation(case, invocation)
                self.assert_adapter_rejects_missing_inputs(invocation)
                self.assert_preflight_rejects_shape_drift(invocation)

    def test_prefill_uses_actual_qconfig_fill_value(self):
        class CustomFillCase(QwenTextAttentionPrefillCase):
            def ptq_config(self, cfg):
                return replace(super().ptq_config(cfg), attention_mask_fill_value=-77.0)

        case = CustomFillCase()
        quantized, sample, calibration = _prepare_case(case)
        for eager_sample in (*calibration, sample):
            self.assert_causal_mask(eager_sample, -77.0)
        invocation = case.export_invocation(quantized, sample, {})
        invocation.validate_static_decoder_inputs()
        self.assert_causal_mask(invocation.inputs, -77.0)


if __name__ == "__main__":
    unittest.main()
