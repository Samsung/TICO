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

"""Host-input contracts for every guarded Gemma4 wrapper-smoke adapter."""

import inspect
import unittest
from dataclasses import replace
from typing import cast

import torch

from tico.quantization.recipes.debug.wrapper_smoke import get_case
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    _has_gemma4,
    Gemma4TextAttentionBaseCase,
    Gemma4TextDecoderLayerBaseCase,
)
from tico.quantization.wrapq.utils.static_export import require_static_decoder_inputs

_STRICT_CASES = (
    "gemma4_text_attention_prefill",
    "gemma4_text_attention_decode",
    "gemma4_text_attention_sliding_prefill",
    "gemma4_text_attention_sliding_decode",
    "gemma4_text_attention_k_eq_v_prefill",
    "gemma4_text_attention_shared_kv_prefill",
    "gemma4_text_attention_shared_kv_decode",
    "gemma4_text_decoder_layer_prefill",
    "gemma4_text_decoder_layer_sliding_prefill",
    "gemma4_text_decoder_layer_decode",
    "gemma4_text_decoder_layer_shared_kv",
)
_AVAILABILITY = _has_gemma4()


@unittest.skipUnless(
    _AVAILABILITY.available,
    _AVAILABILITY.reason or "Gemma4 is unavailable",
)
class TestGemma4WrapperSmokeExportContracts(unittest.TestCase):
    """Bind actual tiny adapters to explicit host mask, RoPE, and cache inputs."""

    def test_all_guarded_static_cases(self):
        """Preserve full/sliding causal semantics and ordinary/shared cache ABIs."""
        for name in _STRICT_CASES:
            with self.subTest(case=name), torch.no_grad():
                case = cast(
                    Gemma4TextAttentionBaseCase | Gemma4TextDecoderLayerBaseCase,
                    get_case(name),
                )
                model, reference = case.build({})
                prepared = case.prepare_model(model, {})
                case.after_prepare(prepared, {})
                calibration = case.calibration_inputs(prepared, {})[0]
                case.forward(prepared, calibration)
                sample = case.eval_input(prepared, {})
                reference_output = (
                    case.forward(prepared, sample)
                    if case.compare_reference_source == "prepared"
                    else case.reference_forward(reference, sample)
                )
                self.assertEqual(
                    case.output_tensor(reference_output).shape,
                    sample.kwargs["hidden_states"].shape,
                )
                quantized = case.convert_model(prepared, {})
                invocation = case.export_invocation(quantized, sample, {})
                invocation.validate_static_decoder_inputs()
                bound = inspect.signature(invocation.module.forward).bind(
                    *invocation.inputs.args, **invocation.inputs.kwargs
                )
                bound.apply_defaults()
                inputs = bound.arguments
                hidden = inputs["hidden_states"]
                mask = inputs["attention_mask"]
                cos, sin = inputs["position_embeddings"]
                require_static_decoder_inputs(mask, (cos, sin))
                batch, query, _ = hidden.shape
                self.assertEqual(batch, 1)
                self.assertEqual(mask.shape[:2], (1, 1))
                self.assertEqual(mask.shape[-2], query)
                self.assertEqual(cos.shape, (1, query, case.attention_head_dim))
                self.assertEqual(sin.shape, cos.shape)
                self.assertTrue(torch.isfinite(mask).all())
                torch.testing.assert_close(
                    mask, sample.kwargs["attention_mask"], atol=0, rtol=0
                )

                decode = case.export_mode == "decode"
                key_length = case.max_seq if decode else query
                self.assertEqual(mask.shape[-1], key_length)
                if "shared_kv" in name:
                    key, value = inputs["shared_key_value"]
                    self.assertEqual(key.shape[2], key_length)
                    self.assertIsNone(inputs.get("past_key_value"))
                    missing_shared = replace(
                        invocation,
                        inputs=replace(
                            invocation.inputs,
                            kwargs={
                                key: value
                                for key, value in invocation.inputs.kwargs.items()
                                if key != "shared_key_value"
                            },
                        ),
                    )
                    with self.assertRaisesRegex(ValueError, "shared"):
                        missing_shared.validate_static_decoder_inputs()
                elif decode:
                    key, value = inputs["past_key_value"]
                    self.assertEqual(key.shape[2], key_length - 1)
                else:
                    key = value = None
                if key is not None:
                    self.assertEqual(key.shape, value.shape)
                    self.assertEqual(
                        key.shape[:2], (1, case.text_cfg.num_key_value_heads)
                    )
                    self.assertEqual(key.shape[-1], case.attention_head_dim)

                query_positions = torch.arange(query) + (key_length - query)
                key_positions = torch.arange(key_length)
                visible = key_positions[None, :] <= query_positions[:, None]
                if "sliding" in name:
                    window = getattr(case, "sliding_window")
                    visible &= (
                        key_positions[None, :] > query_positions[:, None] - window
                    )
                expected = torch.where(
                    visible,
                    0.0,
                    float(case.ptq_config({}).attention_mask_fill_value),
                )
                torch.testing.assert_close(mask[0, 0], expected, atol=0, rtol=0)

                # This is the real static adapter, with frozen tiny observers.
                # Its guarded path must consume these exact host inputs.
                output = invocation.module(
                    *invocation.inputs.args, **invocation.inputs.kwargs
                )
                self.assertEqual(case.output_tensor(output).shape, hidden.shape)

    def test_prefill_adapter_keeps_negative_input_guards(self):
        """Missing or malformed host inputs fail directly before attention runs."""
        case = get_case("gemma4_text_attention_prefill")
        model, _ = case.build({})
        prepared = case.prepare_model(model, {})
        sample = case.eval_input(prepared, {})
        module = case.export_module(prepared, {})
        hidden = sample.kwargs["hidden_states"]
        mask = sample.kwargs["attention_mask"]
        rope = sample.kwargs["position_embeddings"]
        with self.assertRaisesRegex(ValueError, "explicit attention_mask"):
            module(hidden, None, rope)
        for bad_rope in (None, (rope[0],), (rope[0], None)):
            with (
                self.subTest(rope=bad_rope),
                self.assertRaisesRegex(ValueError, "explicit .*cos, sin"),
            ):
                module(hidden, mask, bad_rope)


if __name__ == "__main__":
    unittest.main()
