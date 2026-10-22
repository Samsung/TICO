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

"""Exercise eager context lengths independently of calibration and export."""

import math
import unittest
from unittest.mock import patch

import torch
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.static_export import prepared_static_export
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attention import (
    QuantQwen3VLTextAttention,
)
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_model import (
    QuantQwen3VLTextModel,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


@unittest.skipUnless(
    has_transformers_for("qwen3-vl"), "Qwen3-VL Transformers support is required"
)
class TestQwen3VLEagerContext(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)

    @staticmethod
    def _make_model(max_position_embeddings=64):
        """Create a small random model without downloading a checkpoint."""
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        config = Qwen3VLTextConfig.from_dict(
            {
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "max_position_embeddings": max_position_embeddings,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
                "attention_dropout": 0.0,
                "use_cache": True,
            }
        )
        config._attn_implementation = "eager"
        return Qwen3VLTextModel(config).eval()

    def test_native_context_does_not_allocate_templates(self):
        """Preparing a long-context model must not allocate native-sized masks."""
        native_context = 262_144
        model = self._make_model(native_context)
        original_full = torch.full

        def bounded_full(size, *args, **kwargs):
            self.assertLess(
                math.prod(size),
                native_context,
                "Wrapper construction allocated a native-context tensor",
            )
            return original_full(size, *args, **kwargs)

        # Fail before allocating the enormous tensor in a regressed constructor.
        with patch("torch.full", side_effect=bounded_full):
            wrapped = QuantQwen3VLTextModel(model).eval()

        self.assertEqual(model.config.max_position_embeddings, native_context)
        for name, tensor in wrapped.named_buffers():
            with self.subTest(buffer=name):
                self.assertLess(tensor.numel(), native_context)
        self.assertFalse(hasattr(wrapped, "causal_mask_template"))
        self.assertFalse(hasattr(wrapped, "position_ids_template"))
        self.assertFalse(hasattr(wrapped, "cos_template"))
        self.assertFalse(hasattr(wrapped, "sin_template"))

    def test_runtime_multimodal_positions_and_padding_match_fp(self):
        """Each eager request uses its supplied mRoPE positions and padding."""
        model = self._make_model()
        wrapped = QuantQwen3VLTextModel(model).eval()
        ids = torch.randint(1, 32, (2, 12))
        keep = torch.ones_like(ids)
        keep[0, :3] = 0
        keep[1, :2] = 0
        text_positions = keep.cumsum(-1) - 1
        text_positions.masked_fill_(keep == 0, 1)

        for offset in (0, 11):
            positions = text_positions.unsqueeze(0).repeat(4, 1, 1)
            positions[1] += offset
            positions[2, :, 4:8] += 3
            positions[3, :, 4:8] += 6
            with self.subTest(offset=offset), torch.no_grad():
                actual = wrapped(
                    input_ids=ids,
                    attention_mask=keep,
                    position_ids=positions,
                    use_cache=False,
                ).last_hidden_state
                expected = model(
                    input_ids=ids,
                    attention_mask=keep,
                    position_ids=positions,
                    use_cache=False,
                ).last_hidden_state
            # Fully masked padding queries have no reference output contract.
            torch.testing.assert_close(
                actual[keep.bool()], expected[keep.bool()], atol=1e-5, rtol=1e-4
            )

    def test_short_calibration_keeps_qparams_for_long_prefill_and_decode(self):
        """Longer eager calls reuse frozen qparams and agree with full forward."""
        model = self._make_model(max_position_embeddings=4)
        wrapped = QuantQwen3VLTextModel(model, qcfg=make_affine_ptq_config()).eval()
        wrapped.enable_calibration()
        with torch.no_grad():
            for _ in range(3):
                calibration_ids = torch.randint(1, 32, (1, 4))
                prefill = wrapped(input_ids=calibration_ids[:, :3], use_cache=True)
                wrapped(
                    input_ids=calibration_ids[:, 3:],
                    attention_mask=torch.ones_like(calibration_ids),
                    past_key_values=prefill.past_key_values,
                    use_cache=True,
                )
        wrapped.freeze_qparams()
        observers = dict(wrapped.named_observers())
        frozen = {
            name: {key: value.clone() for key, value in obs.state_dict().items()}
            for name, obs in observers.items()
        }

        ids = torch.randint(1, 32, (1, 13))
        with torch.no_grad():
            full = wrapped(input_ids=ids, use_cache=False).last_hidden_state
            prefix = wrapped(input_ids=ids[:, :9], use_cache=True)
            suffix = wrapped(
                input_ids=ids[:, 9:],
                attention_mask=torch.ones_like(ids),
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
        torch.testing.assert_close(
            suffix.last_hidden_state, full[:, 9:], atol=1e-5, rtol=1e-4
        )
        self.assertEqual(suffix.past_key_values.get_seq_length(), 13)
        self.assertIs(wrapped._mode, Mode.QUANT)
        self.assertEqual(model.config.max_position_embeddings, 4)
        for name, obs in observers.items():
            self.assertFalse(obs.enabled)
            for key, value in obs.state_dict().items():
                with self.subTest(observer=name, buffer=key):
                    torch.testing.assert_close(
                        value, frozen[name][key], atol=0, rtol=0, equal_nan=True
                    )

    def test_attention_causal_offset_uses_actual_cache_span(self):
        """Chunked decode masks only future keys after the real cached prefix."""
        model = self._make_model(max_position_embeddings=4)
        attention = QuantQwen3VLTextAttention(model.layers[0].self_attn).eval()
        actual = attention._build_attention_mask(
            attention_mask=None,
            q_len=3,
            k_len=11,
            device=torch.device("cpu"),
        )
        expected = torch.tensor(
            [[[[0.0] * 9 + [-120.0, -120.0], [0.0] * 10 + [-120.0], [0.0] * 11]]]
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_default_cache_policy_matches_model_configuration(self):
        """Direct forward calls honor the same cache policy as the FP model."""
        model = self._make_model()
        wrapped = QuantQwen3VLTextModel(model).eval()
        ids = torch.randint(1, 32, (1, 7))
        for use_cache in (False, True):
            model.config.use_cache = use_cache
            with self.subTest(use_cache=use_cache), torch.no_grad():
                actual = wrapped(input_ids=ids)
                expected = model(input_ids=ids)
                self.assertEqual(actual.past_key_values is not None, use_cache)
                self.assertEqual(expected.past_key_values is not None, use_cache)
                torch.testing.assert_close(
                    actual.last_hidden_state,
                    expected.last_hidden_state,
                    atol=1e-5,
                    rtol=1e-4,
                )
                override = wrapped(input_ids=ids, use_cache=not use_cache)
                self.assertEqual(override.past_key_values is not None, not use_cache)

    def test_full_text_export_builds_only_the_captured_visual_positions(self):
        """Full-text capture preserves fixed-grid mRoPE without native templates."""
        model = self._make_model()
        qcfg = make_affine_ptq_config(
            model_args={
                "vision": {
                    "grid_thw": (1, 4, 4),
                    "visual_start_idx": 2,
                    "spatial_merge_size": 2,
                }
            }
        )
        wrapped = QuantQwen3VLTextModel(model, qcfg=qcfg).eval()
        ids = torch.randint(1, 32, (1, 12))
        positions = torch.tensor(
            [
                [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9],
            ]
        ).unsqueeze(1)
        kwargs = {"input_ids": ids, "use_cache": False, "return_dict": False}
        with torch.no_grad():
            expected = wrapped(position_ids=positions, **kwargs)[0]
            with prepared_static_export(wrapped, ids.shape[1]):
                exported = torch.export.export(wrapped, (), kwargs=kwargs, strict=False)
            actual = exported.module()(**kwargs)[0]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_explicit_static_stage_shapes_still_export(self):
        """Static adapters preserve fixed cache shapes and delta-KV outputs."""
        model = self._make_model(max_position_embeddings=4)
        attention = QuantQwen3VLTextAttention(model.layers[0].self_attn).eval()
        prefill = attention.as_export_module("prefill", return_kv=True).eval()
        hidden = torch.randn(1, 6, 16)
        phases = torch.randn(1, 6, 8)
        rope = (phases.cos(), phases.sin())
        mask = torch.full((1, 1, 6, 6), -120.0).triu_(1)
        kwargs = {"position_embeddings": rope, "attention_mask": mask}
        with torch.no_grad():
            expected_prefill = prefill(hidden, **kwargs)
            exported_prefill = torch.export.export(
                prefill, (hidden,), kwargs=kwargs, strict=False
            )
            actual_prefill = exported_prefill.module()(hidden, **kwargs)
        for actual, expected in zip(actual_prefill, expected_prefill):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(actual_prefill[1].shape, (1, 1, 6, 8))

        decode = attention.as_export_module("decode", return_kv=True).eval()
        hidden = torch.randn(1, 1, 16)
        phases = torch.randn(1, 1, 8)
        past_k = torch.randn(1, 1, 7, 8)
        past_v = torch.randn_like(past_k)
        mask = torch.zeros(1, 1, 1, 8)
        mask[..., 6] = -120.0
        kwargs = {
            "position_embeddings": (phases.cos(), phases.sin()),
            "attention_mask": mask,
            "past_key_values": (past_k, past_v),
        }
        with torch.no_grad():
            expected_decode = decode(hidden, **kwargs)
            exported_decode = torch.export.export(
                decode, (hidden,), kwargs=kwargs, strict=False
            )
            actual_decode = exported_decode.module()(hidden, **kwargs)
        for actual, expected in zip(actual_decode, expected_decode):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(actual_decode[1].shape, (1, 1, 1, 8))
