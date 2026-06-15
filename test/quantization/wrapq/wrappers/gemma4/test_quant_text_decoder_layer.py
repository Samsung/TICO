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

"""Unit tests for the Gemma4 text decoder-layer PTQ wrapper."""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode


_SKIP_MSG = "required transformers Gemma4 modules are not installed"

_GEMMA4_FULL_ROPE_PARAMETERS = {
    "rope_type": "proportional",
    "partial_rotary_factor": 0.25,
    "rope_theta": 1_000_000.0,
}
_GEMMA4_SLIDING_ROPE_PARAMETERS = {
    "rope_type": "default",
    "rope_theta": 10_000.0,
}


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextDecoderLayer,
        )
    except Exception:
        return False
    return True


def _rope_parameters_for_layer_types(layer_types: list[str]) -> dict[str, dict]:
    """Return Gemma4 RoPE parameters matching the requested layer types."""
    params = {}
    if "sliding_attention" in layer_types:
        params["sliding_attention"] = dict(_GEMMA4_SLIDING_ROPE_PARAMETERS)
    if "full_attention" in layer_types:
        params["full_attention"] = dict(_GEMMA4_FULL_ROPE_PARAMETERS)
    return params


def _make_text_config(**overrides):
    """Create a tiny dense Gemma4 text config for synthetic decoder tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    layer_types = list(overrides.pop("layer_types", ["full_attention"]))
    kwargs = dict(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        sliding_window=8,
        layer_types=layer_types,
        rope_parameters=_rope_parameters_for_layer_types(layer_types),
        hidden_size_per_layer_input=0,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        enable_moe_block=False,
    )
    kwargs.update(overrides)
    cfg = Gemma4TextConfig(**kwargs)
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _rope(batch_size: int, seq_len: int, head_dim: int):
    """Create synthetic RoPE tables shaped like Gemma4 text RoPE output."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


def _zero_mask(batch_size: int, q_len: int, k_len: int) -> torch.Tensor:
    """Return an additive attention mask that keeps every key."""
    return torch.zeros(batch_size, 1, q_len, k_len)


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4TextDecoderLayer(unittest.TestCase):
    """Validate dense Gemma4 text decoder-layer wrapper behavior."""

    def setUp(self):
        """Create deterministic test inputs."""
        torch.manual_seed(2026)

    @staticmethod
    def _make_layer(cfg=None, layer_idx: int = 0):
        """Create a floating-point Gemma4 text decoder layer."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        cfg = cfg if cfg is not None else _make_text_config()
        return Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval()

    def test_00_prepare_wraps_decoder_layer_when_registered(self):
        """Check that registry-based prepare wraps Gemma4TextDecoderLayer."""
        from tico.quantization import prepare
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )
        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        fp_layer = self._make_layer(_make_text_config())
        prepared = prepare(fp_layer, PTQConfig())

        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4TextDecoderLayer)

    def test_no_quant_forward_matches_hf_dense_decoder_layer(self):
        """Check that the wrapper matches Hugging Face eager decoder output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        cfg = _make_text_config(layer_types=["sliding_attention", "full_attention"])
        fp_layer = self._make_layer(cfg, layer_idx=1)
        qlayer = QuantGemma4TextDecoderLayer(fp_layer).eval()

        batch_size, seq_len = 2, 5
        hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
        position_embeddings = _rope(batch_size, seq_len, cfg.head_dim)
        mask = _zero_mask(batch_size, seq_len, seq_len)

        with torch.no_grad():
            quant_out = qlayer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                shared_kv_states={},
            )
            fp_out = fp_layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                shared_kv_states={},
            )

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_mode_transitions_and_child_override(self):
        """Check lifecycle transitions and child attention override propagation."""
        from tico.quantization.wrapq.dtypes import DType
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )
        from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

        cfg = PTQConfig(
            overrides={
                "self_attn": {
                    "q_proj": {
                        "act_in": {"dtype": DType.uint(4)},
                        "act_out": {"dtype": DType.uint(4)},
                    }
                }
            }
        )
        qlayer = QuantGemma4TextDecoderLayer(self._make_layer(), qcfg=cfg)

        self.assertIs(qlayer._mode, Mode.NO_QUANT)
        qlayer.enable_calibration()
        self.assertIs(qlayer._mode, Mode.CALIB)

        hidden = torch.randn(1, 4, qlayer.config.hidden_size)
        qlayer(
            hidden,
            position_embeddings=_rope(1, 4, qlayer.config.head_dim),
            attention_mask=_zero_mask(1, 4, 4),
            shared_kv_states={},
        )
        qlayer.freeze_qparams()

        self.assertIs(qlayer._mode, Mode.QUANT)
        self.assertIsInstance(qlayer.self_attn.wrapped.q_proj.wrapped, QuantLinear)
        self.assertEqual(
            qlayer.self_attn.wrapped.q_proj.wrapped.obs_act_in.dtype, DType.uint(4)
        )
        self.assertEqual(
            qlayer.self_attn.wrapped.q_proj.wrapped.obs_act_out.dtype, DType.uint(4)
        )

    def test_prefill_export_adapter_returns_flat_kv_delta(self):
        """Validate the static prefill export adapter output contract."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        cfg = _make_text_config()
        qlayer = QuantGemma4TextDecoderLayer(self._make_layer(cfg)).eval()
        adapter = qlayer.as_export_module("prefill", return_kv=True).eval()

        batch_size, seq_len = 1, 4
        hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
        mask = _zero_mask(batch_size, seq_len, seq_len)
        position_embeddings = _rope(batch_size, seq_len, cfg.head_dim)

        with torch.no_grad():
            output = adapter(hidden, mask, position_embeddings)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)
        hidden_out, key_delta, value_delta = output
        self.assertEqual(hidden_out.shape, (batch_size, seq_len, cfg.hidden_size))
        self.assertEqual(
            key_delta.shape,
            (batch_size, cfg.num_key_value_heads, seq_len, cfg.head_dim),
        )
        self.assertEqual(value_delta.shape, key_delta.shape)

    def test_decode_export_adapter_returns_single_token_kv_delta(self):
        """Validate the static decode export adapter output contract."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        cfg = _make_text_config()
        qlayer = QuantGemma4TextDecoderLayer(self._make_layer(cfg)).eval()
        adapter = qlayer.as_export_module("decode", return_kv=True).eval()

        batch_size, past_len = 1, 4
        hidden = torch.randn(batch_size, 1, cfg.hidden_size)
        past = (
            torch.randn(batch_size, cfg.num_key_value_heads, past_len, cfg.head_dim),
            torch.randn(batch_size, cfg.num_key_value_heads, past_len, cfg.head_dim),
        )
        mask = _zero_mask(batch_size, 1, past_len + 1)
        position_embeddings = _rope(batch_size, 1, cfg.head_dim)

        with torch.no_grad():
            output = adapter(
                hidden,
                mask,
                position_embeddings,
                past_key_value=past,
            )

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)
        hidden_out, key_delta, value_delta = output
        self.assertEqual(hidden_out.shape, (batch_size, 1, cfg.hidden_size))
        self.assertEqual(
            key_delta.shape,
            (batch_size, cfg.num_key_value_heads, 1, cfg.head_dim),
        )
        self.assertEqual(value_delta.shape, key_delta.shape)

    def test_shared_kv_export_adapter_consumes_explicit_tuple(self):
        """Check that a shared-KV decoder layer does not return a new cache delta."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        cfg = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            num_kv_shared_layers=1,
        )
        qlayer = QuantGemma4TextDecoderLayer(self._make_layer(cfg, layer_idx=1)).eval()
        adapter = qlayer.as_export_module("prefill", return_kv=True).eval()

        batch_size, seq_len = 1, 4
        hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
        key_value = (
            torch.randn(batch_size, cfg.num_key_value_heads, seq_len, cfg.head_dim),
            torch.randn(batch_size, cfg.num_key_value_heads, seq_len, cfg.head_dim),
        )

        with torch.no_grad():
            output = adapter(
                hidden,
                _zero_mask(batch_size, seq_len, seq_len),
                _rope(batch_size, seq_len, cfg.head_dim),
                shared_key_value=key_value,
            )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, seq_len, cfg.hidden_size))

    def test_ple_path_matches_hf_and_requires_input(self):
        """Validate the optional per-layer embedding path."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        cfg = _make_text_config(hidden_size_per_layer_input=8)
        fp_layer = self._make_layer(cfg)
        qlayer = QuantGemma4TextDecoderLayer(fp_layer).eval()

        hidden = torch.randn(1, 3, cfg.hidden_size)
        position_embeddings = _rope(1, 3, cfg.head_dim)
        mask = _zero_mask(1, 3, 3)

        with self.assertRaisesRegex(ValueError, "per_layer_input"):
            qlayer(hidden, position_embeddings=position_embeddings, attention_mask=mask)

        per_layer_input = torch.randn(1, 3, cfg.hidden_size_per_layer_input)
        with torch.no_grad():
            quant_out = qlayer(
                hidden,
                per_layer_input=per_layer_input,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                shared_kv_states={},
            )
            fp_out = fp_layer(
                hidden,
                per_layer_input=per_layer_input,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                shared_kv_states={},
            )

        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_moe_decoder_layers_are_rejected_for_e2b_scope(self):
        """Check that the E2B decoder wrapper rejects MoE layers explicitly."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        cfg = _make_text_config(
            enable_moe_block=True,
            num_experts=2,
            top_k_experts=1,
            moe_intermediate_size=16,
        )
        fp_layer = self._make_layer(cfg)

        with self.assertRaisesRegex(NotImplementedError, "dense decoder layers only"):
            QuantGemma4TextDecoderLayer(fp_layer)


if __name__ == "__main__":
    unittest.main()
