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

"""Tests for mode-specific Gemma4 text attention export adapters."""

import unittest

import torch

from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4TextAttentionDecodeExportAdapter,
    Gemma4TextAttentionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)


_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed Transformers package provides Gemma4."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextAttention,
        )
    except Exception:
        return False
    return True


def _make_text_config(**overrides):
    """Create a tiny dense Gemma4 text config for export-adapter tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    kwargs = dict(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
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
        layer_types=["full_attention"],
        hidden_size_per_layer_input=0,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        enable_moe_block=False,
    )
    kwargs.update(overrides)
    config = Gemma4TextConfig(**kwargs)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    else:
        setattr(config, "_attn_implementation", "eager")
    return config


def _rope(batch_size: int, seq_len: int, head_dim: int):
    """Create synthetic Gemma4 RoPE tables."""
    embedding = torch.randn(batch_size, seq_len, head_dim)
    return embedding.cos(), embedding.sin()


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4TextAttentionExportAdapters(unittest.TestCase):
    """Validate the prefill and decode adapter contracts."""

    def setUp(self):
        """Use deterministic synthetic inputs."""
        torch.manual_seed(2026)

    @staticmethod
    def _make_attention(config, layer_idx: int = 0):
        """Create one quantized Gemma4 text attention wrapper."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        return QuantGemma4TextAttention(
            Gemma4TextAttention(config, layer_idx=layer_idx).eval()
        ).eval()

    def test_as_export_module_selects_mode_specific_adapters(self):
        """Select the adapter class from the requested execution mode."""
        attention = self._make_attention(_make_text_config())

        self.assertIsInstance(
            attention.as_export_module("prefill"),
            Gemma4TextAttentionPrefillExportAdapter,
        )
        self.assertIsInstance(
            attention.as_export_module("decode"),
            Gemma4TextAttentionDecodeExportAdapter,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported Gemma4 export mode"):
            attention.as_export_module("invalid")

    def test_prefill_adapter_returns_full_prompt_kv_delta(self):
        """Return one K/V delta entry for every prefill token."""
        config = _make_text_config()
        attention = self._make_attention(config)
        adapter = attention.as_export_module("prefill")
        seq_len = 5
        hidden = torch.randn(1, seq_len, config.hidden_size)
        mask = torch.zeros(1, 1, seq_len, seq_len)

        output, key, value = adapter(hidden, mask, _rope(1, seq_len, config.head_dim))

        self.assertEqual(output.shape, (1, seq_len, config.hidden_size))
        self.assertEqual(
            key.shape,
            (1, config.num_key_value_heads, seq_len, config.head_dim),
        )
        self.assertEqual(value.shape, key.shape)

    def test_decode_adapter_returns_single_token_kv_delta(self):
        """Return only the current-token K/V delta during decode."""
        config = _make_text_config()
        attention = self._make_attention(config)
        adapter = attention.as_export_module("decode")
        past_len = 6
        hidden = torch.randn(1, 1, config.hidden_size)
        mask = torch.zeros(1, 1, 1, past_len + 1)
        past_key = torch.randn(
            1,
            config.num_key_value_heads,
            past_len,
            config.head_dim,
        )

        output, key, value = adapter(
            hidden,
            mask,
            _rope(1, 1, config.head_dim),
            past_key_value=(past_key, torch.randn_like(past_key)),
        )

        self.assertEqual(output.shape, (1, 1, config.hidden_size))
        self.assertEqual(
            key.shape,
            (1, config.num_key_value_heads, 1, config.head_dim),
        )
        self.assertEqual(value.shape, key.shape)

    def test_shared_kv_decode_returns_hidden_states_only(self):
        """Do not expose a K/V delta from a shared-KV consumer layer."""
        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            num_kv_shared_layers=1,
        )
        attention = self._make_attention(config, layer_idx=1)
        adapter = attention.as_export_module("decode")
        kv_len = 7
        hidden = torch.randn(1, 1, config.hidden_size)
        shared_key = torch.randn(
            1,
            config.num_key_value_heads,
            kv_len,
            config.head_dim,
        )

        output = adapter(
            hidden,
            torch.zeros(1, 1, 1, kv_len),
            _rope(1, 1, config.head_dim),
            shared_key_value=(shared_key, torch.randn_like(shared_key)),
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1, config.hidden_size))


if __name__ == "__main__":
    unittest.main()
