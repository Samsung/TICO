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

"""Smoke tests for Gemma4 text attention prepare-calibrate-convert flow."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4."""
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


def _make_text_config():
    """Create a tiny Gemma4 text config for synthetic smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
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
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _rope(batch_size: int, seq_len: int, head_dim: int):
    """Create synthetic Gemma4 text RoPE tables."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4TextAttentionSmoke(unittest.TestCase):
    """Exercise the prepare-calibrate-convert flow for Gemma4 text attention."""

    def setUp(self):
        """Create a deterministic tiny Gemma4 text attention module."""
        torch.manual_seed(2026)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        self.cfg = _make_text_config()
        self.fp_attn = Gemma4TextAttention(self.cfg, layer_idx=0).eval()
        self.fp_ref = copy.deepcopy(self.fp_attn).eval()

    def test_prepare_convert_text_attention_flow(self):
        """Quantize Gemma4 text attention and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
            QuantGemma4TextAttention,
        )

        prepared = prepare(self.fp_attn, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4TextAttention)

        batch_size, seq_len = 1, 8
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        with torch.no_grad():
            for _ in range(3):
                hidden = torch.randn(batch_size, seq_len, self.cfg.hidden_size)
                prepared(
                    hidden,
                    _rope(batch_size, seq_len, self.cfg.head_dim),
                    attention_mask=mask,
                    shared_kv_states={},
                )

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        hidden = torch.randn(batch_size, seq_len, self.cfg.hidden_size)
        position_embeddings = _rope(batch_size, seq_len, self.cfg.head_dim)
        with torch.no_grad():
            quant_out = quantized(
                hidden,
                position_embeddings,
                attention_mask=mask,
                shared_kv_states={},
            )[0]
            fp_out = self.fp_ref(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                shared_kv_states={},
            )[0]

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()
