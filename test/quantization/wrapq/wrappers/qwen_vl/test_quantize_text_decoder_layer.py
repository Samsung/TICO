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

"""Smoke tests migrated from the Qwen3-VL text decoder-layer quantization example."""

import copy
import os
import unittest

import tico.quantization

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Qwen3-VL text decoder tests"


def _make_text_config():
    """Create a tiny Qwen3-VL text config for decoder-layer tests."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig

    cfg = Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        attention_dropout=0.0,
        use_cache=False,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenTextDecoderLayerExample(unittest.TestCase):
    """Exercise the old text decoder-layer PTQ flow with synthetic tensors."""

    def setUp(self):
        """Create a tiny deterministic Qwen text decoder layer."""
        torch.manual_seed(123)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        self.cfg = _make_text_config()
        self.fp_layer = Qwen3VLTextDecoderLayer(self.cfg, layer_idx=0).eval()
        self.fp_ref = copy.deepcopy(self.fp_layer).eval()

    def _inputs(self, seq_len: int = 8):
        """Create synthetic hidden states, RoPE tensors, masks, and position IDs."""
        hidden = torch.randn(1, seq_len, self.cfg.hidden_size)
        position_embeddings = (
            torch.randn(1, seq_len, self.cfg.head_dim),
            torch.randn(1, seq_len, self.cfg.head_dim),
        )
        attention_mask = torch.zeros(1, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        return hidden, position_embeddings, attention_mask, position_ids

    def test_prepare_convert_text_decoder_layer_flow_matches_example(self):
        """Quantize one text decoder layer and validate a synthetic output."""
        prepared = tico.quantization.prepare(self.fp_layer, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)

        with torch.no_grad():
            for _ in range(3):
                hidden, pos, mask, position_ids = self._inputs()
                prepared(
                    hidden_states=hidden,
                    position_embeddings=pos,
                    attention_mask=mask,
                    position_ids=position_ids,
                )

        quantized = tico.quantization.convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)

        hidden, pos, mask, position_ids = self._inputs()
        with torch.no_grad():
            quant_out = quantized(
                hidden_states=hidden,
                position_embeddings=pos,
                attention_mask=mask,
                position_ids=position_ids,
            )
            fp_out = self.fp_ref(
                hidden_states=hidden,
                position_embeddings=pos,
                attention_mask=mask,
                position_ids=position_ids,
            )

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()
