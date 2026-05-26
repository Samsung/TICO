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

"""Smoke tests migrated from the Llama attention decode quantization example."""

import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


skip_msg = "required transformers not installed — skipping Llama attention decode example tests"


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestLlamaAttentionDecodeExample(unittest.TestCase):
    """Exercise the prepare-calibrate-convert flow for one decode attention step."""

    def setUp(self):
        """Create a tiny deterministic LlamaAttention module."""
        torch.manual_seed(123)
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        self.max_seq = 16
        self.cfg = LlamaConfig(
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
            max_position_embeddings=self.max_seq,
        )
        self.fp_attn = LlamaAttention(self.cfg, layer_idx=0).eval()

    def _make_decode_batch(self, qattn=None):
        """Create static decode inputs matching the old example contract."""
        batch_size = 1
        hidden = torch.randn(batch_size, 1, self.cfg.hidden_size)
        cos = torch.randn(batch_size, 1, self.cfg.head_dim)
        sin = torch.randn(batch_size, 1, self.cfg.head_dim)
        if qattn is not None and qattn.wrapped.attn_options.rope == "pre_negated_sin":
            sin = sin.clone()
            sin[..., : self.cfg.head_dim // 2] = -sin[..., : self.cfg.head_dim // 2]
        mask = torch.zeros(batch_size, 1, self.max_seq)
        past = (
            torch.randn(
                batch_size,
                self.cfg.num_key_value_heads,
                self.max_seq - 1,
                self.cfg.head_dim,
            ),
            torch.randn(
                batch_size,
                self.cfg.num_key_value_heads,
                self.max_seq - 1,
                self.cfg.head_dim,
            ),
        )
        return hidden, (cos, sin), mask, past

    def test_prepare_convert_decode_flow_matches_example(self):
        """Quantize one Llama attention block and run a decode-step sanity check."""
        from tico.quantization.wrapq.wrappers.llama.quant_attention import (
            QuantLlamaAttention,
        )

        prepared = prepare(self.fp_attn, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantLlamaAttention)
        self.assertIs(prepared._mode, Mode.CALIB)

        with torch.no_grad():
            for _ in range(4):
                hidden, pos, mask, past = self._make_decode_batch(prepared)
                float_like_out = prepared(
                    hidden,
                    pos,
                    attention_mask=mask,
                    past_key_value=past,
                    use_cache=True,
                )

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            quant_out = quantized(
                hidden,
                pos,
                attention_mask=mask,
                past_key_value=past,
                use_cache=True,
            )

        self.assertEqual(quant_out[0].shape, float_like_out[0].shape)
        self.assertTrue(torch.isfinite(quant_out[0]).all())
        self.assertGreater((quant_out[0] - float_like_out[0]).abs().mean().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
