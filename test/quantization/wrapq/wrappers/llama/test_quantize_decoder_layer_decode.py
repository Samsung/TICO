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

"""Smoke tests migrated from the Llama decoder-layer decode quantization example."""

import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


skip_msg = (
    "required transformers not installed — skipping Llama decoder decode example tests"
)


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestLlamaDecoderLayerDecodeExample(unittest.TestCase):
    """Exercise one static decode step of a quantized Llama decoder layer."""

    def setUp(self):
        """Create a tiny deterministic LlamaDecoderLayer module."""
        torch.manual_seed(123)
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        self.max_seq = 16
        self.cfg = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
            max_position_embeddings=self.max_seq,
        )
        self.fp_layer = LlamaDecoderLayer(self.cfg, layer_idx=0).eval()

    def _make_decode_batch(self):
        """Create static decode inputs matching the old decoder-layer example."""
        batch_size = 1
        hidden = torch.randn(batch_size, 1, self.cfg.hidden_size)
        pos = (
            torch.randn(batch_size, 1, self.cfg.head_dim),
            torch.randn(batch_size, 1, self.cfg.head_dim),
        )
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
        return hidden, pos, mask, past

    def test_prepare_convert_decode_flow_matches_example(self):
        """Quantize a Llama decoder layer and validate hidden/cache shapes."""
        from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
            QuantLlamaDecoderLayer,
        )

        prepared = prepare(self.fp_layer, PTQConfig())
        prepared.wrapped.return_type = "tuple"
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantLlamaDecoderLayer)

        with torch.no_grad():
            for _ in range(4):
                hidden, pos, mask, past = self._make_decode_batch()
                float_like_hidden, float_like_present = prepared(
                    hidden_states=hidden,
                    attention_mask=mask,
                    past_key_value=past,
                    position_embeddings=pos,
                    use_cache=True,
                )

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            quant_hidden, quant_present = quantized(
                hidden_states=hidden,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        self.assertEqual(quant_hidden.shape, float_like_hidden.shape)
        self.assertEqual(quant_present[0].shape, float_like_present[0].shape)
        self.assertEqual(quant_present[1].shape, float_like_present[1].shape)
        self.assertTrue(torch.isfinite(quant_hidden).all())


if __name__ == "__main__":
    unittest.main()
