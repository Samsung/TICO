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

"""Smoke tests migrated from the Llama decoder-layer prefill quantization example."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Llama decoder example tests"


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestLlamaDecoderLayerPrefillExample(unittest.TestCase):
    """Exercise the example flow for one full Llama decoder layer."""

    hidden_size: int
    head_dim: int
    max_seq: int

    def setUp(self):
        """Create a tiny deterministic LlamaDecoderLayer module for each test."""
        torch.manual_seed(0)

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
        self.fp_ref = copy.deepcopy(self.fp_layer).eval()
        self.hidden_size = self.cfg.hidden_size
        self.head_dim = self.cfg.head_dim

    def _rand_rope(self, batch_size: int, seq_len: int):
        """Return Hugging Face-style RoPE tensors for the tiny decoder layer."""
        emb = torch.randn(batch_size, seq_len, self.head_dim)
        return emb.cos(), emb.sin()

    def _calibrate(self, prepared: PTQWrapper) -> None:
        """Run a short synthetic prefill calibration sweep."""
        with torch.no_grad():
            for _ in range(3):
                hidden = torch.randn(2, self.max_seq, self.hidden_size)
                rope = self._rand_rope(2, self.max_seq)
                mask = torch.ones(2, self.max_seq, self.max_seq, dtype=torch.bool)
                _ = prepared(
                    hidden,
                    attention_mask=mask,
                    position_embeddings=rope,
                )

    def test_prepare_convert_prefill_flow_matches_example(self):
        """Quantize one decoder layer and compare the quantized prefill output."""
        from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
            QuantLlamaDecoderLayer,
        )

        qcfg = PTQConfig(model_args={"profile": "reference_eval"})
        prepared = prepare(self.fp_layer, qcfg)

        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantLlamaDecoderLayer)
        self.assertIs(prepared._mode, Mode.CALIB)

        self._calibrate(prepared)
        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        hidden = torch.randn(2, self.max_seq, self.hidden_size)
        rope = self._rand_rope(2, self.max_seq)
        mask = torch.ones(2, self.max_seq, self.max_seq, dtype=torch.bool)

        with torch.no_grad():
            quant_out = quantized(
                hidden,
                attention_mask=mask,
                position_embeddings=rope,
            )
            quant_out = quant_out[0] if isinstance(quant_out, tuple) else quant_out
            fp_out = self.fp_ref(
                hidden,
                attention_mask=mask.unsqueeze(1),
                position_embeddings=rope,
            )
            fp_out = fp_out[0] if isinstance(fp_out, tuple) else fp_out

        diff = (quant_out - fp_out).abs().mean().item()
        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1.2)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()
