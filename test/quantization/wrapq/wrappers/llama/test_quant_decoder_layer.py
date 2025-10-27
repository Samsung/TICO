# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import importlib.util
import unittest

import torch

from tico.experimental.quantization.wrapq.mode import Mode
from tico.experimental.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)

trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping LlamaDecoderLayer tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantLlamaDecoderLayer(unittest.TestCase):
    fp_layer: torch.nn.Module

    @classmethod
    def setUpClass(cls):
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        cls.cfg = LlamaConfig(  # type: ignore[attr-defined]
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
        )
        cls.fp_layer = LlamaDecoderLayer(cls.cfg, layer_idx=0)  # type: ignore[attr-defined]

    # dummy RoPE tables with correct last dim
    def _rand_rope(self, B, S):
        h = self.cfg.head_dim  # type: ignore[attr-defined]
        emb = torch.randn(B, S, h)
        return emb.cos(), emb.sin()

    def test_mode_transitions(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        self.assertIs(qlayer._mode, Mode.NO_QUANT)

        qlayer.enable_calibration()
        self.assertIs(qlayer._mode, Mode.CALIB)

        hidden = torch.randn(2, 6, 16)
        pos = self._rand_rope(2, 6)
        attn_mask = torch.ones(2, 1, 6, 6, dtype=torch.bool)
        _ = qlayer(hidden, attention_mask=attn_mask, position_embeddings=pos)

        qlayer.freeze_qparams()
        self.assertIs(qlayer._mode, Mode.QUANT)

    def test_forward_diff(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        qlayer.enable_calibration()
        for _ in range(4):
            hidden = torch.randn(2, 6, 16)
            pos = self._rand_rope(2, 6)
            attn_mask = torch.ones(2, 1, 6, 6, dtype=torch.bool)
            _ = qlayer(hidden, attention_mask=attn_mask, position_embeddings=pos)
        qlayer.freeze_qparams()

        hidden = torch.randn(2, 6, 16)
        pos = self._rand_rope(2, 6)
        attn_mask = torch.ones(2, 1, 6, 6, dtype=torch.bool)

        with torch.no_grad():
            q_out = qlayer(hidden, attention_mask=attn_mask, position_embeddings=pos)
            q_out = q_out[0] if isinstance(q_out, tuple) else q_out
            fp_out = self.fp_layer(
                hidden, attention_mask=attn_mask, position_embeddings=pos
            )
            fp_out = fp_out[0] if isinstance(fp_out, tuple) else fp_out

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.5)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_layernorm_preserved(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        self.assertIsInstance(
            qlayer.input_layernorm, type(self.fp_layer.input_layernorm)
        )
        self.assertIsInstance(
            qlayer.post_attention_layernorm,
            type(self.fp_layer.post_attention_layernorm),
        )
