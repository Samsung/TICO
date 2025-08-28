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

"""
The tests run only if *transformers* is available (they depend on the genuine
`transformers.models.llama.modeling_llama.LlamaAttention`).
"""

import importlib.util
import unittest

import torch

from tico.experimental.quantization.ptq.dtypes import DType
from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.llama.quant_attn import (
    QuantLlamaAttention,
)
from tico.experimental.quantization.ptq.wrappers.nn.quant_linear import QuantLinear


trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping LlamaAttention tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantLlamaAttention(unittest.TestCase):
    fp_attn: torch.nn.Module
    head_dim: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        cfg = LlamaConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
        )
        cls.fp_attn = LlamaAttention(cfg, layer_idx=0)
        cls.head_dim = cfg.head_dim  # 4

    # dummy RoPE tables with correct last dim
    def _rand_rope(self, B, S):
        h = self.head_dim
        emb = torch.randn(B, S, h)
        return emb.cos(), emb.sin()

    def test_mode_transitions(self):
        qattn = QuantLlamaAttention(self.fp_attn)
        self.assertIs(qattn._mode, Mode.NO_QUANT)

        qattn.enable_calibration()
        self.assertIs(qattn._mode, Mode.CALIB)

        x = torch.randn(2, 5, 8)
        pos = self._rand_rope(2, 5)
        _ = qattn(x, pos)  # gather stats

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

    def test_forward_diff(self):
        qattn = QuantLlamaAttention(self.fp_attn)
        qattn.enable_calibration()
        for _ in range(4):
            inp = torch.randn(2, 6, 8)
            pos = self._rand_rope(2, 6)
            _ = qattn(inp, pos)
        qattn.freeze_qparams()

        x = torch.randn(2, 6, 8)
        pos = self._rand_rope(2, 6)
        with torch.no_grad():
            q_out, _ = qattn(x, pos)
            fp_outs = self.fp_attn(x, position_embeddings=pos, attention_mask=None)
            fp_out = fp_outs[0]

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_per_projection_override(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8),
            overrides={
                "q_proj": {
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                }
            },
        )
        qattn = QuantLlamaAttention(self.fp_attn, qcfg=cfg)
        q_lin = qattn.q_proj.wrapped  # PTQWrapper → LinearQuant

        self.assertIsInstance(q_lin, QuantLinear)
        self.assertEqual(q_lin.act_in_obs.dtype, DType.uint(4))
        self.assertEqual(q_lin.act_out_obs.dtype, DType.uint(4))

    def test_forward_with_float_attention_mask(self):
        torch.manual_seed(123)

        # fresh wrapper
        qattn = QuantLlamaAttention(self.fp_attn)

        # build a float mask (all-zero here, but could include −1e4 etc.)
        B, S = 2, 4
        float_mask = torch.zeros(1, 1, S, S)
        # quick calibration
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(2, 4, 8)
            pos = self._rand_rope(2, 4)
            qattn(x, pos, float_mask)
        qattn.freeze_qparams()

        # run forward — should not raise
        x = torch.randn(B, S, 8)
        pos = self._rand_rope(B, S)
        with torch.no_grad():
            q_out, attn_w = qattn(x, pos, attention_mask=float_mask)
            fp_outs = self.fp_attn(
                x, position_embeddings=pos, attention_mask=float_mask
            )
            fp_out = fp_outs[0]

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(q_out.shape, (B, S, 8))
        self.assertEqual(attn_w.shape, (B, 2, S, S))
