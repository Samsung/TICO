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
from tico.experimental.quantization.config.ptq import PTQConfig

from tico.experimental.quantization.wrapq.dtypes import DType
from tico.experimental.quantization.wrapq.mode import Mode
from tico.experimental.quantization.wrapq.wrappers.llama.quant_attn import (
    QuantLlamaAttention,
)
from tico.experimental.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear


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
        cfg = PTQConfig(
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

    def test_cache_tuple_concat_prefill_then_decode(self):
        torch.manual_seed(0)

        qattn = QuantLlamaAttention(self.fp_attn)
        # Quick calibration on a couple of random batches so fake-quant params lock-in
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(2, 4, 8)
            pos = self._rand_rope(2, 4)
            _ = qattn(x, pos)
        qattn.freeze_qparams()

        B, S_prefill = 2, 4
        x0 = torch.randn(B, S_prefill, 8)
        pos0 = self._rand_rope(B, S_prefill)

        # Prefill: no past_key_value, but request cache
        with torch.no_grad():
            out0, attn_w0, present0 = qattn(
                x0, pos0, attention_mask=None, past_key_value=None, use_cache=True
            )

        # Check shapes after prefill
        self.assertEqual(out0.shape, (B, S_prefill, 8))
        self.assertEqual(attn_w0.shape, (B, 2, S_prefill, S_prefill))
        k0, v0 = present0
        # n_kv = 1, head_dim = 4
        self.assertEqual(k0.shape, (B, 1, S_prefill, self.head_dim))
        self.assertEqual(v0.shape, (B, 1, S_prefill, self.head_dim))

        # Decode step: S=1, pass tuple past
        S = 1
        x1 = torch.randn(B, S, 8)
        pos1 = self._rand_rope(B, 1)
        with torch.no_grad():
            out1, attn_w1, present1 = qattn(
                x1, pos1, attention_mask=None, past_key_value=present0, use_cache=True
            )

        # Now key length should be S_prefill + 1
        k1, v1 = present1
        self.assertEqual(out1.shape, (B, 1, 8))
        self.assertEqual(attn_w1.shape, (B, 2, 1, S_prefill + 1))
        self.assertEqual(k1.shape, (B, 1, S_prefill + 1, self.head_dim))
        self.assertEqual(v1.shape, (B, 1, S_prefill + 1, self.head_dim))

    def test_cache_mock_object_update(self):
        class MockCache:
            def __init__(self):
                self.k = None
                self.v = None

            def update(self, k, v):
                # k, v: (B, n_kv, S, H)
                if self.k is None:
                    self.k = k
                    self.v = v
                else:
                    self.k = torch.cat([self.k, k], dim=2)  # type: ignore[list-item]
                    self.v = torch.cat([self.v, v], dim=2)  # type: ignore[list-item]
                return self.k, self.v

        torch.manual_seed(1)
        qattn = QuantLlamaAttention(self.fp_attn)

        # Quick calibration
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(2, 3, 8)
            pos = self._rand_rope(2, 3)
            _ = qattn(x, pos)
        qattn.freeze_qparams()

        cache = MockCache()
        B = 2

        # Step 1: length = 2
        x_a = torch.randn(B, 2, 8)
        pos_a = self._rand_rope(B, 2)
        with torch.no_grad():
            out_a, attn_a, present_a = qattn(
                x_a, pos_a, past_key_value=cache, use_cache=True
            )
        k_a, v_a = present_a
        self.assertEqual(k_a.shape[2], 2)
        self.assertEqual(v_a.shape[2], 2)
        self.assertEqual(attn_a.shape, (B, 2, 2, 2))

        # Step 2: length += 1 (decode)
        x_b = torch.randn(B, 1, 8)
        pos_b = self._rand_rope(B, 1)
        with torch.no_grad():
            out_b, attn_b, present_b = qattn(
                x_b, pos_b, past_key_value=cache, use_cache=True
            )
        k_b, v_b = present_b
        self.assertEqual(k_b.shape[2], 3)
        self.assertEqual(v_b.shape[2], 3)
        self.assertEqual(attn_b.shape, (B, 2, 1, 3))
        self.assertEqual(out_b.shape, (B, 1, 8))

    def test_mask_slicing_with_cache_q_len_lt_k_len(self):
        torch.manual_seed(2)
        qattn = QuantLlamaAttention(self.fp_attn)

        # Calibrate and freeze
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(1, 5, 8)
            pos = self._rand_rope(1, 5)
            _ = qattn(x, pos)
        qattn.freeze_qparams()

        # Prefill K=3
        B = 1
        x0 = torch.randn(B, 3, 8)
        pos0 = self._rand_rope(B, 3)
        with torch.no_grad():
            _, _, present0 = qattn(x0, pos0, use_cache=True)

        x1 = torch.randn(B, 2, 8)
        pos1 = self._rand_rope(B, 2)
        with torch.no_grad():
            _, attn_w, present1 = qattn(
                x1, pos1, past_key_value=present0, use_cache=True
            )
        # n_h = 2, q_len = 2, k_len = 5 (3 + 2)
        self.assertEqual(attn_w.shape, (B, 2, 2, 5))
        k1, v1 = present1
        self.assertEqual(k1.shape, (B, 1, 5, self.head_dim))
        self.assertEqual(v1.shape, (B, 1, 5, self.head_dim))

    def test_use_cache_no_past_equivalence(self):
        torch.manual_seed(3)
        qattn = QuantLlamaAttention(self.fp_attn)

        # Calibrate and freeze
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(2, 4, 8)
            pos = self._rand_rope(2, 4)
            _ = qattn(x, pos)
        qattn.freeze_qparams()

        B, S = 2, 4
        x = torch.randn(B, S, 8)
        pos = self._rand_rope(B, S)

        with torch.no_grad():
            out_nc, attn_nc = qattn(x, pos)  # no cache
            out_c, attn_c, present = qattn(x, pos, use_cache=True)  # cache requested

        # Outputs must match closely; caching should not change the result
        self.assertTrue(torch.allclose(out_nc, out_c, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(attn_nc, attn_c, atol=1e-5, rtol=1e-5))
        k, v = present
        self.assertEqual(k.shape, (B, 1, S, self.head_dim))
        self.assertEqual(v.shape, (B, 1, S, self.head_dim))

    def test_cache_grows_across_multiple_single_token_steps(self):
        torch.manual_seed(4)
        qattn = QuantLlamaAttention(self.fp_attn)

        # Minimal calibration
        qattn.enable_calibration()
        _ = qattn(torch.randn(1, 2, 8), self._rand_rope(1, 2))
        qattn.freeze_qparams()

        B = 1
        # Start with an empty past via tuple path
        past = None
        total_len = 0
        for step in range(5):
            x = torch.randn(B, 1, 8)
            pos = self._rand_rope(B, 1)
            with torch.no_grad():
                _, _, present = qattn(x, pos, past_key_value=past, use_cache=True)
            k, v = present
            total_len += 1 if past is None else 1
            self.assertEqual(k.shape[2], total_len)
            self.assertEqual(v.shape[2], total_len)
            past = present  # feed back as past
