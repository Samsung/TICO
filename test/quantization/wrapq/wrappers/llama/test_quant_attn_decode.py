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

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.quant_attn_decode import (
    QuantLlamaAttentionDecode,
)
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

skip_msg = "required transformers not installed — skipping LlamaAttentionDecode tests"


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestQuantLlamaAttentionDecode(unittest.TestCase):
    fp_attn: torch.nn.Module
    head_dim: int
    n_kv: int
    n_h: int
    max_seq: int

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        cls.max_seq = 16
        cfg = LlamaConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
            max_position_embeddings=cls.max_seq,
        )
        cls.fp_attn = LlamaAttention(cfg, layer_idx=0)
        cls.head_dim = cfg.head_dim  # 4
        cls.n_kv = cfg.num_key_value_heads  # 1
        cls.n_h = cfg.num_attention_heads  # 2

    def _rand_rope_current(self, B: int):
        """Return dummy (cos, sin) RoPE tables for the *current* token only: (B,1,H)."""
        h = self.head_dim
        emb = torch.randn(B, 1, h)
        return emb.cos(), emb.sin()

    def _rand_additive_mask(self, B: int):
        """
        Additive attention mask: (B,1,max_seq).
        0 means keep, negative means masked.
        """
        max_seq = self.max_seq
        L_eff = torch.randint(low=1, high=max_seq + 1, size=(1,)).item()
        mask = torch.zeros(B, 1, max_seq, dtype=torch.float32)
        if L_eff < max_seq:
            mask[:, :, L_eff:] = float("-120")
        return mask

    def _rand_past(self, B: int):
        """Static past KV with shape (B,n_kv,max_seq-1,H). Past is assumed RoPE-applied."""
        H = self.head_dim
        max_seq = self.max_seq
        past_k = torch.randn(B, self.n_kv, max_seq - 1, H)
        past_v = torch.randn(B, self.n_kv, max_seq - 1, H)
        return past_k, past_v

    def test_mode_transitions(self):
        qattn = QuantLlamaAttentionDecode(self.fp_attn)
        self.assertIs(qattn._mode, Mode.NO_QUANT)

        qattn.enable_calibration()
        self.assertIs(qattn._mode, Mode.CALIB)

        B, D = 1, 8
        x = torch.randn(B, 1, D)
        pos = self._rand_rope_current(B)
        mask = self._rand_additive_mask(B)
        past = self._rand_past(B)

        # gather stats
        _ = qattn(
            hidden_states=x,
            position_embeddings=pos,
            attention_mask=mask,
            past_key_value=past,
            use_cache=True,
        )

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

    def test_forward_shapes_and_cache_delta(self):
        torch.manual_seed(1)

        qattn = QuantLlamaAttentionDecode(self.fp_attn)
        qattn.enable_calibration()
        for _ in range(3):
            x = torch.randn(1, 1, 8)
            pos = self._rand_rope_current(1)
            mask = self._rand_additive_mask(1)
            past = self._rand_past(1)
            _ = qattn(x, pos, mask, past, use_cache=True)
        qattn.freeze_qparams()

        B, D = 1, 8
        x = torch.randn(B, 1, D)
        pos = self._rand_rope_current(B)
        mask = self._rand_additive_mask(B)
        past_k, past_v = self._rand_past(B)

        with torch.no_grad():
            out, new_kv = qattn(
                hidden_states=x,
                position_embeddings=pos,
                attention_mask=mask,
                past_key_value=(past_k, past_v),
                use_cache=True,
            )

        self.assertEqual(out.shape, (B, 1, D))
        new_k, new_v = new_kv
        self.assertEqual(new_k.shape, (B, self.n_kv, 1, self.head_dim))
        self.assertEqual(new_v.shape, (B, self.n_kv, 1, self.head_dim))

    def test_forward_requires_additive_mask_not_bool(self):
        """
        Decode wrapper must reject boolean masks to avoid dynamic bool->float conversions.
        """
        qattn = QuantLlamaAttentionDecode(self.fp_attn)
        qattn.enable_calibration()

        B, D = 1, 8
        x = torch.randn(B, 1, D)
        pos = self._rand_rope_current(B)
        past = self._rand_past(B)

        bool_mask = torch.zeros(B, 1, self.max_seq, dtype=torch.bool)
        with self.assertRaises(AssertionError):
            _ = qattn(
                hidden_states=x,
                position_embeddings=pos,
                attention_mask=bool_mask,
                past_key_value=past,
                use_cache=True,
            )

    def test_forward_static_shape_asserts(self):
        """
        Decode wrapper must enforce strict static shapes:
        - x must be (B,1,D)
        - mask must be (B,1,max_seq)
        - past must be (B,n_kv,max_seq-1,H)
        - pos must be (B,1,H)
        """
        qattn = QuantLlamaAttentionDecode(self.fp_attn)

        B, D = 1, 8
        x_bad = torch.randn(B, 2, D)  # bad seq len
        pos = self._rand_rope_current(B)
        mask = self._rand_additive_mask(B)
        past = self._rand_past(B)
        with self.assertRaises(AssertionError):
            _ = qattn(x_bad, pos, mask, past, use_cache=True)

        x = torch.randn(B, 1, D)
        mask_bad = torch.zeros(B, 1, self.max_seq - 1, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            _ = qattn(x, pos, mask_bad, past, use_cache=True)

        past_k, past_v = past
        past_bad = (past_k[:, :, :-1, :], past_v)  # wrong length
        with self.assertRaises(AssertionError):
            _ = qattn(x, pos, mask, past_bad, use_cache=True)

        pos_bad = (torch.randn(B, 2, self.head_dim), torch.randn(B, 2, self.head_dim))
        with self.assertRaises(AssertionError):
            _ = qattn(x, pos_bad, mask, past, use_cache=True)

    def test_per_projection_override(self):
        """
        Verify that per-projection override flows into the wrapped QuantLinear observers.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            wrapper_variant="decode",
            overrides={
                "q_proj": {
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                }
            },
        )
        qattn = QuantLlamaAttentionDecode(self.fp_attn, qcfg=cfg)
        q_lin = qattn.q_proj.wrapped  # PTQWrapper -> QuantLinear (or LinearQuant)

        self.assertIsInstance(q_lin, QuantLinear)
        self.assertEqual(q_lin.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_lin.obs_act_out.dtype, DType.uint(4))

    def test_forward_diff_vs_self_consistency(self):
        """
        We do not have an easy FP baseline for this decode wrapper without reconstructing
        HF's cache/update semantics. Instead, we sanity-check that quantization changes
        the output slightly but preserves shape and does not explode.

        This is similar in spirit to the prefill test's diff bound, but compares:
            CALIB-mode output (no fake quant) vs QUANT-mode output (fake quant).
        """
        torch.manual_seed(7)

        qattn = QuantLlamaAttentionDecode(self.fp_attn)
        qattn.enable_calibration()

        # Calibrate a bit.
        for _ in range(4):
            x = torch.randn(1, 1, 8)
            pos = self._rand_rope_current(1)
            mask = self._rand_additive_mask(1)
            past = self._rand_past(1)
            _ = qattn(x, pos, mask, past, use_cache=True)

        # Capture a "fp-ish" output (still wrapper math, but pre-freeze).
        x = torch.randn(1, 1, 8)
        pos = self._rand_rope_current(1)
        mask = self._rand_additive_mask(1)
        past = self._rand_past(1)

        with torch.no_grad():
            cal_out, _ = qattn(x, pos, mask, past, use_cache=True)

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

        with torch.no_grad():
            q_out, _ = qattn(x, pos, mask, past, use_cache=True)

        diff = (cal_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        # Loose bound: model is tiny; quant noise shouldn't explode.
        self.assertLess(diff, 1.0)
        self.assertEqual(cal_out.shape, q_out.shape)
