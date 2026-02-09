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
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attn import (
    QuantQwen3VLTextAttention,
)


trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping Qwen3VLTextAttention tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantQwen3VLTextAttention(unittest.TestCase):
    fp_attn: torch.nn.Module
    head_dim: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        cfg = Qwen3VLTextConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=2048,
        )

        # Ensure eager attention implementation so outputs are deterministic
        # and do not require GPU flash attention kernels.
        # Some versions use `_attn_implementation`, others expose `attn_implementation`.
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"

        cls.fp_attn = Qwen3VLTextAttention(cfg, layer_idx=0)
        cls.head_dim = cfg.head_dim
        cls.hidden_size = cfg.hidden_size
        cls.num_heads = cfg.num_attention_heads
        cls.num_kv_heads = cfg.num_key_value_heads

    def _rand_rope(self, B: int, S: int):
        # Build dummy RoPE tables with shape (B, S, head_dim),
        # consistent with HF apply_rotary_pos_emb() expectations.
        h = self.head_dim
        emb = torch.randn(B, S, h)
        return emb.cos(), emb.sin()

    def test_mode_transitions(self):
        qattn = QuantQwen3VLTextAttention(self.fp_attn)
        self.assertIs(qattn._mode, Mode.NO_QUANT)

        qattn.enable_calibration()
        self.assertIs(qattn._mode, Mode.CALIB)

        x = torch.randn(2, 5, self.hidden_size)
        pos = self._rand_rope(2, 5)
        _ = qattn(x, pos)

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

    def test_forward_diff(self):
        qattn = QuantQwen3VLTextAttention(self.fp_attn)
        qattn.enable_calibration()
        for _ in range(4):
            inp = torch.randn(2, 6, self.hidden_size)
            pos = self._rand_rope(2, 6)
            _ = qattn(inp, pos)
        qattn.freeze_qparams()

        x = torch.randn(2, 6, self.hidden_size)
        pos = self._rand_rope(2, 6)
        with torch.no_grad():
            q_out, _ = qattn(x, pos, attention_mask=None)
            fp_out, _ = self.fp_attn(x, position_embeddings=pos, attention_mask=None)

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
        qattn = QuantQwen3VLTextAttention(self.fp_attn, qcfg=cfg)
        q_lin = qattn.q_proj.wrapped

        self.assertIsInstance(q_lin, QuantLinear)
        self.assertEqual(q_lin.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_lin.obs_act_out.dtype, DType.uint(4))

    def test_forward_with_float_attention_mask(self):
        torch.manual_seed(123)

        qattn = QuantQwen3VLTextAttention(self.fp_attn)

        B, S = 2, 4
        float_mask = torch.zeros(1, 1, S, S)  # additive mask (all zeros here)

        # Quick calibration
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(B, S, self.hidden_size)
            pos = self._rand_rope(B, S)
            _ = qattn(x, pos, attention_mask=float_mask)
        qattn.freeze_qparams()

        # Forward should not raise, and shapes should match
        x = torch.randn(B, S, self.hidden_size)
        pos = self._rand_rope(B, S)
        with torch.no_grad():
            q_out, attn_w = qattn(x, pos, attention_mask=float_mask)
            fp_out, fp_attn_w = self.fp_attn(
                x, position_embeddings=pos, attention_mask=float_mask
            )

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(q_out.shape, (B, S, self.hidden_size))
        self.assertEqual(attn_w.shape, (B, self.num_heads, S, S))
        self.assertEqual(fp_attn_w.shape, (B, self.num_heads, S, S))

    def test_cache_mock_object_update_prefill_then_decode(self):
        """
        `QuantQwen3VLTextAttention` uses HF Cache-like update().
        This test validates:
          - cache grows along sequence dim (dim=2)
          - attention weights use the grown key length
        """

        class MockCache:
            def __init__(self):
                self.k = None
                self.v = None

            def update(self, k, v, *args, **kwargs):
                # k, v: (B, n_kv, S, H)
                if self.k is None:
                    self.k = k
                    self.v = v
                else:
                    self.k = torch.cat([self.k, k], dim=2)  # type: ignore[list-item]
                    self.v = torch.cat([self.v, v], dim=2)  # type: ignore[list-item]
                return self.k, self.v

        torch.manual_seed(0)
        qattn = QuantQwen3VLTextAttention(self.fp_attn)

        # Minimal calibration
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(2, 3, self.hidden_size)
            pos = self._rand_rope(2, 3)
            _ = qattn(x, pos, attention_mask=None)
        qattn.freeze_qparams()

        cache = MockCache()
        B = 2

        # Prefill: S=4
        S0 = 4
        x0 = torch.randn(B, S0, self.hidden_size)
        pos0 = self._rand_rope(B, S0)
        with torch.no_grad():
            out0, attn0 = qattn(
                x0,
                pos0,
                attention_mask=None,
                past_key_values=cache,
                cache_position=torch.arange(S0),
            )
        self.assertEqual(out0.shape, (B, S0, self.hidden_size))
        self.assertEqual(attn0.shape, (B, self.num_heads, S0, S0))
        self.assertIsNotNone(cache.k)
        self.assertIsNotNone(cache.v)
        assert isinstance(cache.k, torch.Tensor)
        assert isinstance(cache.v, torch.Tensor)
        self.assertEqual(cache.k.shape, (B, self.num_kv_heads, S0, self.head_dim))
        self.assertEqual(cache.v.shape, (B, self.num_kv_heads, S0, self.head_dim))

        # Decode: S=1, total K should become S0+1
        S1 = 1
        x1 = torch.randn(B, S1, self.hidden_size)
        pos1 = self._rand_rope(B, S1)
        with torch.no_grad():
            out1, attn1 = qattn(
                x1,
                pos1,
                attention_mask=None,
                past_key_values=cache,
                cache_position=torch.tensor([S0]),
            )
        self.assertEqual(out1.shape, (B, S1, self.hidden_size))
        self.assertEqual(attn1.shape, (B, self.num_heads, S1, S0 + 1))
        self.assertEqual(cache.k.shape, (B, self.num_kv_heads, S0 + 1, self.head_dim))
        self.assertEqual(cache.v.shape, (B, self.num_kv_heads, S0 + 1, self.head_dim))

    def test_mask_slicing_with_cache_q_len_lt_k_len(self):
        """
        Validate causal mask slicing when q_len < k_len due to cache growth.
        This specifically checks that attention weights have shape (..., q_len, k_len).
        """
        torch.manual_seed(2)
        qattn = QuantQwen3VLTextAttention(self.fp_attn)

        # Calibrate and freeze
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(1, 5, self.hidden_size)
            pos = self._rand_rope(1, 5)
            _ = qattn(x, pos, attention_mask=None)
        qattn.freeze_qparams()

        class MockCache:
            def __init__(self):
                self.k = None
                self.v = None

            def update(self, k, v, *args, **kwargs):
                if self.k is None:
                    self.k = k
                    self.v = v
                else:
                    self.k = torch.cat([self.k, k], dim=2)  # type: ignore[list-item]
                    self.v = torch.cat([self.v, v], dim=2)  # type: ignore[list-item]
                return self.k, self.v

        cache = MockCache()

        # Prefill K=3
        B = 1
        x0 = torch.randn(B, 3, self.hidden_size)
        pos0 = self._rand_rope(B, 3)
        with torch.no_grad():
            _ = qattn(
                x0,
                pos0,
                attention_mask=None,
                past_key_values=cache,
                cache_position=torch.arange(3),
            )

        # Now decode with q_len=2 => k_len should be 5
        x1 = torch.randn(B, 2, self.hidden_size)
        pos1 = self._rand_rope(B, 2)
        with torch.no_grad():
            _, attn_w = qattn(
                x1,
                pos1,
                attention_mask=None,
                past_key_values=cache,
                cache_position=torch.arange(3, 5),
            )

        self.assertEqual(attn_w.shape, (B, self.num_heads, 2, 5))
        assert isinstance(cache.k, torch.Tensor)
        assert isinstance(cache.v, torch.Tensor)
        self.assertEqual(cache.k.shape, (B, self.num_kv_heads, 5, self.head_dim))
        self.assertEqual(cache.v.shape, (B, self.num_kv_heads, 5, self.head_dim))
