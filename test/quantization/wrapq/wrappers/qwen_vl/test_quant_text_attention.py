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

import inspect
import unittest

import torch
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attention import (
    QuantQwen3VLTextAttention,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


skip_msg = "required transformers not installed — skipping Qwen3VLTextAttention tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
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
        """Create synthetic RoPE tables with Qwen text-attention shapes."""
        h = self.head_dim
        emb = torch.randn(B, S, h)
        return emb.cos(), emb.sin()

    def _collect_cache_calibration(self, qattn: QuantQwen3VLTextAttention) -> None:
        """Collect observer statistics for static KV cache tensors."""
        batch_size = 2
        past_len = 2
        q_len = 1
        past_k = torch.randn(batch_size, self.num_kv_heads, past_len, self.head_dim)
        past_v = torch.randn_like(past_k)
        hidden = torch.randn(batch_size, q_len, self.hidden_size)
        pos = self._rand_rope(batch_size, q_len)
        mask = torch.zeros(batch_size, 1, q_len, past_len + q_len)
        _ = qattn(
            hidden,
            pos,
            attention_mask=mask,
            past_key_values=(past_k, past_v),
            use_cache=True,
            cache_output_mode="present",
        )

    def _calibrate_cache_paths(self, qattn: QuantQwen3VLTextAttention) -> None:
        """Calibrate the no-cache and static tuple-cache execution paths."""
        qattn.enable_calibration()

        batch_size, prefill_len, decode_len = 2, 4, 1
        for _ in range(2):
            x0 = torch.randn(batch_size, prefill_len, self.hidden_size)
            pos0 = self._rand_rope(batch_size, prefill_len)
            _, _, delta0 = qattn(
                x0,
                pos0,
                attention_mask=None,
                use_cache=True,
                cache_output_mode="delta",
            )

            past_k, past_v = delta0
            x1 = torch.randn(batch_size, decode_len, self.hidden_size)
            pos1 = self._rand_rope(batch_size, decode_len)
            _ = qattn(
                x1,
                pos1,
                attention_mask=None,
                past_key_values=(past_k, past_v),
                use_cache=True,
                cache_output_mode="delta",
            )

        qattn.freeze_qparams()

    def test_mode_transitions(self):
        qattn = QuantQwen3VLTextAttention(self.fp_attn)
        self.assertIs(qattn._mode, Mode.NO_QUANT)

        qattn.enable_calibration()
        self.assertIs(qattn._mode, Mode.CALIB)

        x = torch.randn(2, 5, self.hidden_size)
        pos = self._rand_rope(2, 5)
        _ = qattn(x, pos)
        self._collect_cache_calibration(qattn)

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

    def test_default_attention_profile_is_unrolled(self):
        """Verify that the default profile keeps the NPU-export-friendly layout."""
        qattn = QuantQwen3VLTextAttention(self.fp_attn)

        self.assertEqual(qattn.attn_options.layout, "unrolled")
        self.assertEqual(qattn.attn_options.scale_fusion, "k_norm")

    def test_reference_eval_profile_uses_batched_attention(self):
        """Verify that the reference profile selects the HF-like batched path."""
        cfg = make_affine_ptq_config(model_args={"profile": "reference_eval"})
        qattn = QuantQwen3VLTextAttention(self.fp_attn, qcfg=cfg)

        self.assertEqual(qattn.attn_options.layout, "batched")
        self.assertEqual(qattn.attn_options.scale_fusion, "none")

        x = torch.randn(2, 5, self.hidden_size)
        pos = self._rand_rope(2, 5)
        with torch.no_grad():
            out, attn_weights = qattn(x, pos, attention_mask=None)

        self.assertEqual(out.shape, (2, 5, self.hidden_size))
        self.assertEqual(attn_weights.shape, (2, self.num_heads, 5, 5))

    def test_batched_layout_matches_unrolled_layout_without_quantization(self):
        """Compare batched and unrolled layouts before fake quantization is enabled."""
        torch.manual_seed(11)
        unrolled = QuantQwen3VLTextAttention(self.fp_attn)
        cfg = make_affine_ptq_config(model_args={"profile": "reference_eval"})
        batched = QuantQwen3VLTextAttention(self.fp_attn, qcfg=cfg)

        x = torch.randn(2, 5, self.hidden_size)
        pos = self._rand_rope(2, 5)
        with torch.no_grad():
            out_unrolled, attn_unrolled = unrolled(x, pos, attention_mask=None)
            out_batched, attn_batched = batched(x, pos, attention_mask=None)

        self.assertTrue(torch.allclose(out_unrolled, out_batched, rtol=1e-4, atol=1e-5))
        self.assertTrue(
            torch.allclose(attn_unrolled, attn_batched, rtol=1e-4, atol=1e-5)
        )

    def test_forward_diff(self):
        qattn = QuantQwen3VLTextAttention(self.fp_attn)
        qattn.enable_calibration()
        for _ in range(4):
            inp = torch.randn(2, 6, self.hidden_size)
            pos = self._rand_rope(2, 6)
            _ = qattn(inp, pos)
        self._collect_cache_calibration(qattn)
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
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
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
        self._collect_cache_calibration(qattn)
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

    def test_public_cache_argument_is_past_key_values(self):
        """Verify that the public static-runtime cache argument is plural only."""
        params = inspect.signature(QuantQwen3VLTextAttention.forward).parameters
        self.assertIn("past_key_values", params)
        self.assertNotIn("past_key_value", params)

    def test_static_tuple_cache_returns_delta_kv_for_prefill_and_decode(self):
        """Validate delta-only KV outputs for the static tuple-cache path."""
        torch.manual_seed(7)
        qattn = QuantQwen3VLTextAttention(self.fp_attn)
        self._calibrate_cache_paths(qattn)

        B, S0 = 2, 4
        x0 = torch.randn(B, S0, self.hidden_size)
        pos0 = self._rand_rope(B, S0)

        with torch.no_grad():
            out0, attn0, delta0 = qattn(
                x0,
                pos0,
                attention_mask=None,
                use_cache=True,
                cache_output_mode="delta",
            )

        new_k0, new_v0 = delta0
        self.assertEqual(out0.shape, (B, S0, self.hidden_size))
        self.assertEqual(attn0.shape, (B, self.num_heads, S0, S0))
        self.assertEqual(new_k0.shape, (B, self.num_kv_heads, S0, self.head_dim))
        self.assertEqual(new_v0.shape, (B, self.num_kv_heads, S0, self.head_dim))

        x1 = torch.randn(B, 1, self.hidden_size)
        pos1 = self._rand_rope(B, 1)
        with torch.no_grad():
            out1, attn1, delta1 = qattn(
                x1,
                pos1,
                attention_mask=None,
                past_key_values=(new_k0, new_v0),
                use_cache=True,
                cache_output_mode="delta",
            )

        new_k1, new_v1 = delta1
        self.assertEqual(out1.shape, (B, 1, self.hidden_size))
        self.assertEqual(attn1.shape, (B, self.num_heads, 1, S0 + 1))
        self.assertEqual(new_k1.shape, (B, self.num_kv_heads, 1, self.head_dim))
        self.assertEqual(new_v1.shape, (B, self.num_kv_heads, 1, self.head_dim))

    def test_static_tuple_cache_returns_present_kv_when_requested(self):
        """Validate full present KV outputs for the static tuple-cache path."""
        torch.manual_seed(9)
        qattn = QuantQwen3VLTextAttention(self.fp_attn)
        self._calibrate_cache_paths(qattn)

        B, S0 = 1, 3
        x0 = torch.randn(B, S0, self.hidden_size)
        pos0 = self._rand_rope(B, S0)

        with torch.no_grad():
            _, _, present0 = qattn(
                x0,
                pos0,
                attention_mask=None,
                use_cache=True,
                cache_output_mode="present",
            )

        present_k0, present_v0 = present0
        self.assertEqual(present_k0.shape, (B, self.num_kv_heads, S0, self.head_dim))
        self.assertEqual(present_v0.shape, (B, self.num_kv_heads, S0, self.head_dim))

        x1 = torch.randn(B, 1, self.hidden_size)
        pos1 = self._rand_rope(B, 1)
        with torch.no_grad():
            _, attn1, present1 = qattn(
                x1,
                pos1,
                attention_mask=None,
                past_key_values=(present_k0, present_v0),
                use_cache=True,
                cache_output_mode="present",
            )

        present_k1, present_v1 = present1
        self.assertEqual(attn1.shape, (B, self.num_heads, 1, S0 + 1))
        self.assertEqual(
            present_k1.shape,
            (B, self.num_kv_heads, S0 + 1, self.head_dim),
        )
        self.assertEqual(
            present_v1.shape,
            (B, self.num_kv_heads, S0 + 1, self.head_dim),
        )

    def test_invalid_cache_output_mode_raises(self):
        """Reject unsupported cache output policies."""
        qattn = QuantQwen3VLTextAttention(self.fp_attn)
        x = torch.randn(1, 2, self.hidden_size)
        pos = self._rand_rope(1, 2)

        with self.assertRaises(ValueError):
            _ = qattn(
                x,
                pos,
                attention_mask=None,
                use_cache=True,
                cache_output_mode="full",  # type: ignore[arg-type]
            )

    def test_cache_mock_object_update_prefill_then_decode(self):
        """
        Validate HF Cache-like update semantics for eager wrapper execution.

        The exported static runtime uses tuple caches, but full-model eager
        execution may still pass a Cache-like object with an update method.
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
        self._collect_cache_calibration(qattn)
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
        Validate causal mask slicing when q_len is smaller than cached key length.
        """
        torch.manual_seed(2)
        qattn = QuantQwen3VLTextAttention(self.fp_attn)

        # Calibrate and freeze
        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(1, 5, self.hidden_size)
            pos = self._rand_rope(1, 5)
            _ = qattn(x, pos, attention_mask=None)
        self._collect_cache_calibration(qattn)
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

        # Now decode with q_len=2, so k_len should be 5.
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

    def test_qwen_text_attention_bool_mask_is_combined_with_causal_mask(self):
        qcfg = make_affine_ptq_config(attention_mask_fill_value=-100.0)
        attn = QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg)

        mask = torch.full((1, 1, 8, 8), -100.0)
        mask.triu_(1)
        attn.register_buffer("causal_mask_template", mask, persistent=False)

        # B=1, K=4. Last key is padding, so every query should mask key index 3.
        keep_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

        out = attn._build_attention_mask(
            attention_mask=keep_mask,
            q_len=4,
            k_len=4,
            device=torch.device("cpu"),
        )

        assert out.shape == (1, 1, 4, 4)

        # Causal mask: query 0 cannot attend to keys 1, 2, and 3.
        assert out[0, 0, 0, 0].item() == 0.0
        assert out[0, 0, 0, 1].item() == -100.0
        assert out[0, 0, 0, 2].item() == -100.0

        # Padding mask: key 3 is masked for all queries.
        assert torch.all(out[..., 3] == -100.0)

        # Query 2 can attend to keys 0, 1, and 2, but not key 3.
        assert out[0, 0, 2, 0].item() == 0.0
        assert out[0, 0, 2, 1].item() == 0.0
        assert out[0, 0, 2, 2].item() == 0.0
        assert out[0, 0, 2, 3].item() == -100.0
