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
import io
import itertools
import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

from unittest.mock import patch

import torch
from tico.quantization.config.specs import affine, mx
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attention import (
    QuantQwen3VLTextAttention,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config

skip_msg = "required transformers not installed — skipping Qwen3VLTextAttention tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLTextAttention(unittest.TestCase):
    fp_attn: "Qwen3VLTextAttention"
    head_dim: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(71)
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
            max_position_embeddings=16,
        )

        # Ensure eager attention implementation so outputs are deterministic
        # and do not require GPU flash attention kernels.
        # Some versions use `_attn_implementation`, others expose `attn_implementation`.
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"

        cls.fp_attn = Qwen3VLTextAttention(cfg, layer_idx=0)
        assert cfg.head_dim is not None
        cls.head_dim = cfg.head_dim
        cls.hidden_size = cfg.hidden_size
        cls.num_heads = cfg.num_attention_heads
        assert cfg.num_key_value_heads is not None
        cls.num_kv_heads = cfg.num_key_value_heads

    def setUp(self):
        torch.manual_seed(71)

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

    def test_cache_calibration_does_not_filter_masked_past_positions(self):
        """Compute statistics cover all past positions; storage observes new KV."""
        qattn = QuantQwen3VLTextAttention(self.fp_attn)
        fill = float(qattn.qcfg.attention_mask_fill_value)
        past_k = torch.tensor([1000.0, 1.0, 2.0, -1000.0]).reshape(1, 1, 4, 1)
        past_k = past_k.expand(1, self.num_kv_heads, 4, self.head_dim)
        past_v = torch.tensor([3000.0, 3.0, 4.0, -3000.0]).reshape(1, 1, 4, 1)
        past_v = past_v.expand_as(past_k)
        hidden = torch.zeros(1, 1, self.hidden_size)
        rope = (torch.ones(1, 1, self.head_dim), torch.zeros(1, 1, self.head_dim))
        for padding in (float("-inf"), fill - 1.0):
            with self.subTest(padding=padding):
                qattn.enable_calibration()
                mask = torch.tensor([fill, -0.25, 0.1, padding, 0.0]).reshape(
                    1, 1, 1, 5
                )
                qattn(
                    hidden,
                    rope,
                    attention_mask=mask,
                    past_key_values=(past_k, past_v),
                    use_cache=True,
                    cache_output_mode="delta",
                )
                for observer, extent in (
                    (qattn.obs_present_key, 1000.0),
                    (qattn.obs_present_value, 3000.0),
                ):
                    assert isinstance(observer, AffineObserverBase)
                    self.assertEqual(observer.min_val.item(), -extent)
                    self.assertEqual(observer.max_val.item(), extent)
                for observer in (qattn.obs_new_k, qattn.obs_new_v):
                    assert isinstance(observer, AffineObserverBase)
                    self.assertEqual(observer.min_val.item(), 0.0)
                    self.assertEqual(observer.max_val.item(), 0.0)
                self.assertFalse(hasattr(qattn, "obs_past_key"))
                self.assertFalse(hasattr(qattn, "obs_past_value"))

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

    def test_unrolled_layout_runs_attention_per_query_head(self):
        """Verify that grouped-query attention does not broadcast a KV head."""
        qattn = QuantQwen3VLTextAttention(self.fp_attn)

        batch_size, seq_len = 1, 4
        x = torch.randn(batch_size, seq_len, self.hidden_size)
        pos = self._rand_rope(batch_size, seq_len)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)

        logits_shapes: list[tuple[int, ...]] = []
        apply_scale = qattn._apply_attention_scale_if_needed

        def record_logits_shape(logits: torch.Tensor) -> torch.Tensor:
            logits_shapes.append(tuple(logits.shape))
            return apply_scale(logits)

        with patch.object(
            qattn,
            "_apply_attention_scale_if_needed",
            side_effect=record_logits_shape,
        ):
            with torch.no_grad():
                out, attn_weights = qattn(x, pos, attention_mask=mask)

        expected_logits_shape = (batch_size, seq_len, seq_len)
        self.assertEqual(logits_shapes, [expected_logits_shape] * self.num_heads)
        self.assertEqual(out.shape, (batch_size, seq_len, self.hidden_size))
        self.assertEqual(
            attn_weights.shape,
            (batch_size, self.num_heads, seq_len, seq_len),
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
            out0, attn0, _ = qattn(
                x0,
                pos0,
                attention_mask=None,
                past_key_values=cache,
                cache_position=torch.arange(S0),
                use_cache=True,
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
            out1, attn1, _ = qattn(
                x1,
                pos1,
                attention_mask=None,
                past_key_values=cache,
                cache_position=torch.tensor([S0]),
                use_cache=True,
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
                use_cache=True,
            )

        # Now decode with q_len=2, so k_len should be 5.
        x1 = torch.randn(B, 2, self.hidden_size)
        pos1 = self._rand_rope(B, 2)
        with torch.no_grad():
            _, attn_w, _ = qattn(
                x1,
                pos1,
                attention_mask=None,
                past_key_values=cache,
                cache_position=torch.arange(3, 5),
                use_cache=True,
            )

        self.assertEqual(attn_w.shape, (B, self.num_heads, 2, 5))
        assert isinstance(cache.k, torch.Tensor)
        assert isinstance(cache.v, torch.Tensor)
        self.assertEqual(cache.k.shape, (B, self.num_kv_heads, 5, self.head_dim))
        self.assertEqual(cache.v.shape, (B, self.num_kv_heads, 5, self.head_dim))

    def test_qwen_text_attention_bool_mask_is_combined_with_causal_mask(self):
        qcfg = make_affine_ptq_config(attention_mask_fill_value=-100.0)
        attn = QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg)

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

    def _prefill_calibrated(self, *, dtype=DType.uint(8), layout="unrolled"):
        qcfg = make_affine_ptq_config(
            dtype=dtype,
            model_args={"attention": {"layout": layout}},
        )
        attention = QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg).eval()
        attention.enable_calibration()
        with torch.no_grad():
            attention(
                torch.randn(2, 4, self.hidden_size),
                self._rand_rope(2, 4),
                use_cache=False,
            )
        attention.freeze_qparams()
        return attention

    @staticmethod
    def _dynamic_kv(cache: "DynamicCache") -> tuple[torch.Tensor, torch.Tensor]:
        keys = getattr(cache.layers[0], "keys", None)
        values = getattr(cache.layers[0], "values", None)
        assert isinstance(keys, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        return keys, values

    def test_prefill_without_cache_output_calibrates_only_storage_and_compute(self):
        for layout in ("batched", "unrolled"):
            with self.subTest(layout=layout):
                attention = self._prefill_calibrated(layout=layout)
                names = dict(attention.named_observers(recurse=False))
                self.assertNotIn("past_key", names)
                self.assertNotIn("past_value", names)
                for storage_name, compute_name in (
                    ("new_k", "present_key"),
                    ("new_v", "present_value"),
                ):
                    storage = names[storage_name]
                    compute = names[compute_name]
                    self.assertIsNot(storage, compute)
                    assert isinstance(storage, AffineObserverBase)
                    assert isinstance(compute, AffineObserverBase)
                    for observer in (storage, compute):
                        self.assertTrue(torch.isfinite(observer.min_val).all())
                        self.assertTrue(torch.isfinite(observer.max_val).all())
                        self.assertTrue(observer.has_qparams)
                        self.assertFalse(observer.enabled)
                    torch.testing.assert_close(storage.min_val, compute.min_val)
                    torch.testing.assert_close(storage.max_val, compute.max_val)
                attention.require_calibrated_cache()
                self.assertIsInstance(
                    attention.as_export_module("decode"), torch.nn.Module
                )

    def test_storage_contract_across_cache_types_layouts_and_output_modes(self):
        """A prefill and two decodes keep S-grid storage identical for tuple/HF."""
        from transformers.cache_utils import DynamicCache

        for dtype, layout, equal_grids, output_mode in itertools.product(
            (DType.uint(8), DType.int(16)),
            ("batched", "unrolled"),
            (False, True),
            ("delta", "present"),
        ):
            with (
                self.subTest(
                    dtype=dtype, layout=layout, equal=equal_grids, mode=output_mode
                ),
                torch.no_grad(),
            ):
                attention = self._prefill_calibrated(dtype=dtype, layout=layout)
                storage_zp = 0 if dtype.signed else 113
                compute_zp = storage_zp if equal_grids else (0 if dtype.signed else 91)
                for observer in (attention.obs_new_k, attention.obs_new_v):
                    assert isinstance(observer, AffineObserverBase)
                    observer.load_qparams(
                        torch.tensor(0.09375), torch.tensor(storage_zp)
                    )
                for observer in (
                    attention.obs_present_key,
                    attention.obs_present_value,
                ):
                    assert isinstance(observer, AffineObserverBase)
                    observer.load_qparams(
                        torch.tensor(0.09375 if equal_grids else 0.21875),
                        torch.tensor(compute_zp),
                    )
                frozen = {
                    name: (observer._cached_scale.clone(), observer._cached_zp.clone())
                    for name, observer in attention.named_observers(recurse=False)
                    if isinstance(observer, AffineObserverBase)
                }
                hf_cache = DynamicCache(config=attention.config)
                tuple_cache = None
                length = 0
                for query_length in (3, 1, 1):
                    hidden = torch.randn(2, query_length, self.hidden_size)
                    rope = self._rand_rope(2, query_length)
                    keep = torch.ones(2, length + query_length, dtype=torch.bool)
                    # A real padding slot persists through both cached decodes.
                    keep[0, 0] = False
                    _, _, delta = attention(
                        hidden, rope, use_cache=True, cache_output_mode="delta"
                    )
                    expected_cache = (
                        delta
                        if tuple_cache is None
                        else tuple(
                            torch.cat((past, new), dim=2)
                            for past, new in zip(tuple_cache, delta)
                        )
                    )
                    static_result = attention(
                        hidden,
                        rope,
                        attention_mask=keep,
                        past_key_values=tuple_cache,
                        use_cache=True,
                        cache_output_mode=output_mode,
                    )
                    hf_result = attention(
                        hidden,
                        rope,
                        attention_mask=keep,
                        past_key_values=hf_cache,
                        cache_position=torch.arange(length, length + query_length),
                        use_cache=True,
                        cache_output_mode=output_mode,
                    )
                    for static_tensor, hf_tensor in zip(
                        static_result[:2], hf_result[:2]
                    ):
                        # The same attention graph must see the same C-grid inputs.
                        torch.testing.assert_close(
                            static_tensor, hf_tensor, atol=0, rtol=0
                        )
                    expected_output = (
                        delta if output_mode == "delta" else expected_cache
                    )
                    hf_output = hf_result[2]
                    if output_mode == "present":
                        self.assertIs(hf_output, hf_cache)
                        hf_output = self._dynamic_kv(hf_cache)
                    for actual, expected in zip(static_result[2], expected_output):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for actual, expected in zip(hf_output, expected_output):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for actual, expected, observer in zip(
                        self._dynamic_kv(hf_cache),
                        expected_cache,
                        (attention.obs_new_k, attention.obs_new_v),
                    ):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        torch.testing.assert_close(
                            actual, observer.fake_quant(actual), atol=0, rtol=0
                        )
                    tuple_cache = expected_cache
                    length += query_length
                self.assertEqual(hf_cache.get_seq_length(), 5)
                for name, observer in attention.named_observers(recurse=False):
                    if isinstance(observer, AffineObserverBase):
                        scale, zero_point = frozen[name]
                        torch.testing.assert_close(
                            observer._cached_scale, scale, atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            observer._cached_zp, zero_point, atol=0, rtol=0
                        )

    def test_use_cache_false_reads_without_mutating_dynamic_cache(self):
        from transformers.cache_utils import DynamicCache

        for layout in ("batched", "unrolled"):
            with self.subTest(layout=layout), torch.no_grad():
                attention = self._prefill_calibrated(layout=layout)
                cache = DynamicCache(config=attention.config)
                attention(
                    torch.randn(1, 3, self.hidden_size),
                    self._rand_rope(1, 3),
                    past_key_values=cache,
                    use_cache=True,
                )
                past = tuple(value.clone() for value in self._dynamic_kv(cache))
                hidden = torch.randn(1, 1, self.hidden_size)
                rope = self._rand_rope(1, 1)
                expected = attention(
                    hidden, rope, past_key_values=past, use_cache=False
                )
                actual = attention(hidden, rope, past_key_values=cache, use_cache=False)
                self.assertEqual(len(actual), 2)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, atol=0, rtol=0)
                self.assertEqual(cache.get_seq_length(), 3)
                for actual, expected in zip(self._dynamic_kv(cache), past):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                empty_cache = DynamicCache(config=attention.config)
                attention(hidden, rope, past_key_values=empty_cache, use_cache=False)
                self.assertEqual(empty_cache.get_seq_length(), 0)

    def test_obsolete_past_overrides_are_rejected_with_migration_guidance(self):
        for name in ("past_key", "past_value"):
            with self.subTest(observer=name):
                qcfg = make_affine_ptq_config(
                    overrides={name: {"dtype": DType.uint(8)}}
                )
                with self.assertRaisesRegex(
                    ValueError, "new_k.*new_v.*present_key.*present_value"
                ):
                    QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg)

    def test_obsolete_state_dict_rejected_even_for_nonstrict_nested_load(self):
        for strict, prefix in itertools.product((False, True), ("", "attention.")):
            with self.subTest(strict=strict, prefix=prefix):
                attention = self._prefill_calibrated()
                target = (
                    torch.nn.ModuleDict({"attention": attention})
                    if prefix
                    else attention
                )
                state = target.state_dict()
                state[prefix + "obs_past_key.min_val"] = torch.tensor(-1.0)
                state[prefix + "obs_past_value.max_val"] = torch.tensor(1.0)
                with self.assertRaisesRegex(RuntimeError, "reprepare.*recalibrate"):
                    target.load_state_dict(state, strict=strict)

    def test_obsolete_serialized_module_requires_repreparation(self):
        for name in ("obs_past_key", "obs_past_value"):
            with self.subTest(observer=name):
                attention = self._prefill_calibrated()
                attention.add_module(name, MinMaxObserver(name=name))
                checkpoint = io.BytesIO()
                torch.save(attention, checkpoint)
                checkpoint.seek(0)
                with self.assertRaisesRegex(RuntimeError, "reprepare.*recalibrate"):
                    torch.load(checkpoint, weights_only=False)

    def test_two_domain_checkpoints_preserve_frozen_qparams_and_outputs(self):
        attention = self._prefill_calibrated()
        hidden = torch.randn(1, 2, self.hidden_size)
        rope = self._rand_rope(1, 2)
        # Materialize the per-channel projection observer buffer shapes before
        # loading, as required by the existing affine observer state-dict API.
        state_clone = self._prefill_calibrated()
        state_clone.load_state_dict(attention.state_dict())
        state_clone.freeze_qparams()
        checkpoint = io.BytesIO()
        torch.save(attention, checkpoint)
        checkpoint.seek(0)
        module_clone = torch.load(checkpoint, weights_only=False)
        with torch.no_grad():
            expected = attention(hidden, rope, use_cache=True)
            for clone in (state_clone, module_clone):
                actual = clone(hidden, rope, use_cache=True)
                for value, reference in zip(
                    actual[:2] + actual[2], expected[:2] + expected[2]
                ):
                    torch.testing.assert_close(value, reference, atol=0, rtol=0)
                for name in ("new_k", "new_v", "present_key", "present_value"):
                    original = attention.get_observer(name)
                    restored = clone.get_observer(name)
                    assert isinstance(original, AffineObserverBase)
                    assert isinstance(restored, AffineObserverBase)
                    torch.testing.assert_close(
                        original._cached_scale, restored._cached_scale, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        original._cached_zp, restored._cached_zp, atol=0, rtol=0
                    )

    def test_matching_token_local_mx_cache_policies_preserve_tuple_hf_parity(self):
        from transformers.cache_utils import DynamicCache

        qcfg = make_affine_ptq_config(
            overrides={
                name: mx(elem_format="int8", axis=-1)
                for name in ("new_k", "new_v", "present_key", "present_value")
            }
        )
        attention = QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg).eval()
        attention.enable_calibration()
        hidden = torch.randn(1, 3, self.hidden_size)
        rope = self._rand_rope(1, 3)
        attention(hidden, rope, use_cache=False)
        attention.freeze_qparams()
        cache = DynamicCache(config=attention.config)
        with torch.no_grad():
            _, _, past = attention(hidden, rope, use_cache=True)
            attention(hidden, rope, past_key_values=cache, use_cache=True)
            hidden = torch.randn(1, 1, self.hidden_size)
            rope = self._rand_rope(1, 1)
            expected = attention(hidden, rope, past_key_values=past, use_cache=True)
            actual = attention(hidden, rope, past_key_values=cache, use_cache=True)
        for value, reference in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        for value, reference in zip(self._dynamic_kv(cache), expected[2]):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)

    def test_unsupported_mx_cache_grid_combinations_are_explicit(self):
        token_mx = mx(elem_format="int8", axis=-1)
        affine_policy = affine(DType.uint(8))
        for storage, compute in (
            (token_mx, affine_policy),
            (affine_policy, token_mx),
            (token_mx, mx(elem_format="int4", axis=-1)),
            (mx(elem_format="int8", axis=2), mx(elem_format="int8", axis=2)),
        ):
            with self.subTest(storage=storage, compute=compute):
                qcfg = make_affine_ptq_config(
                    overrides={
                        "new_k": storage,
                        "new_v": storage,
                        "present_key": compute,
                        "present_value": compute,
                    }
                )
                with self.assertRaisesRegex((ValueError, RuntimeError), "MX"):
                    attention = QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg)
                    attention.enable_calibration()
                    attention(
                        torch.randn(1, 2, self.hidden_size), self._rand_rope(1, 2)
                    )
                    attention.freeze_qparams()
                    attention.as_export_module("decode")

    def test_legacy_stateless_cache_checkpoint_metadata_requires_repreparation(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                qcfg = make_affine_ptq_config(
                    overrides={
                        name: mx(elem_format="int8", axis=-1)
                        for name in ("new_k", "new_v", "present_key", "present_value")
                    }
                )
                attention = QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg)
                attention.add_module("obs_past_key", MXObserver(name="past_key"))
                attention.add_module("obs_past_value", MXObserver(name="past_value"))
                source = torch.nn.ModuleDict({"attention": attention})
                target = torch.nn.ModuleDict(
                    {"attention": QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg)}
                )
                state = source.state_dict()
                # Recreate the old stateless schema, which had no observer keys.
                for key in list(state):
                    if "obs_past_" in key:
                        del state[key]
                getattr(state, "_metadata")["attention"]["version"] = 1
                with self.assertRaisesRegex(RuntimeError, "reprepare.*recalibrate"):
                    target.load_state_dict(state, strict=strict)

    def test_affine_cache_channel_axis_cannot_span_sequence(self):
        for axis in (2, -2):
            with self.subTest(axis=axis):
                qcfg = make_affine_ptq_config(
                    overrides={
                        name: affine(
                            DType.int(16),
                            qscheme=QScheme.PER_CHANNEL_SYMM,
                            channel_axis=axis,
                        )
                        for name in ("new_k", "new_v", "present_key", "present_value")
                    }
                )
                with self.assertRaisesRegex((ValueError, RuntimeError), "sequence"):
                    attention = QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg)
                    attention.enable_calibration()
                    attention(
                        torch.randn(1, 2, self.hidden_size), self._rand_rope(1, 2)
                    )
                    attention.freeze_qparams()
                    attention.as_export_module("decode")

    def test_fused_scale_sites_are_structurally_inactive(self):
        for fusion in ("k_norm", "none"):
            with self.subTest(fusion=fusion):
                qcfg = make_affine_ptq_config(
                    model_args={"attention": {"scale_fusion": fusion}}
                )
                attention = QuantQwen3VLTextAttention(self.fp_attn, qcfg=qcfg)
                attention.enable_calibration()
                hidden = torch.randn(1, 3, self.hidden_size)
                rope = self._rand_rope(1, 3)
                attention(hidden, rope, use_cache=False)
                attention.freeze_qparams()
                names = dict(attention.named_observers(recurse=False))
                self.assertEqual("logits_raw" in names, fusion == "none")
                self.assertEqual("scale" in names, fusion == "none")
                self.assertEqual(len(attention(hidden, rope, use_cache=False)), 2)

    def test_imported_cache_grids_need_no_range_reconstruction(self):
        attention = self._prefill_calibrated()
        for observer in (attention.obs_new_k, attention.obs_new_v):
            observer.reset()
            observer.load_qparams(torch.tensor(0.125), torch.tensor(113))
        attention.require_calibrated_cache()
        hidden = torch.randn(1, 1, self.hidden_size)
        rope = self._rand_rope(1, 1)
        outputs = attention(hidden, rope, use_cache=True)
        self.assertEqual(outputs[2][0].shape, (1, self.num_kv_heads, 1, self.head_dim))
