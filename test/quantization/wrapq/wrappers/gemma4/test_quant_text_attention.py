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

"""Unit tests for Gemma4 text attention PTQ wrapper."""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    LayerKV,
    QuantGemma4TextAttention,
)
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear


_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextAttention,
        )
    except Exception:
        return False
    return True


def _make_text_config(**overrides):
    """Create a tiny dense Gemma4 text config for synthetic attention tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    kwargs = dict(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        sliding_window=8,
        layer_types=["full_attention"],
        hidden_size_per_layer_input=0,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        enable_moe_block=False,
    )
    kwargs.update(overrides)
    cfg = Gemma4TextConfig(**kwargs)
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4TextAttention(unittest.TestCase):
    """Validate dense Gemma4 text attention wrapper behavior."""

    def setUp(self):
        """Create deterministic test inputs."""
        torch.manual_seed(1234)

    @staticmethod
    def _rope(batch_size: int, seq_len: int, head_dim: int):
        """Create synthetic RoPE tables shaped like Gemma4 text RoPE output."""
        emb = torch.randn(batch_size, seq_len, head_dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def _zero_mask(batch_size: int, q_len: int, k_len: int) -> torch.Tensor:
        """Return an additive attention mask that keeps every key."""
        return torch.zeros(batch_size, 1, q_len, k_len)

    def _make_attention(self, cfg=None, layer_idx: int = 0):
        """Create a floating-point Gemma4 text attention module."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        cfg = cfg if cfg is not None else _make_text_config()
        return Gemma4TextAttention(cfg, layer_idx=layer_idx).eval()

    def test_no_quant_forward_matches_hf_eager_attention(self):
        """Check that the wrapper matches Hugging Face eager attention in NO_QUANT."""
        cfg = _make_text_config()
        fp_attn = self._make_attention(cfg)
        qattn = QuantGemma4TextAttention(fp_attn).eval()

        batch_size, seq_len = 2, 5
        hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
        rope = self._rope(batch_size, seq_len, cfg.head_dim)
        mask = self._zero_mask(batch_size, seq_len, seq_len)

        with torch.no_grad():
            quant_out, quant_weights = qattn(
                hidden,
                rope,
                attention_mask=mask,
                shared_kv_states={},
            )
            fp_out, fp_weights = fp_attn(
                hidden,
                position_embeddings=rope,
                attention_mask=mask,
                shared_kv_states={},
            )

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertEqual(quant_weights.shape, fp_weights.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(quant_weights, fp_weights, atol=1e-5, rtol=1e-5))

    def test_mode_transitions_and_projection_override(self):
        """Check lifecycle transitions and a child projection quant override."""
        from tico.quantization.wrapq.dtypes import DType

        cfg = PTQConfig(
            overrides={
                "q_proj": {
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                }
            }
        )
        qattn = QuantGemma4TextAttention(self._make_attention(), qcfg=cfg)

        self.assertIs(qattn._mode, Mode.NO_QUANT)
        qattn.enable_calibration()
        self.assertIs(qattn._mode, Mode.CALIB)

        hidden = torch.randn(1, 4, qattn.config.hidden_size)
        rope = self._rope(1, 4, qattn.head_dim)
        qattn(hidden, rope, attention_mask=self._zero_mask(1, 4, 4))
        qattn.freeze_qparams()

        self.assertIs(qattn._mode, Mode.QUANT)
        self.assertIsInstance(qattn.q_proj.wrapped, QuantLinear)
        self.assertEqual(qattn.q_proj.wrapped.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(qattn.q_proj.wrapped.obs_act_out.dtype, DType.uint(4))

    def test_cache_delta_prefill_then_decode(self):
        """Validate explicit tuple cache handling for prefill and decode."""
        cfg = _make_text_config()
        qattn = QuantGemma4TextAttention(self._make_attention(cfg)).eval()

        batch_size, prefill_len = 1, 4
        hidden = torch.randn(batch_size, prefill_len, cfg.hidden_size)
        rope = self._rope(batch_size, prefill_len, cfg.head_dim)
        mask = self._zero_mask(batch_size, prefill_len, prefill_len)

        with torch.no_grad():
            out0, weights0, kv0 = qattn(
                hidden,
                rope,
                attention_mask=mask,
                use_cache=True,
                cache_output_mode="delta",
            )

        self.assertEqual(out0.shape, (batch_size, prefill_len, cfg.hidden_size))
        self.assertEqual(
            weights0.shape,
            (batch_size, cfg.num_attention_heads, prefill_len, prefill_len),
        )
        self.assertIsInstance(kv0, tuple)
        key0, value0 = kv0
        self.assertEqual(
            key0.shape, (batch_size, cfg.num_key_value_heads, prefill_len, cfg.head_dim)
        )
        self.assertEqual(value0.shape, key0.shape)

        decode_len = 1
        hidden1 = torch.randn(batch_size, decode_len, cfg.hidden_size)
        rope1 = self._rope(batch_size, decode_len, cfg.head_dim)
        mask1 = self._zero_mask(batch_size, decode_len, prefill_len + decode_len)
        with torch.no_grad():
            out1, weights1, kv1 = qattn(
                hidden1,
                rope1,
                attention_mask=mask1,
                past_key_value=kv0,
                use_cache=True,
                cache_output_mode="delta",
            )

        self.assertEqual(out1.shape, (batch_size, decode_len, cfg.hidden_size))
        self.assertEqual(
            weights1.shape,
            (batch_size, cfg.num_attention_heads, decode_len, prefill_len + decode_len),
        )
        key1, value1 = kv1
        self.assertEqual(
            key1.shape, (batch_size, cfg.num_key_value_heads, decode_len, cfg.head_dim)
        )
        self.assertEqual(value1.shape, key1.shape)

    def test_alternative_attention_k_equals_v_matches_hf(self):
        """Check the Gemma4 full-attention path where V reuses raw K states."""
        cfg = _make_text_config(attention_k_eq_v=True)
        fp_attn = self._make_attention(cfg)
        qattn = QuantGemma4TextAttention(fp_attn).eval()

        self.assertTrue(qattn.use_alternative_attention)
        self.assertIsNone(qattn.v_proj)

        batch_size, seq_len = 1, 5
        hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
        rope = self._rope(batch_size, seq_len, cfg.head_dim)
        mask = self._zero_mask(batch_size, seq_len, seq_len)

        with torch.no_grad():
            quant_out, _ = qattn(
                hidden,
                rope,
                attention_mask=mask,
                shared_kv_states={},
            )
            fp_out, _ = fp_attn(
                hidden,
                position_embeddings=rope,
                attention_mask=mask,
                shared_kv_states={},
            )

        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_shared_kv_layer_consumes_stored_states(self):
        """Validate Gemma4 shared-KV producer and consumer layer behavior."""
        cfg = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            num_kv_shared_layers=1,
        )
        fp_attn0 = self._make_attention(cfg, layer_idx=0)
        fp_attn1 = self._make_attention(cfg, layer_idx=1)
        qattn0 = QuantGemma4TextAttention(fp_attn0).eval()
        qattn1 = QuantGemma4TextAttention(fp_attn1).eval()

        self.assertTrue(qattn0.store_full_length_kv)
        self.assertTrue(qattn1.is_kv_shared_layer)

        batch_size, seq_len = 1, 4
        hidden = torch.randn(batch_size, seq_len, cfg.hidden_size)
        rope = self._rope(batch_size, seq_len, cfg.head_dim)
        mask = self._zero_mask(batch_size, seq_len, seq_len)

        fp_shared: dict[str, LayerKV] = {}
        q_shared: dict[str, LayerKV] = {}
        with torch.no_grad():
            fp_attn0(
                hidden,
                position_embeddings=rope,
                attention_mask=mask,
                shared_kv_states=fp_shared,
            )
            fp_out, fp_weights = fp_attn1(
                hidden,
                position_embeddings=rope,
                attention_mask=mask,
                shared_kv_states=fp_shared,
            )

            qattn0(
                hidden,
                rope,
                attention_mask=mask,
                shared_kv_states=q_shared,
            )
            quant_out, quant_weights = qattn1(
                hidden,
                rope,
                attention_mask=mask,
                shared_kv_states=q_shared,
            )

        self.assertIn("full_attention", q_shared)
        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertEqual(quant_weights.shape, fp_weights.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_bool_mask_is_combined_with_causal_mask(self):
        """Check keep-mask conversion and causal-mask combination."""
        qcfg = PTQConfig(attention_mask_fill_value=-100.0)
        qattn = QuantGemma4TextAttention(self._make_attention(), qcfg=qcfg)

        keep_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        out = qattn._build_attention_mask(
            attention_mask=keep_mask,
            q_len=4,
            k_len=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(out.shape), (1, 1, 4, 4))
        self.assertEqual(out[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(out[0, 0, 0, 1].item(), -100.0)
        self.assertEqual(out[0, 0, 0, 2].item(), -100.0)
        self.assertTrue(torch.all(out[..., 3] == -100.0))
        self.assertEqual(out[0, 0, 2, 0].item(), 0.0)
        self.assertEqual(out[0, 0, 2, 1].item(), 0.0)
        self.assertEqual(out[0, 0, 2, 2].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
