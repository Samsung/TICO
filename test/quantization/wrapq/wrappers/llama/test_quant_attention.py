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
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaAttentionDecodeExportAdapter,
    LlamaAttentionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.llama.quant_attention import QuantLlamaAttention
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear


skip_msg = "required transformers not installed — skipping LlamaAttention tests"


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestQuantLlamaAttention(unittest.TestCase):
    fp_attn: torch.nn.Module
    head_dim: int
    n_kv: int
    n_h: int
    max_seq: int
    hidden_size: int

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
        cls.head_dim = cfg.head_dim
        cls.n_kv = cfg.num_key_value_heads
        cls.n_h = cfg.num_attention_heads
        cls.hidden_size = cfg.hidden_size

    def _make_qattn(self, qcfg: PTQConfig | None = None) -> QuantLlamaAttention:
        qattn = QuantLlamaAttention(self.fp_attn, qcfg=qcfg, layer_idx=0)
        qattn.enable_calibration()

        for _ in range(3):
            x = torch.randn(2, 4, self.hidden_size)
            pos = self._rand_rope_for(qattn, 2, 4)
            _ = qattn(x, pos)

        qattn.freeze_qparams()
        return qattn

    def _rand_rope(
        self,
        batch_size: int,
        seq_len: int,
        *,
        pre_negated_sin: bool = False,
    ):
        emb = torch.randn(batch_size, seq_len, self.head_dim)
        cos = emb.cos()
        sin = emb.sin()
        if pre_negated_sin:
            half_dim = self.head_dim // 2
            sin = sin.clone()
            sin[..., :half_dim] = -sin[..., :half_dim]
        return cos, sin

    def _rand_rope_for(
        self,
        qattn: QuantLlamaAttention,
        batch_size: int,
        seq_len: int,
    ):
        return self._rand_rope(
            batch_size,
            seq_len,
            pre_negated_sin=qattn.attn_options.rope == "pre_negated_sin",
        )

    def _rand_additive_mask(self, batch_size: int, q_len: int, k_len: int):
        mask = torch.zeros(batch_size, q_len, k_len, dtype=torch.float32)
        if k_len > 1:
            valid_len = torch.randint(low=1, high=k_len + 1, size=(1,)).item()
            if valid_len < k_len:
                mask[:, :, valid_len:] = float("-120")
        return mask

    def _rand_bool_mask(self, batch_size: int, q_len: int, k_len: int):
        mask = torch.ones(batch_size, q_len, k_len, dtype=torch.bool)
        if k_len > 1:
            valid_len = torch.randint(low=1, high=k_len + 1, size=(1,)).item()
            if valid_len < k_len:
                mask[:, :, valid_len:] = False
        return mask

    def _rand_past(self, batch_size: int, past_len: int):
        past_k = torch.randn(batch_size, self.n_kv, past_len, self.head_dim)
        past_v = torch.randn(batch_size, self.n_kv, past_len, self.head_dim)
        return past_k, past_v

    def test_default_profile_preserves_npu_export_options(self):
        qattn = QuantLlamaAttention(self.fp_attn)

        self.assertEqual(qattn.attn_options.scale_fusion, "k_proj")
        self.assertEqual(qattn.attn_options.rope, "pre_negated_sin")
        self.assertEqual(qattn.attn_options.layout, "unrolled")

    def test_reference_eval_profile_selects_batched_hf_like_path(self):
        qcfg = PTQConfig(model_args={"profile": "reference_eval"})
        qattn = QuantLlamaAttention(self.fp_attn, qcfg=qcfg)

        self.assertEqual(qattn.attn_options.scale_fusion, "none")
        self.assertEqual(qattn.attn_options.rope, "hf")
        self.assertEqual(qattn.attn_options.layout, "batched")

    def test_projection_scale_fusion_is_profile_controlled(self):
        scale = float(getattr(self.fp_attn, "scaling", self.head_dim**-0.5))

        qattn_npu = QuantLlamaAttention(self.fp_attn)
        torch.testing.assert_close(
            qattn_npu.q_proj.wrapped.module.weight,
            self.fp_attn.q_proj.weight,
        )
        torch.testing.assert_close(
            qattn_npu.k_proj.wrapped.module.weight,
            self.fp_attn.k_proj.weight * scale,
        )

        qattn_ref = QuantLlamaAttention(
            self.fp_attn,
            qcfg=PTQConfig(model_args={"profile": "reference_eval"}),
        )
        torch.testing.assert_close(
            qattn_ref.q_proj.wrapped.module.weight,
            self.fp_attn.q_proj.weight,
        )
        torch.testing.assert_close(
            qattn_ref.k_proj.wrapped.module.weight,
            self.fp_attn.k_proj.weight,
        )

        qattn_q_fused = QuantLlamaAttention(
            self.fp_attn,
            qcfg=PTQConfig(
                model_args={
                    "attention": {
                        "scale_fusion": "q_proj",
                        "rope": "hf",
                        "layout": "batched",
                    }
                }
            ),
        )
        torch.testing.assert_close(
            qattn_q_fused.q_proj.wrapped.module.weight,
            self.fp_attn.q_proj.weight * scale,
        )
        torch.testing.assert_close(
            qattn_q_fused.k_proj.wrapped.module.weight,
            self.fp_attn.k_proj.weight,
        )

    def test_reference_eval_and_npu_export_profiles_are_float_equivalent(self):
        torch.manual_seed(11)

        qattn_ref = QuantLlamaAttention(
            self.fp_attn,
            qcfg=PTQConfig(model_args={"profile": "reference_eval"}),
        )
        qattn_npu = QuantLlamaAttention(
            self.fp_attn,
            qcfg=PTQConfig(model_args={"profile": "npu_export"}),
        )

        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, self.hidden_size)
        cos, sin = self._rand_rope(batch_size, seq_len)
        _, pre_negated_sin = self._rand_rope(
            batch_size,
            seq_len,
            pre_negated_sin=True,
        )
        # Keep the same sine magnitudes for both profiles and only switch the
        # convention-dependent sign.
        pre_negated_sin = sin.clone()
        pre_negated_sin[..., : self.head_dim // 2] = -pre_negated_sin[
            ..., : self.head_dim // 2
        ]
        mask = torch.zeros(batch_size, seq_len, seq_len)

        with torch.no_grad():
            ref_out, ref_attn = qattn_ref(
                x,
                (cos, sin),
                attention_mask=mask,
            )
            npu_out, npu_attn = qattn_npu(
                x,
                (cos, pre_negated_sin),
                attention_mask=mask,
            )

        torch.testing.assert_close(ref_out, npu_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ref_attn, npu_attn, rtol=1e-5, atol=1e-5)

    def test_as_export_module_can_require_npu_profile(self):
        qattn_npu = QuantLlamaAttention(self.fp_attn)
        adapter = qattn_npu.as_export_module("prefill", require_npu_profile=True)
        self.assertIsInstance(adapter, LlamaAttentionPrefillExportAdapter)

        qattn_ref = QuantLlamaAttention(
            self.fp_attn,
            qcfg=PTQConfig(model_args={"profile": "reference_eval"}),
        )
        with self.assertRaises(ValueError):
            qattn_ref.as_export_module("prefill", require_npu_profile=True)

    def test_mode_transitions_prefill(self):
        qattn = QuantLlamaAttention(self.fp_attn)
        self.assertIs(qattn._mode, Mode.NO_QUANT)

        qattn.enable_calibration()
        self.assertIs(qattn._mode, Mode.CALIB)

        x = torch.randn(2, 5, self.hidden_size)
        pos = self._rand_rope_for(qattn, 2, 5)
        _ = qattn(x, pos)

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

    def test_mode_transitions_decode(self):
        qattn = QuantLlamaAttention(self.fp_attn, layer_idx=0)
        self.assertIs(qattn._mode, Mode.NO_QUANT)

        qattn.enable_calibration()
        self.assertIs(qattn._mode, Mode.CALIB)

        batch_size = 1
        x = torch.randn(batch_size, 1, self.hidden_size)
        pos = self._rand_rope_for(qattn, batch_size, 1)
        past = self._rand_past(batch_size, self.max_seq - 1)
        mask = self._rand_additive_mask(batch_size, 1, self.max_seq)

        _ = qattn(
            hidden_states=x,
            position_embeddings=pos,
            attention_mask=mask,
            past_key_value=past,
            use_cache=True,
        )

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

    def test_forward_diff_prefill(self):
        qcfg = PTQConfig(model_args={"profile": "reference_eval"})
        qattn = QuantLlamaAttention(self.fp_attn, qcfg=qcfg)
        qattn.enable_calibration()
        for _ in range(4):
            inp = torch.randn(2, 6, self.hidden_size)
            pos = self._rand_rope_for(qattn, 2, 6)
            mask = torch.zeros(2, 6, 6)
            _ = qattn(inp, pos, attention_mask=mask)
        qattn.freeze_qparams()

        x = torch.randn(2, 6, self.hidden_size)
        pos = self._rand_rope_for(qattn, 2, 6)
        mask = torch.zeros(2, 6, 6)
        with torch.no_grad():
            q_out, _ = qattn(x, pos, attention_mask=mask)
            fp_out = self.fp_attn(
                x,
                position_embeddings=pos,
                attention_mask=mask.unsqueeze(1),
            )[0]

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_forward_with_float_attention_mask_prefill(self):
        torch.manual_seed(123)

        qcfg = PTQConfig(model_args={"profile": "reference_eval"})
        qattn = QuantLlamaAttention(self.fp_attn, qcfg=qcfg)
        batch_size, seq_len = 2, 4
        float_mask = torch.zeros(batch_size, seq_len, seq_len)

        qattn.enable_calibration()
        for _ in range(2):
            x = torch.randn(batch_size, seq_len, self.hidden_size)
            pos = self._rand_rope_for(qattn, batch_size, seq_len)
            _ = qattn(x, pos, attention_mask=float_mask)
        qattn.freeze_qparams()

        x = torch.randn(batch_size, seq_len, self.hidden_size)
        pos = self._rand_rope_for(qattn, batch_size, seq_len)
        with torch.no_grad():
            q_out, attn_w = qattn(x, pos, attention_mask=float_mask)
            fp_out = self.fp_attn(
                x,
                position_embeddings=pos,
                attention_mask=float_mask.unsqueeze(1),
            )[0]

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(q_out.shape, (batch_size, seq_len, self.hidden_size))
        self.assertEqual(attn_w.shape, (batch_size, self.n_h, seq_len, seq_len))

    def test_forward_with_bool_attention_mask_prefill(self):
        qattn = self._make_qattn()

        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len, self.hidden_size)
        pos = self._rand_rope_for(qattn, batch_size, seq_len)
        bool_mask = self._rand_bool_mask(batch_size, seq_len, seq_len)

        with torch.no_grad():
            out, attn_w = qattn(x, pos, attention_mask=bool_mask)

        self.assertEqual(out.shape, (batch_size, seq_len, self.hidden_size))
        self.assertEqual(attn_w.shape, (batch_size, self.n_h, seq_len, seq_len))

    def test_cache_tuple_concat_prefill_and_present_decode(self):
        torch.manual_seed(0)

        qattn = self._make_qattn()

        batch_size = 2
        seq_prefill = 4
        x0 = torch.randn(batch_size, seq_prefill, self.hidden_size)
        pos0 = self._rand_rope_for(qattn, batch_size, seq_prefill)

        with torch.no_grad():
            out0, attn_w0, present0 = qattn(
                x0,
                pos0,
                attention_mask=None,
                past_key_value=None,
                use_cache=True,
            )

        self.assertEqual(out0.shape, (batch_size, seq_prefill, self.hidden_size))
        self.assertEqual(
            attn_w0.shape, (batch_size, self.n_h, seq_prefill, seq_prefill)
        )
        k0, v0 = present0
        self.assertEqual(k0.shape, (batch_size, self.n_kv, seq_prefill, self.head_dim))
        self.assertEqual(v0.shape, (batch_size, self.n_kv, seq_prefill, self.head_dim))

        seq_decode = 1
        x1 = torch.randn(batch_size, seq_decode, self.hidden_size)
        pos1 = self._rand_rope_for(qattn, batch_size, seq_decode)
        mask1 = self._rand_additive_mask(
            batch_size, seq_decode, seq_prefill + seq_decode
        )

        with torch.no_grad():
            out1, attn_w1, present1 = qattn(
                x1,
                pos1,
                attention_mask=mask1,
                past_key_value=present0,
                use_cache=True,
            )

        self.assertEqual(out1.shape, (batch_size, seq_decode, self.hidden_size))
        self.assertEqual(
            attn_w1.shape, (batch_size, self.n_h, seq_decode, seq_prefill + seq_decode)
        )
        k1, v1 = present1
        self.assertEqual(
            k1.shape, (batch_size, self.n_kv, seq_prefill + seq_decode, self.head_dim)
        )
        self.assertEqual(
            v1.shape, (batch_size, self.n_kv, seq_prefill + seq_decode, self.head_dim)
        )
        # Decode with a legacy tuple past may requantize the cached prefix through
        # obs_past_key/obs_past_value and obs_present_key/obs_present_value before
        # returning the full present cache, so exact prefix equality is not guaranteed.
        self.assertEqual(k1[:, :, :seq_prefill, :].shape, k0.shape)
        self.assertEqual(v1[:, :, :seq_prefill, :].shape, v0.shape)

    def test_forward_shapes_and_cache_delta_decode(self):
        torch.manual_seed(1)

        qattn = self._make_qattn()

        batch_size = 1
        x = torch.randn(batch_size, 1, self.hidden_size)
        pos = self._rand_rope_for(qattn, batch_size, 1)
        mask = self._rand_additive_mask(batch_size, 1, self.max_seq)
        past_k, past_v = self._rand_past(batch_size, self.max_seq - 1)

        with torch.no_grad():
            out, _, new_kv = qattn(
                hidden_states=x,
                position_embeddings=pos,
                attention_mask=mask,
                past_key_value=(past_k, past_v),
                use_cache=True,
                cache_output_mode="delta",
            )

        self.assertEqual(out.shape, (batch_size, 1, self.hidden_size))
        new_k, new_v = new_kv
        self.assertEqual(new_k.shape, (batch_size, self.n_kv, 1, self.head_dim))
        self.assertEqual(new_v.shape, (batch_size, self.n_kv, 1, self.head_dim))

    def test_invalid_cache_output_mode_raises(self):
        qattn = self._make_qattn()

        x = torch.randn(1, 1, self.hidden_size)
        pos = self._rand_rope_for(qattn, 1, 1)

        with self.assertRaises(ValueError):
            _ = qattn(
                x,
                pos,
                use_cache=True,
                cache_output_mode="invalid",
            )

    def test_prefill_export_adapter_returns_delta_kv(self):
        qattn = self._make_qattn()
        adapter = LlamaAttentionPrefillExportAdapter(qattn, return_kv=True)

        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len, self.hidden_size)
        pos = self._rand_rope_for(qattn, batch_size, seq_len)

        with torch.no_grad():
            hidden, new_k, new_v = adapter(
                hidden_states=x,
                position_embeddings=pos,
                attention_mask=None,
            )

        self.assertEqual(hidden.shape, (batch_size, seq_len, self.hidden_size))
        self.assertEqual(new_k.shape, (batch_size, self.n_kv, seq_len, self.head_dim))
        self.assertEqual(new_v.shape, (batch_size, self.n_kv, seq_len, self.head_dim))

    def test_decode_export_adapter_returns_delta_kv(self):
        qattn = self._make_qattn()
        adapter = LlamaAttentionDecodeExportAdapter(qattn, return_kv=True)

        batch_size = 1
        q_len = 1
        past_len = self.max_seq - 1
        x = torch.randn(batch_size, q_len, self.hidden_size)
        pos = self._rand_rope_for(qattn, batch_size, q_len)
        past = self._rand_past(batch_size, past_len)
        mask = self._rand_additive_mask(batch_size, q_len, past_len + q_len)

        with torch.no_grad():
            hidden, new_k, new_v = adapter(
                hidden_states=x,
                position_embeddings=pos,
                attention_mask=mask,
                past_key_value=past,
            )

        self.assertEqual(hidden.shape, (batch_size, q_len, self.hidden_size))
        self.assertEqual(new_k.shape, (batch_size, self.n_kv, q_len, self.head_dim))
        self.assertEqual(new_v.shape, (batch_size, self.n_kv, q_len, self.head_dim))

    def test_export_adapters_without_cache_return_hidden_only(self):
        qattn = self._make_qattn()

        prefill_adapter = LlamaAttentionPrefillExportAdapter(qattn, return_kv=False)
        decode_adapter = LlamaAttentionDecodeExportAdapter(qattn, return_kv=False)

        batch_size = 1
        q_len = 1
        x = torch.randn(batch_size, q_len, self.hidden_size)
        pos = self._rand_rope_for(qattn, batch_size, q_len)
        past = self._rand_past(batch_size, self.max_seq - 1)
        mask = self._rand_additive_mask(batch_size, q_len, self.max_seq)

        with torch.no_grad():
            prefill_hidden = prefill_adapter(
                hidden_states=x,
                position_embeddings=pos,
                attention_mask=None,
            )
            decode_hidden = decode_adapter(
                hidden_states=x,
                position_embeddings=pos,
                attention_mask=mask,
                past_key_value=past,
            )

        self.assertEqual(prefill_hidden.shape, (batch_size, q_len, self.hidden_size))
        self.assertEqual(decode_hidden.shape, (batch_size, q_len, self.hidden_size))

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
        q_lin = qattn.q_proj.wrapped

        self.assertIsInstance(q_lin, QuantLinear)
        self.assertEqual(q_lin.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_lin.obs_act_out.dtype, DType.uint(4))

    def test_forward_diff_vs_self_consistency_decode(self):
        torch.manual_seed(7)

        qattn = QuantLlamaAttention(self.fp_attn, layer_idx=0)
        qattn.enable_calibration()

        for _ in range(4):
            x = torch.randn(1, 1, self.hidden_size)
            pos = self._rand_rope_for(qattn, 1, 1)
            mask = self._rand_additive_mask(1, 1, self.max_seq)
            past = self._rand_past(1, self.max_seq - 1)
            _ = qattn(x, pos, mask, past, use_cache=True)

        x = torch.randn(1, 1, self.hidden_size)
        pos = self._rand_rope_for(qattn, 1, 1)
        mask = self._rand_additive_mask(1, 1, self.max_seq)
        past = self._rand_past(1, self.max_seq - 1)

        with torch.no_grad():
            cal_out, _, _ = qattn(x, pos, mask, past, use_cache=True)

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

        with torch.no_grad():
            q_out, _, _ = qattn(x, pos, mask, past, use_cache=True)

        diff = (cal_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1.0)
        self.assertEqual(cal_out.shape, q_out.shape)


if __name__ == "__main__":
    unittest.main()
