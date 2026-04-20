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
    LlamaDecoderLayerDecodeExportAdapter,
    LlamaDecoderLayerPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)


skip_msg = "required transformers not installed — skipping LlamaDecoderLayer tests"


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestQuantLlamaDecoderLayer(unittest.TestCase):
    fp_layer: torch.nn.Module
    cfg: object
    max_seq: int
    head_dim: int
    n_kv: int
    hidden_size: int

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        cls.max_seq = 32
        cls.cfg = LlamaConfig(
            hidden_size=16,
            max_position_embeddings=cls.max_seq,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
        )
        cls.fp_layer = LlamaDecoderLayer(cls.cfg, layer_idx=0)
        cls.head_dim = cls.cfg.head_dim  # type: ignore[attr-defined]
        cls.n_kv = cls.cfg.num_key_value_heads  # type: ignore[attr-defined]
        cls.hidden_size = cls.cfg.hidden_size  # type: ignore[attr-defined]

    def _rand_rope(self, batch_size: int, seq_len: int):
        emb = torch.randn(batch_size, seq_len, self.head_dim)
        return emb.cos(), emb.sin()

    def _rand_additive_mask(self, batch_size: int, q_len: int, k_len: int):
        mask = torch.zeros(batch_size, q_len, k_len, dtype=torch.float32)
        if k_len > 1:
            valid_len = torch.randint(low=1, high=k_len + 1, size=(1,)).item()
            if valid_len < k_len:
                mask[:, :, valid_len:] = float("-120")
        return mask

    def _rand_past(self, batch_size: int, past_len: int):
        past_k = torch.randn(batch_size, self.n_kv, past_len, self.head_dim)
        past_v = torch.randn(batch_size, self.n_kv, past_len, self.head_dim)
        return past_k, past_v

    def test_mode_transitions_prefill(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        self.assertIs(qlayer._mode, Mode.NO_QUANT)

        qlayer.enable_calibration()
        self.assertIs(qlayer._mode, Mode.CALIB)

        seq_len = self.max_seq
        hidden = torch.randn(2, seq_len, self.hidden_size)
        attn_mask = torch.ones(2, seq_len, seq_len, dtype=torch.bool)
        _ = qlayer(hidden, attention_mask=attn_mask)

        qlayer.freeze_qparams()
        self.assertIs(qlayer._mode, Mode.QUANT)

    def test_mode_transitions_decode(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        qlayer.return_type = "tuple"
        self.assertIs(qlayer._mode, Mode.NO_QUANT)

        qlayer.enable_calibration()
        self.assertIs(qlayer._mode, Mode.CALIB)

        batch_size = 1
        x = torch.randn(batch_size, 1, self.hidden_size)
        pos = self._rand_rope(batch_size, 1)
        past = self._rand_past(batch_size, self.max_seq - 1)
        mask = self._rand_additive_mask(batch_size, 1, self.max_seq)

        _ = qlayer(
            hidden_states=x,
            attention_mask=mask,
            past_key_value=past,
            position_embeddings=pos,
            use_cache=True,
        )

        qlayer.freeze_qparams()
        self.assertIs(qlayer._mode, Mode.QUANT)

    def test_forward_diff_prefill(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        qlayer.enable_calibration()
        seq_len = self.max_seq
        for _ in range(4):
            hidden = torch.randn(2, seq_len, self.hidden_size)
            attn_mask = torch.ones(2, seq_len, seq_len, dtype=torch.bool)
            _ = qlayer(hidden, attention_mask=attn_mask)
        qlayer.freeze_qparams()

        hidden = torch.randn(2, seq_len, self.hidden_size)
        pos = self._rand_rope(2, seq_len)
        attn_mask = torch.ones(2, seq_len, seq_len, dtype=torch.bool)

        with torch.no_grad():
            q_out = qlayer(hidden, attention_mask=attn_mask)
            q_out = q_out[0] if isinstance(q_out, tuple) else q_out
            fp_out = self.fp_layer(
                hidden,
                attention_mask=attn_mask.unsqueeze(1),
                position_embeddings=pos,
            )
            fp_out = fp_out[0] if isinstance(fp_out, tuple) else fp_out

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.5)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_forward_shapes_and_full_present_cache_decode(self):
        torch.manual_seed(1)

        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        qlayer.return_type = "tuple"
        qlayer.enable_calibration()
        for _ in range(3):
            x = torch.randn(1, 1, self.hidden_size)
            pos = self._rand_rope(1, 1)
            mask = self._rand_additive_mask(1, 1, self.max_seq)
            past = self._rand_past(1, self.max_seq - 1)
            _ = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )
        qlayer.freeze_qparams()

        batch_size = 1
        x = torch.randn(batch_size, 1, self.hidden_size)
        pos = self._rand_rope(batch_size, 1)
        mask = self._rand_additive_mask(batch_size, 1, self.max_seq)
        past_k, past_v = self._rand_past(batch_size, self.max_seq - 1)

        with torch.no_grad():
            hidden_out, present = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=(past_k, past_v),
                position_embeddings=pos,
                use_cache=True,
            )

        self.assertEqual(hidden_out.shape, (batch_size, 1, self.hidden_size))
        present_k, present_v = present
        self.assertEqual(
            present_k.shape, (batch_size, self.n_kv, self.max_seq, self.head_dim)
        )
        self.assertEqual(
            present_v.shape, (batch_size, self.n_kv, self.max_seq, self.head_dim)
        )

    def test_none_attention_mask_is_supported_in_decode_path(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        qlayer.return_type = "tuple"

        batch_size = 1
        x = torch.randn(batch_size, 1, self.hidden_size)
        pos = self._rand_rope(batch_size, 1)
        past = self._rand_past(batch_size, self.max_seq - 1)

        with torch.no_grad():
            out = qlayer(
                hidden_states=x,
                attention_mask=None,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        self.assertIsInstance(out, tuple)
        self.assertEqual(out[0].shape, (batch_size, 1, self.hidden_size))
        present_k, present_v = out[1]
        self.assertEqual(
            present_k.shape, (batch_size, self.n_kv, self.max_seq, self.head_dim)
        )
        self.assertEqual(
            present_v.shape, (batch_size, self.n_kv, self.max_seq, self.head_dim)
        )

    def test_bool_attention_mask_is_supported_in_decode_path(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        qlayer.return_type = "tuple"

        batch_size = 1
        x = torch.randn(batch_size, 1, self.hidden_size)
        pos = self._rand_rope(batch_size, 1)
        past = self._rand_past(batch_size, self.max_seq - 1)
        bool_mask = torch.ones(batch_size, 1, self.max_seq, dtype=torch.bool)

        with torch.no_grad():
            out = qlayer(
                hidden_states=x,
                attention_mask=bool_mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        self.assertIsInstance(out, tuple)
        self.assertEqual(out[0].shape, (batch_size, 1, self.hidden_size))
        present_k, present_v = out[1]
        self.assertEqual(
            present_k.shape, (batch_size, self.n_kv, self.max_seq, self.head_dim)
        )
        self.assertEqual(
            present_v.shape, (batch_size, self.n_kv, self.max_seq, self.head_dim)
        )

    def test_dtype_override(self):
        cfg = PTQConfig(
            default_dtype=DType.int(16),
            overrides={
                "mlp_residual_out": {"dtype": DType.uint(8)},
            },
        )
        qcustom = QuantLlamaDecoderLayer(self.fp_layer, qcfg=cfg)
        self.assertEqual(qcustom.obs_mlp_residual_out.dtype, DType.uint(8))

    def test_calib_vs_quant_diff_sanity_decode(self):
        torch.manual_seed(7)

        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        qlayer.return_type = "tuple"
        qlayer.enable_calibration()

        for _ in range(4):
            x = torch.randn(1, 1, self.hidden_size)
            pos = self._rand_rope(1, 1)
            mask = self._rand_additive_mask(1, 1, self.max_seq)
            past = self._rand_past(1, self.max_seq - 1)
            _ = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        x = torch.randn(1, 1, self.hidden_size)
        pos = self._rand_rope(1, 1)
        mask = self._rand_additive_mask(1, 1, self.max_seq)
        past = self._rand_past(1, self.max_seq - 1)

        with torch.no_grad():
            cal_hidden, (cal_present_k, cal_present_v) = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        qlayer.freeze_qparams()
        self.assertIs(qlayer._mode, Mode.QUANT)

        with torch.no_grad():
            q_hidden, (q_present_k, q_present_v) = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        diff_h = (cal_hidden - q_hidden).abs().mean().item()
        self.assertGreater(diff_h, 0.0)
        self.assertLess(diff_h, 2.0)
        self.assertEqual(cal_hidden.shape, q_hidden.shape)

        diff_k = (cal_present_k - q_present_k).abs().mean().item()
        diff_v = (cal_present_v - q_present_v).abs().mean().item()
        self.assertGreater(diff_k, 0.0)
        self.assertGreater(diff_v, 0.0)
        self.assertLess(diff_k, 2.0)
        self.assertLess(diff_v, 2.0)
        self.assertEqual(cal_present_k.shape, q_present_k.shape)
        self.assertEqual(cal_present_v.shape, q_present_v.shape)

    def test_export_prefill_adapter_returns_delta_kv(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        adapter = qlayer.as_export_module(mode="prefill", return_kv=True)
        self.assertIsInstance(adapter, LlamaDecoderLayerPrefillExportAdapter)

        batch_size = 2
        seq_len = 4
        hidden = torch.randn(batch_size, seq_len, self.hidden_size)
        pos = self._rand_rope(batch_size, seq_len)
        mask = self._rand_additive_mask(batch_size, seq_len, seq_len)

        with torch.no_grad():
            hidden_out, new_k, new_v = adapter(
                hidden_states=hidden,
                attention_mask=mask,
                position_embeddings=pos,
            )

        self.assertEqual(hidden_out.shape, (batch_size, seq_len, self.hidden_size))
        self.assertEqual(new_k.shape, (batch_size, self.n_kv, seq_len, self.head_dim))
        self.assertEqual(new_v.shape, (batch_size, self.n_kv, seq_len, self.head_dim))

    def test_export_decode_adapter_returns_delta_kv(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        adapter = qlayer.as_export_module(mode="decode", return_kv=True)
        self.assertIsInstance(adapter, LlamaDecoderLayerDecodeExportAdapter)

        batch_size = 1
        x = torch.randn(batch_size, 1, self.hidden_size)
        pos = self._rand_rope(batch_size, 1)
        mask = self._rand_additive_mask(batch_size, 1, self.max_seq)
        past = self._rand_past(batch_size, self.max_seq - 1)

        with torch.no_grad():
            hidden_out, new_k, new_v = adapter(
                hidden_states=x,
                attention_mask=mask,
                position_embeddings=pos,
                past_key_value=past,
            )

        self.assertEqual(hidden_out.shape, (batch_size, 1, self.hidden_size))
        self.assertEqual(new_k.shape, (batch_size, self.n_kv, 1, self.head_dim))
        self.assertEqual(new_v.shape, (batch_size, self.n_kv, 1, self.head_dim))

    def test_export_adapter_without_kv_returns_hidden_only(self):
        qlayer = QuantLlamaDecoderLayer(self.fp_layer)
        adapter = qlayer.as_export_module(mode="decode", return_kv=False)

        batch_size = 1
        x = torch.randn(batch_size, 1, self.hidden_size)
        pos = self._rand_rope(batch_size, 1)
        mask = self._rand_additive_mask(batch_size, 1, self.max_seq)
        past = self._rand_past(batch_size, self.max_seq - 1)

        with torch.no_grad():
            hidden_out = adapter(
                hidden_states=x,
                attention_mask=mask,
                position_embeddings=pos,
                past_key_value=past,
            )

        self.assertEqual(hidden_out.shape, (batch_size, 1, self.hidden_size))


if __name__ == "__main__":
    unittest.main()
