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
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer_decode import (
    QuantLlamaDecoderLayerDecode,
)

skip_msg = (
    "required transformers not installed — skipping LlamaDecoderLayerDecode tests"
)


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestQuantLlamaDecoderLayerDecode(unittest.TestCase):
    fp_layer: torch.nn.Module
    cfg: object
    max_seq: int
    head_dim: int
    n_kv: int
    D: int

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        cls.max_seq = 32
        cls.cfg = LlamaConfig(  # type: ignore[attr-defined]
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
        cls.fp_layer = LlamaDecoderLayer(cls.cfg, layer_idx=0)  # type: ignore[attr-defined]
        cls.head_dim = cls.cfg.head_dim  # type: ignore[attr-defined]
        cls.n_kv = cls.cfg.num_key_value_heads  # type: ignore[attr-defined]
        cls.D = cls.cfg.hidden_size  # type: ignore[attr-defined]

    def _rand_rope_current(self, B: int):
        """Return dummy (cos, sin) for the current token only: each (B,1,head_dim)."""
        emb = torch.randn(B, 1, self.head_dim)
        return emb.cos(), emb.sin()

    def _rand_additive_mask(self, B: int):
        """Return additive mask (B,1,max_seq): 0 keep, -120 masked."""
        L_eff = torch.randint(low=1, high=self.max_seq + 1, size=(1,)).item()
        mask = torch.zeros(B, 1, self.max_seq, dtype=torch.float32)
        if L_eff < self.max_seq:
            mask[:, :, L_eff:] = float("-120")
        return mask

    def _rand_past(self, B: int):
        """Static past KV: (B,n_kv,max_seq-1,head_dim)."""
        past_k = torch.randn(B, self.n_kv, self.max_seq - 1, self.head_dim)
        past_v = torch.randn(B, self.n_kv, self.max_seq - 1, self.head_dim)
        return past_k, past_v

    def test_mode_transitions(self):
        qlayer = QuantLlamaDecoderLayerDecode(self.fp_layer)
        self.assertIs(qlayer._mode, Mode.NO_QUANT)

        qlayer.enable_calibration()
        self.assertIs(qlayer._mode, Mode.CALIB)

        B = 1
        x = torch.randn(B, 1, self.D)
        pos = self._rand_rope_current(B)
        mask = self._rand_additive_mask(B)
        past = self._rand_past(B)

        _ = qlayer(
            hidden_states=x,
            attention_mask=mask,
            past_key_value=past,
            position_embeddings=pos,
            use_cache=True,
        )

        qlayer.freeze_qparams()
        self.assertIs(qlayer._mode, Mode.QUANT)

    def test_forward_shapes_and_cache_delta(self):
        torch.manual_seed(1)

        qlayer = QuantLlamaDecoderLayerDecode(self.fp_layer)
        qlayer.enable_calibration()
        for _ in range(3):
            x = torch.randn(1, 1, self.D)
            pos = self._rand_rope_current(1)
            mask = self._rand_additive_mask(1)
            past = self._rand_past(1)
            _ = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )
        qlayer.freeze_qparams()

        B = 1
        x = torch.randn(B, 1, self.D)
        pos = self._rand_rope_current(B)
        mask = self._rand_additive_mask(B)
        past_k, past_v = self._rand_past(B)

        with torch.no_grad():
            out = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=(past_k, past_v),
                position_embeddings=pos,
                use_cache=True,
            )

        self.assertIsInstance(out, tuple)
        hidden_out, present = out
        self.assertEqual(hidden_out.shape, (B, 1, self.D))

        # Decode layer returns KV delta from attention wrapper.
        new_k, new_v = present
        self.assertEqual(new_k.shape, (B, self.n_kv, 1, self.head_dim))
        self.assertEqual(new_v.shape, (B, self.n_kv, 1, self.head_dim))

    def test_requires_additive_mask(self):
        """Decode layer must reject missing masks (no dynamic mask creation)."""
        qlayer = QuantLlamaDecoderLayerDecode(self.fp_layer)

        B = 1
        x = torch.randn(B, 1, self.D)
        pos = self._rand_rope_current(B)
        past = self._rand_past(B)

        with self.assertRaises(AssertionError):
            _ = qlayer(
                hidden_states=x,
                attention_mask=None,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

    def test_rejects_bool_mask(self):
        """Decode attention rejects boolean masks; layer should propagate that failure."""
        qlayer = QuantLlamaDecoderLayerDecode(self.fp_layer)

        B = 1
        x = torch.randn(B, 1, self.D)
        pos = self._rand_rope_current(B)
        past = self._rand_past(B)

        bool_mask = torch.zeros(B, 1, self.max_seq, dtype=torch.bool)
        with self.assertRaises(AssertionError):
            _ = qlayer(
                hidden_states=x,
                attention_mask=bool_mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

    def test_static_shape_asserts(self):
        """Decode layer must enforce strict static shapes for all required inputs."""
        qlayer = QuantLlamaDecoderLayerDecode(self.fp_layer)

        B = 1

        # hidden_states must be (B,1,D)
        x_bad = torch.randn(B, 2, self.D)
        pos = self._rand_rope_current(B)
        mask = self._rand_additive_mask(B)
        past = self._rand_past(B)
        with self.assertRaises(AssertionError):
            _ = qlayer(
                hidden_states=x_bad,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        # attention_mask must be (B,1,max_seq)
        x = torch.randn(B, 1, self.D)
        mask_bad = torch.zeros(B, 1, self.max_seq - 1, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            _ = qlayer(
                hidden_states=x,
                attention_mask=mask_bad,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        # past must be (B,n_kv,max_seq-1,head_dim)
        past_k, past_v = past
        past_bad = (past_k[:, :, :-1, :], past_v)
        with self.assertRaises(AssertionError):
            _ = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past_bad,
                position_embeddings=pos,
                use_cache=True,
            )

        # position embeddings must be (B,1,head_dim)
        pos_bad = (torch.randn(B, 2, self.head_dim), torch.randn(B, 2, self.head_dim))
        with self.assertRaises(AssertionError):
            _ = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos_bad,
                use_cache=True,
            )

    def test_dtype_override(self):
        """
        Verify that the local observer override works for the decode layer.
        Currently QuantLlamaDecoderLayerDecode creates `mlp_residual_out` locally.
        """
        cfg = PTQConfig(
            default_dtype=DType.int(16),
            wrapper_variant="decode",
            overrides={
                "mlp_residual_out": {"dtype": DType.uint(8)},
            },
        )
        qcustom = QuantLlamaDecoderLayerDecode(self.fp_layer, qcfg=cfg)
        self.assertEqual(qcustom.obs_mlp_residual_out.dtype, DType.uint(8))

    def test_calib_vs_quant_diff_sanity(self):
        """
        Similar to the decode-attn test: compare CALIB output vs QUANT output
        to ensure quantization perturbs outputs but remains numerically stable.
        """
        torch.manual_seed(7)

        qlayer = QuantLlamaDecoderLayerDecode(self.fp_layer)
        qlayer.enable_calibration()

        for _ in range(4):
            x = torch.randn(1, 1, self.D)
            pos = self._rand_rope_current(1)
            mask = self._rand_additive_mask(1)
            past = self._rand_past(1)
            _ = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        x = torch.randn(1, 1, self.D)
        pos = self._rand_rope_current(1)
        mask = self._rand_additive_mask(1)
        past = self._rand_past(1)

        with torch.no_grad():
            cal_hidden, (cal_new_k, cal_new_v) = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        qlayer.freeze_qparams()
        self.assertIs(qlayer._mode, Mode.QUANT)

        with torch.no_grad():
            q_hidden, (q_new_k, q_new_v) = qlayer(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past,
                position_embeddings=pos,
                use_cache=True,
            )

        # hidden output diff
        diff_h = (cal_hidden - q_hidden).abs().mean().item()
        self.assertGreater(diff_h, 0.0)
        self.assertLess(diff_h, 2.0)
        self.assertEqual(cal_hidden.shape, q_hidden.shape)

        # kv delta diffs
        diff_k = (cal_new_k - q_new_k).abs().mean().item()
        diff_v = (cal_new_v - q_new_v).abs().mean().item()
        self.assertGreater(diff_k, 0.0)
        self.assertGreater(diff_v, 0.0)
        self.assertLess(diff_k, 2.0)
        self.assertLess(diff_v, 2.0)
        self.assertEqual(cal_new_k.shape, q_new_k.shape)
        self.assertEqual(cal_new_v.shape, q_new_v.shape)
