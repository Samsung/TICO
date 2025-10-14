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

import unittest
from typing import Dict, Optional

import torch
import torch.nn as nn
from tico.experimental.quantization.config.ptq import PTQConfig

from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.wrappers.fairseq.quant_decoder import (
    QuantFairseqDecoder,
)
from tico.experimental.quantization.ptq.wrappers.fairseq.quant_decoder_layer import (
    QuantFairseqDecoderLayer,
)
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.ptq.wrappers.registry import register


# ────────────────────────────────────────────────────────────
#   minimal FP stubs (decoder + positional embedding + layer)
# ────────────────────────────────────────────────────────────
class _Cfg:
    def __init__(
        self,
        no_scale_embedding: bool = False,
        cross_self_attention: bool = False,
        max_target_positions: int = 2048,
    ):
        self.no_scale_embedding = no_scale_embedding
        self.cross_self_attention = cross_self_attention
        self.max_target_positions = max_target_positions


class DummyPositionalEmbedding(nn.Module):
    """Returns zeros of shape (B, T, C) and exposes max_positions."""

    def __init__(self, embed_dim: int, max_positions: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions

    def forward(
        self, tokens: torch.Tensor, incremental_state: Optional[Dict] = None
    ) -> torch.Tensor:
        B, T = tokens.shape
        return torch.zeros(
            B, T, self.embed_dim, dtype=torch.float32, device=tokens.device
        )


class _DummyMHA(nn.Module):
    """Fairseq-like MHA: carries q/k/v/out_proj linears and flags."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        self_attention: bool,
        enc_dec: bool,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attention = self_attention
        self.encoder_decoder_attention = enc_dec
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


class DummyFPDecoderLayer(nn.Module):
    """
    Lightweight FP layer the PTQ wrapper will replace.
    (The real computation is in QuantFairseqDecoderLayer after wrapping.)
    """

    def __init__(
        self, embed_dim: int, num_heads: int = 4, normalize_before: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalize_before = normalize_before

        # Attentions
        self.self_attn = _DummyMHA(
            embed_dim, num_heads, self_attention=True, enc_dec=False, bias=False
        )
        self.encoder_attn = _DummyMHA(
            embed_dim, num_heads, self_attention=False, enc_dec=True, bias=False
        )

        # FFN
        self.fc1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc2 = nn.Linear(embed_dim, embed_dim, bias=False)

        # LayerNorms
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.activation_fn = nn.GELU()

        # Optional features (left as None by default)
        self.attn_ln = None
        self.c_attn = None
        self.ffn_layernorm = None
        self.w_resid = None
        self.cross_self_attention = False
        self.need_attn = True  # mirrors fairseq default


# register the real quant layer as the wrapper for our dummy FP layer
register(DummyFPDecoderLayer)(QuantFairseqDecoderLayer)


class DummyFPDecoder(nn.Module):
    """
    Fairseq-like FP decoder used to build the QuantFairseqDecoder under test.
    Only the attributes referenced by the wrapper are implemented.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        padding_idx: int = 1,
        max_target_positions: int = 256,
        use_positions: bool = True,
        use_embed_ln: bool = True,
        use_final_ln: bool = True,
        no_scale_embedding: bool = False,
        share_input_output_embed: bool = False,
        version_val: float = 3.0,
    ):
        super().__init__()
        self.cfg = _Cfg(
            no_scale_embedding=no_scale_embedding,
            cross_self_attention=False,
            max_target_positions=max_target_positions,
        )
        self.share_input_output_embed = share_input_output_embed
        self.version = torch.tensor([version_val], dtype=torch.float32)

        self.padding_idx = padding_idx
        self.max_target_positions = max_target_positions

        # Embeddings and LNs
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim
        self.output_embed_dim = embed_dim
        self.layernorm_embedding = nn.LayerNorm(embed_dim) if use_embed_ln else None
        self.layer_norm = nn.LayerNorm(embed_dim) if use_final_ln else None
        self.embed_positions = (
            DummyPositionalEmbedding(embed_dim, max_target_positions)
            if use_positions
            else None
        )

        # Projections
        self.project_in_dim = None
        self.project_out_dim = None

        # Output projection (acts like tied softmax weight when desired)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)

        # Stack of FP layers that will be PTQ-wrapped
        self.layers = nn.ModuleList(
            [DummyFPDecoderLayer(embed_dim) for _ in range(num_layers)]
        )


# ────────────────────────────────────────────────────────────
#   test suite
# ────────────────────────────────────────────────────────────
class TestQuantFairseqDecoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.V = 100
        self.E = 32
        self.B = 4
        self.T = 7
        self.S = 5
        self.L = 3
        self.pad = 1

    def _build_decoder(self, **fp_kwargs):
        fp = DummyFPDecoder(
            vocab_size=self.V,
            embed_dim=self.E,
            num_layers=self.L,
            padding_idx=self.pad,
            **fp_kwargs,
        )
        dec = QuantFairseqDecoder(fp, qcfg=PTQConfig(), fp_name="dec0")
        return dec, fp

    def _mk_inputs(self, pad_last_col: bool = True):
        prev = torch.randint(low=2, high=self.V, size=(self.B, self.T))
        if pad_last_col:
            prev[:, -1] = self.pad
        enc_out = torch.randn(self.S, self.B, self.E)
        enc_kpm = torch.zeros(self.B, self.S, dtype=torch.bool)
        enc_dict = {"encoder_out": [enc_out], "encoder_padding_mask": [enc_kpm]}
        return prev, enc_dict

    # 1) forward() produces logits (B,T,V) and extra dict with attn/inner_states
    def test_forward_logits_and_extra(self):
        dec, _ = self._build_decoder()
        prev, enc = self._mk_inputs()
        logits, extra = dec(
            prev, encoder_out=enc, features_only=False, return_all_hiddens=True
        )
        self.assertEqual(logits.shape, (self.B, self.T, self.V))
        self.assertIn("attn", extra)
        self.assertIn("inner_states", extra)
        # inner_states length = L + 1 (initial)
        self.assertEqual(len(extra["inner_states"]), self.L + 1)
        # attn may be None if alignment layer didn't produce weights; still list with 1 entry
        self.assertEqual(len(extra["attn"]), 1)

    # 2) extract_features_scriptable returns features (B,T,E) and averaged attn (B,T,S)
    def test_extract_features_alignment(self):
        dec, _ = self._build_decoder()
        prev, enc = self._mk_inputs()
        # request alignment at layer L-1 and average all heads
        x, extra = dec.extract_features_scriptable(
            prev_output_tokens=prev,
            encoder_out=enc,
            full_context_alignment=False,
            alignment_layer=self.L - 1,
            alignment_heads=None,  # None → average all heads
        )
        self.assertEqual(x.shape, (self.B, self.T, self.E))
        self.assertIn("attn", extra)
        # if present, attn is (B,T,S)
        if extra["attn"][0] is not None:
            self.assertEqual(extra["attn"][0].shape, (self.B, self.T, self.S))

    # 3) incremental decoding: only last step is consumed; shapes remain (B,1,V)
    def test_incremental_decoding_two_steps(self):
        dec, _ = self._build_decoder()
        enc_prev, enc = self._mk_inputs()
        inc: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

        # step 1 (consume last token)
        prev1 = enc_prev[:, :1]  # start with length 1 for clarity
        logits1, extra1 = dec(
            prev1, encoder_out=enc, incremental_state=inc, features_only=False
        )
        self.assertEqual(logits1.shape, (self.B, 1, self.V))

        # step 2: feed ONLY the new token (length=1), cache holds history
        prev_tokens_step2 = torch.randint(2, self.V, (self.B, 1))
        logits2, _ = dec(
            prev_tokens_step2,
            encoder_out=enc,
            incremental_state=inc,
            features_only=False,
        )
        self.assertEqual(logits2.shape, (self.B, 1, self.V))

    # 4) forward_external_step returns x_out [1,B,E] and per-layer new K/V [B*H,1,Dh]
    def test_forward_external_step(self):
        dec, _ = self._build_decoder()
        # first token embedding (shape [1,B,E] expected by API)
        prev_x = torch.randn(1, self.B, self.E)
        enc_x = torch.randn(self.S, self.B, self.E)
        kpm = torch.zeros(self.B, self.S, dtype=torch.bool)

        # for the very first step, self-attn history length = 0 → mask should be [1,1]
        sam = torch.zeros(1, 1)

        x_out, new_k_list, new_v_list = dec.forward_external_step(
            prev_output_x=prev_x,
            encoder_out_x=enc_x,
            encoder_padding_mask=kpm,
            self_attn_mask=sam,
            prev_self_k_list=None,
            prev_self_v_list=None,
            need_attn=True,
            need_head_weights=True,
        )
        self.assertEqual(x_out.shape, (1, self.B, self.E))
        self.assertEqual(len(new_k_list), self.L)
        self.assertEqual(len(new_v_list), self.L)
        # verify K/V shapes for each layer: [B*H, 1, Dh]
        # we can infer H and Dh from the first wrapped layer
        first_layer = dec.layers[0]
        # PTQWrapper → wrapped is QuantFairseqDecoderLayer → self_attn has num_heads/head_dim
        H = first_layer.wrapped.self_attn.num_heads  # type: ignore[attr-defined]
        Dh = first_layer.wrapped.self_attn.head_dim  # type: ignore[attr-defined]
        for k, v in zip(new_k_list, new_v_list):
            self.assertEqual(k.shape, (self.B * H, 1, Dh))
            self.assertEqual(v.shape, (self.B * H, 1, Dh))

    # 5) get_normalized_probs: softmax/log_softmax shapes and normalization
    def test_get_normalized_probs(self):
        dec, _ = self._build_decoder()
        prev, enc = self._mk_inputs()
        logits, extra = dec(prev, encoder_out=enc, features_only=False)
        # probabilities
        probs = dec.get_normalized_probs((logits, extra), log_probs=False)
        self.assertEqual(probs.shape, (self.B, self.T, self.V))
        self.assertTrue(
            torch.allclose(probs.sum(dim=-1), torch.ones(self.B, self.T), atol=1e-5)
        )
        # log probabilities
        logp = dec.get_normalized_probs((logits, extra), log_probs=True)
        self.assertEqual(logp.shape, (self.B, self.T, self.V))
        self.assertTrue(
            torch.allclose(
                torch.exp(logp).sum(dim=-1), torch.ones(self.B, self.T), atol=1e-5
            )
        )

    # 6) buffered_future_mask returns additive float mask with correct shape
    def test_buffered_future_mask_shape_and_values(self):
        dec, _ = self._build_decoder()
        x_dummy = torch.zeros(self.T, self.B, self.E)
        mask = dec.buffered_future_mask(self.T, self.T, x=x_dummy)
        self.assertEqual(mask.shape, (self.T, self.T))
        # Strictly upper triangle should be negative, diag/lower zero
        # Build a boolean mask for the upper triangle (j > i)
        upper = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)

        # Strict upper triangle should be negative
        self.assertTrue((mask[upper] < 0).all())

        # Diagonal and lower triangle should be exactly zero
        self.assertTrue((mask[~upper] == 0).all())

    # 7) max_positions respects positional embedding limit
    def test_max_positions(self):
        dec, fp = self._build_decoder(max_target_positions=512, use_positions=True)
        fp.embed_positions.max_positions = 20
        self.assertEqual(dec.max_positions(), 20)

        dec2, _ = self._build_decoder(max_target_positions=16, use_positions=True)
        # larger positional limit than target positions → clamp to 16
        dec2.embed_positions.max_positions = 50
        self.assertEqual(dec2.max_positions(), 16)

        dec3, _ = self._build_decoder(use_positions=False, max_target_positions=30)
        self.assertEqual(dec3.max_positions(), 30)

    # 8) lifecycle propagation to child wrappers (CALIB → QUANT)
    def test_lifecycle_propagation(self):
        dec, _ = self._build_decoder()
        self.assertEqual(dec._mode, Mode.NO_QUANT)

        dec.enable_calibration()
        self.assertEqual(dec._mode, Mode.CALIB)
        # All QuantModuleBase should be CALIB
        for m in dec.modules():
            if not isinstance(m, QuantModuleBase):
                continue
            self.assertTrue(m._mode == Mode.CALIB)

        dec.freeze_qparams()
        self.assertEqual(dec._mode, Mode.QUANT)
        for m in dec.modules():
            if not isinstance(m, QuantModuleBase):
                continue
            self.assertTrue(m._mode == Mode.QUANT)

    # 9) reorder_incremental_state_scripting should be a no-op without modules implementing reordering
    def test_reorder_incremental_state_scripting_noop(self):
        dec, _ = self._build_decoder()
        inc: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        order = torch.tensor([2, 0, 3, 1])
        # Should not raise; most submodules don't implement reorder in this setup
        dec.reorder_incremental_state_scripting(
            inc, order
        )  # no assertion; just smoke test
