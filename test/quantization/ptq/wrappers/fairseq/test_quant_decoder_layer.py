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

from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.quant_config import QuantConfig

from tico.experimental.quantization.ptq.wrappers.fairseq.quant_decoder_layer import (
    QuantFairseqDecoderLayer,
)


# ────────────────────────────────────────────────────────────
#   Minimal fairseq-like FP building blocks for the tests
# ────────────────────────────────────────────────────────────
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


class DummyFSDecoderLayer(nn.Module):
    """
    Minimal TransformerDecoderLayerBase lookalike that QuantFairseqDecoderLayer expects.
    Includes:
      - self_attn, encoder_attn (modules)
      - fc1, fc2 (FFN linears)
      - self_attn_layer_norm, encoder_attn_layer_norm, final_layer_norm (LayerNorms)
      - activation_fn (callable), normalize_before flag
      - Optional: attn_ln (post-self-attn LN), c_attn (per-head scale), ffn_layernorm, w_resid (residual scale)
      - Optional path: cross_self_attention flag
    """

    def __init__(
        self,
        embed_dim: int = 16,
        num_heads: int = 4,
        normalize_before: bool = True,
        use_attn_ln: bool = True,
        use_head_scale: bool = False,
        use_ffn_ln: bool = False,
        use_resid_scale: bool = False,
        cross_self_attention: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalize_before = normalize_before
        self.cross_self_attention = cross_self_attention

        # Attentions
        self.self_attn = _DummyMHA(
            embed_dim, num_heads, self_attention=True, enc_dec=False, bias=bias
        )
        self.encoder_attn = _DummyMHA(
            embed_dim, num_heads, self_attention=False, enc_dec=True, bias=bias
        )

        # FFN
        self.fc1 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.fc2 = nn.Linear(embed_dim, embed_dim, bias=bias)

        # LayerNorms
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        # Optional LNs/scalers
        self.attn_ln = nn.LayerNorm(embed_dim) if use_attn_ln else None
        self.ffn_layernorm = nn.LayerNorm(embed_dim) if use_ffn_ln else None

        if use_head_scale:
            self.c_attn = nn.Parameter(torch.ones(num_heads))  # per-head scaling
        else:
            self.c_attn = None  # type: ignore[assignment]

        if use_resid_scale:
            self.w_resid = nn.Parameter(torch.ones(1))  # scalar residual scale
        else:
            self.w_resid = None  # type: ignore[assignment]

        # Activation
        self.activation_fn = nn.GELU()

        # Align with fairseq API (not strictly needed here)
        self.need_attn = True


# ────────────────────────────────────────────────────────────
#   Test Suite
# ────────────────────────────────────────────────────────────
class TestQuantFairseqDecoderLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.E = 16
        self.H = 4
        self.B = 3
        self.T = 5
        self.S = 6  # encoder length
        self.qcfg = QuantConfig()

    def _mk_inputs(self, T=None, B=None, E=None, S=None):
        T = T or self.T
        B = B or self.B
        E = E or self.E
        S = S or self.S
        x = torch.randn(T, B, E)  # decoder input [T,B,E]
        enc = torch.randn(S, B, E)  # encoder_out [S,B,E]
        kpm = torch.zeros(B, S, dtype=torch.bool)  # encoder padding mask [B,S]
        return x, enc, kpm

    def _build_layer(self, **fp_kwargs):
        fp = DummyFSDecoderLayer(embed_dim=self.E, num_heads=self.H, **fp_kwargs)
        layer = QuantFairseqDecoderLayer(fp, qcfg=self.qcfg, fp_name="dec0")
        return layer, fp

    # 1) Basic forward: returns (x', attn, None) with correct shapes (pre/post-norm)
    def test_forward_shapes_tuple(self):
        for normalize_before in (True, False):
            layer, _ = self._build_layer(normalize_before=normalize_before)
            x, enc, kpm = self._mk_inputs()
            out_x, attn, third = layer(
                x,
                encoder_out=enc,
                encoder_padding_mask=kpm,
                self_attn_mask=torch.triu(
                    torch.ones(self.T, self.T, dtype=torch.bool), diagonal=1
                ),
                self_attn_padding_mask=torch.zeros(self.B, self.T, dtype=torch.bool),
                need_attn=True,
                need_head_weights=False,
            )
            self.assertEqual(out_x.shape, (self.T, self.B, self.E))
            self.assertIsInstance(
                attn, torch.Tensor
            )  # from encoder_attn when need_attn=True
            self.assertIsNone(third)

    # 2) need_head_weights=True should return per-head weights from encoder_attn
    def test_need_head_weights_shape(self):
        layer, _ = self._build_layer()
        x, enc, kpm = self._mk_inputs()
        out_x, attn, _ = layer(
            x,
            encoder_out=enc,
            encoder_padding_mask=kpm,
            need_attn=True,
            need_head_weights=True,
        )
        # QuantFairseqMultiheadAttention returns [H, B, Tq, Ts] when need_head_weights=True
        self.assertIsNotNone(attn)
        self.assertEqual(attn.shape, (self.H, self.B, self.T, self.S))
        self.assertEqual(out_x.shape, (self.T, self.B, self.E))

    # 3) prev_self_attn_state should seed cache; subsequent call should increase cached length
    def test_prev_self_attn_state_accumulate(self):
        layer, _ = self._build_layer()
        x, enc, kpm = self._mk_inputs()
        inc: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

        # Step 1: run without prev cache
        out_x1, _, _ = layer(
            x[:1], encoder_out=enc, encoder_padding_mask=kpm, incremental_state=inc
        )
        self.assertEqual(out_x1.shape, (1, self.B, self.E))

        # Extract internal cache (BH layout stored as [B,H,L,Dh])
        st = layer.self_attn._get_input_buffer(inc)
        self.assertIsNotNone(st)
        L1 = st["prev_key"].shape[-2]
        self.assertEqual(L1, 1)

        # Step 2: provide prev state explicitly and call again
        prev_k, prev_v = st["prev_key"], st["prev_value"]
        out_x2, _, _ = layer(
            x[:1],
            encoder_out=enc,
            encoder_padding_mask=kpm,
            prev_self_attn_state=[prev_k, prev_v],
            incremental_state=inc,
        )
        self.assertEqual(out_x2.shape, (1, self.B, self.E))
        st2 = layer.self_attn._get_input_buffer(inc)
        self.assertGreater(st2["prev_key"].shape[-2], L1)

    # 4) prev_attn_state + static_kv path should reuse encoder KV (no growth)
    def test_prev_attn_state_static_kv_reuse(self):
        layer, _ = self._build_layer()
        x, enc, kpm = self._mk_inputs()
        inc: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

        # First call to seed cross-attn cache
        _, _, _ = layer(
            x[:1], encoder_out=enc, encoder_padding_mask=kpm, incremental_state=inc
        )
        st = layer.encoder_attn._get_input_buffer(inc)
        self.assertIsNotNone(st)
        L_enc = st["prev_key"].shape[-2]

        # Second call: pass prev_attn_state and ensure length stays the same (static_kv=True in wrapper)
        pk, pv = st["prev_key"], st["prev_value"]
        _, _, _ = layer(
            x[:1],
            encoder_out=enc,
            encoder_padding_mask=kpm,
            prev_attn_state=[pk, pv],
            incremental_state=inc,
        )
        st2 = layer.encoder_attn._get_input_buffer(inc)
        self.assertEqual(st2["prev_key"].shape[-2], L_enc)

    # 5) forward_external single-step: returns new_k/new_v in BH layout and [1,B,E] output
    def test_forward_external_single_step(self):
        layer, _ = self._build_layer()
        # Single step input [1,B,E]
        x = torch.randn(1, self.B, self.E)
        enc = torch.randn(self.S, self.B, self.E)
        kpm = torch.zeros(self.B, self.S, dtype=torch.bool)
        # Optional additive mask shapes: [1,S] or [B,1,S]
        sam = torch.zeros(1, 1)  # no penalty

        out_x, attn, new_k, new_v = layer.forward_external(
            x,
            encoder_out=enc,
            encoder_padding_mask=kpm,
            prev_self_k=None,
            prev_self_v=None,
            self_attn_mask=sam,
            need_attn=True,
            need_head_weights=True,
        )
        self.assertEqual(out_x.shape, (1, self.B, self.E))
        self.assertIsInstance(attn, torch.Tensor)
        self.assertEqual(new_k.shape[0], self.B * self.H)
        self.assertEqual(new_v.shape[0], self.B * self.H)
        self.assertEqual(new_k.shape[1], 1)
        self.assertEqual(new_v.shape[1], 1)

    # 6) _maybe_apply_head_scale scales per head as expected (unit-level test)
    def test_maybe_apply_head_scale(self):
        layer, fp = self._build_layer(use_head_scale=True)
        # Set distinct scales per head
        with torch.no_grad():
            layer.c_attn.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        T, B, E, H = 2, 1, self.E, self.H
        Dh = E // H
        x = torch.zeros(T, B, E)
        # Put ones in head slots so scaling shows directly
        x[:, :, :] = 1.0
        y = layer._maybe_apply_head_scale(x.clone())
        # Reshape to [T,B,H,Dh] and check each head got scaled
        y4 = y.view(T, B, H, Dh)
        self.assertTrue(
            torch.allclose(
                y4[0, 0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-6
            )
        )

    # 7) cross_self_attention path runs and preserves shapes (mask/concat logic)
    def test_cross_self_attention_path(self):
        layer, _ = self._build_layer(cross_self_attention=True)
        layer.need_attn = False
        x, enc, kpm = self._mk_inputs()
        # Regular causal mask for targets
        sam = torch.triu(torch.ones(self.T, self.T, dtype=torch.bool), diagonal=1)
        # Target padding mask
        spm = torch.zeros(self.B, self.T, dtype=torch.bool)
        out_x, attn, _ = layer(
            x,
            encoder_out=enc,
            encoder_padding_mask=kpm,
            self_attn_mask=sam,
            self_attn_padding_mask=spm,
            need_attn=False,
        )
        self.assertEqual(out_x.shape, (self.T, self.B, self.E))
        self.assertIsNone(attn)  # need_attn=False → decoder returns None for attn

    # 8) Lifecycle propagation: enable_calibration → freeze_qparams cascades to children
    def test_lifecycle_propagation(self):
        layer, _ = self._build_layer()
        self.assertEqual(layer._mode, Mode.NO_QUANT)

        layer.enable_calibration()
        self.assertEqual(layer._mode, Mode.CALIB)
        # Ensure at least one child observer is enabled
        enabled_seen = False
        for _, obs in layer.named_observers():
            if getattr(obs, "enabled", False):
                enabled_seen = True
                break
        self.assertTrue(enabled_seen)

        layer.freeze_qparams()
        self.assertEqual(layer._mode, Mode.QUANT)
        # Ensure observers are now disabled
        disabled_seen = False
        for _, obs in layer.named_observers():
            if not getattr(obs, "enabled", True):
                disabled_seen = True
                break
        self.assertTrue(disabled_seen)

        # Forward must still work in QUANT
        x, enc, kpm = self._mk_inputs()
        out_x, attn, _ = layer(
            x, encoder_out=enc, encoder_padding_mask=kpm, need_attn=True
        )
        self.assertEqual(out_x.shape, (self.T, self.B, self.E))
        self.assertIsInstance(attn, torch.Tensor)
