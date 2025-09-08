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
from typing import Optional

import torch
import torch.nn as nn
from tico.experimental.quantization.ptq.dtypes import DType
from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.qscheme import QScheme

from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.fairseq.quant_encoder_layer import (
    QuantFairseqEncoderLayer,
)
from tico.experimental.quantization.ptq.wrappers.fairseq.quant_mha import (
    QuantFairseqMultiheadAttention,
)
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper


# ────────────────────────────────────────────────────────────
#   Minimal fairseq-like FP encoder layer to wrap in tests
# ────────────────────────────────────────────────────────────
class DummyFSEncoderLayer(nn.Module):
    """
    Minimal TransformerEncoderLayerBase lookalike with:
      - self_attn (multi-head self attention module)
      - fc1, fc2 (feedforward linears)
      - self_attn_layer_norm, final_layer_norm (LayerNorm)
      - flags: embed_dim, normalize_before, return_fc
      - activation_fn (callable)
    """

    def __init__(
        self,
        embed_dim=16,
        num_heads=4,
        normalize_before=True,
        return_fc=False,
        bias=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalize_before = normalize_before
        self.return_fc = return_fc

        # Self-attention submodule (self-attention flavor)
        self.self_attn = _DummySelfAttn(embed_dim, num_heads, bias=bias)

        # FFN
        self.fc1 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.fc2 = nn.Linear(embed_dim, embed_dim, bias=bias)

        # LayerNorms
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        # Activation
        self.activation_fn = nn.GELU()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Not used in tests; wrapper handles forward.")


class _DummySelfAttn(nn.Module):
    """Self-attention-shaped module providing q/k/v/out_proj linears and fairseq flags."""

    def __init__(self, embed_dim, num_heads, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attention = True
        self.encoder_decoder_attention = False
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


class TestQuantFairseqEncoderLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.E = 16  # embed dim
        self.H = 4  # num heads
        self.B = 3  # batch size
        self.T = 5  # sequence length

    def _make_inputs(self, T=None):
        T = T or self.T
        x = torch.randn(T, self.B, self.E)
        return x

    def _build_layer(
        self, normalize_before=True, return_fc=False, qcfg: Optional[QuantConfig] = None
    ):
        fp = DummyFSEncoderLayer(
            embed_dim=self.E,
            num_heads=self.H,
            normalize_before=normalize_before,
            return_fc=return_fc,
        )
        layer = QuantFairseqEncoderLayer(fp, qcfg=qcfg, fp_name="enc0")
        return layer, fp

    # 1) Forward shape for both pre/post-norm and with/without return_fc
    def test_forward_shapes_flags(self):
        for normalize_before in (True, False):
            for return_fc in (True, False):
                layer, _ = self._build_layer(normalize_before, return_fc, QuantConfig())
                x = self._make_inputs()
                out = layer(x, encoder_padding_mask=None, attn_mask=None)
                if return_fc:
                    y, fc = out
                    self.assertEqual(y.shape, (self.T, self.B, self.E))
                    self.assertEqual(fc.shape, (self.T, self.B, self.E))
                else:
                    y = out
                    self.assertEqual(y.shape, (self.T, self.B, self.E))

    # 2) Mask handling: boolean causal attn_mask and boolean key_padding_mask
    def test_mask_handling(self):
        layer, _ = self._build_layer(True, False, QuantConfig())
        x = self._make_inputs()
        # Upper-triangular causal mask [T,S] (True = masked)
        attn_mask_bool = torch.triu(
            torch.ones(self.T, self.T, dtype=torch.bool), diagonal=1
        )
        # Key padding mask [B,S] (True = pad)
        kpm_bool = torch.zeros(self.B, self.T, dtype=torch.bool)
        kpm_bool[:, -1] = True  # pad last position
        y = layer(x, encoder_padding_mask=kpm_bool, attn_mask=attn_mask_bool)
        self.assertEqual(y.shape, (self.T, self.B, self.E))

    # 3) Additive float masks should also work
    def test_additive_float_masks(self):
        layer, _ = self._build_layer(True, False, QuantConfig())
        x = self._make_inputs()
        attn_mask_add = torch.zeros(self.T, self.T)
        attn_mask_add = attn_mask_add.fill_diagonal_(0.0)
        attn_mask_add = attn_mask_add + torch.triu(
            torch.full_like(attn_mask_add, -120.0), diagonal=1
        )
        kpm_add = torch.zeros(self.B, self.T)
        kpm_add[:, -1] = -120.0
        y = layer(x, encoder_padding_mask=kpm_add, attn_mask=attn_mask_add)
        self.assertEqual(y.shape, (self.T, self.B, self.E))
        self.assertFalse(torch.isnan(y).any())

    # 4) return_fc semantics (pre-norm): out == residual + fc_result
    def test_return_fc_semantics_pre_norm(self):
        # Only valid to check exactly when normalize_before=True (no post-FFN norm)
        qcfg = QuantConfig()
        layer, _ = self._build_layer(normalize_before=True, return_fc=True, qcfg=qcfg)
        x0 = self._make_inputs()

        # Run the wrapper to get outputs
        out, fc_result = layer(x0, encoder_padding_mask=None, attn_mask=None)

        # Reproduce the "residual" entering FFN block using the wrapper's own submodules
        # (This mirrors the encoder forward up to the FFN residual.)
        # Pre-norm path:
        #   attn_in = self_attn_layer_norm(x0)
        #   attn_out = self_attn(attn_in, attn_in, attn_in)
        #   x_after_attn = x0 + attn_out
        attn_in = layer.self_attn_layer_norm(x0)
        attn_out, _ = layer.self_attn(
            attn_in, attn_in, attn_in, key_padding_mask=None, attn_mask=None
        )
        residual = (
            x0 + attn_out
        )  # this becomes "residual" at the start of FFN in pre-norm

        # In pre-norm, wrapper returns: out = residual + fc_result, without a post-norm
        self.assertTrue(torch.allclose(out, residual + fc_result, atol=1e-5, rtol=1e-5))

    # 5) Lifecycle: enable_calibration → freeze_qparams should propagate to children
    def test_lifecycle_propagation(self):
        layer, _ = self._build_layer(True, False, QuantConfig())
        self.assertEqual(layer._mode, Mode.NO_QUANT)

        layer.enable_calibration()
        self.assertEqual(layer._mode, Mode.CALIB)
        for name, obs in layer.named_observers():
            self.assertTrue(
                getattr(obs, "enabled", False),
                f"observer {name} should be enabled in CALIB",
            )

        layer.freeze_qparams()
        self.assertEqual(layer._mode, Mode.QUANT)
        for name, obs in layer.named_observers():
            self.assertFalse(
                getattr(obs, "enabled", True),
                f"observer {name} should be disabled in QUANT",
            )

        # Forward should still work in QUANT mode
        x = self._make_inputs()
        y = layer(x, encoder_padding_mask=None, attn_mask=None)
        if layer.return_fc:
            y = y[0]
        self.assertEqual(y.shape, (self.T, self.B, self.E))

    # 6) Observer wiring and child configs: override activation observer dtype/qscheme
    def test_qcfg_child_overrides_for_activation_observer(self):
        qcfg = QuantConfig(
            default_dtype=DType.uint(8),
            default_qscheme=QScheme.PER_TENSOR_ASYMM,
            overrides={
                # The encoder layer defines an internal observer named "activation_fn"
                "activation_fn": {
                    "dtype": DType.uint(4),
                    "qscheme": QScheme.PER_CHANNEL_ASYMM,
                },
                # Also ensure child scoping is well-formed for attention/ffn if needed
                "self_attn": {},
                "fc1": {},
                "fc2": {},
            },
        )
        layer, _ = self._build_layer(True, False, qcfg)
        act_obs = layer.get_observer("activation_fn")
        self.assertIsNotNone(act_obs)
        self.assertEqual(getattr(act_obs, "dtype", None), DType.uint(4))
        self.assertEqual(getattr(act_obs, "qscheme", None), QScheme.PER_CHANNEL_ASYMM)

    # 7) Sanity: submodules are properly wrapped
    def test_submodules_wrapped(self):
        layer, _ = self._build_layer(True, False, QuantConfig())
        # Self-attention should be QuantFairseqMultiheadAttention
        self.assertIsInstance(layer.self_attn, QuantFairseqMultiheadAttention)
        # FFN and norms are wrapped by PTQWrapper
        self.assertIsInstance(layer.fc1, PTQWrapper)
        self.assertIsInstance(layer.fc2, PTQWrapper)
        self.assertIsInstance(layer.self_attn_layer_norm, PTQWrapper)
        self.assertIsInstance(layer.final_layer_norm, PTQWrapper)

    # 8) Pre- and post-norm numerical sanity (no NaNs; gradient not required)
    def test_numerical_sanity(self):
        for normalize_before in (True, False):
            layer, _ = self._build_layer(normalize_before, False, QuantConfig())
            x = self._make_inputs()
            y = layer(x, encoder_padding_mask=None, attn_mask=None)
            self.assertFalse(torch.isnan(y).any())
