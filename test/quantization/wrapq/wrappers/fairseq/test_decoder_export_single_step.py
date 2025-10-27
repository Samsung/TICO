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
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tico.quantization.wrapq.wrappers.fairseq.decoder_export_single_step import (
    DecoderExportSingleStep,
    export_decoder_single_step,
    make_example_inputs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dummies for testing (no fairseq needed)
# ─────────────────────────────────────────────────────────────────────────────
class _DummySelfAttnMeta:
    """Carries num_heads and head_dim; mirrors what decoder layer exposes."""

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim


class _DummyLayerForExport(nn.Module):
    """Layer object that `DecoderExportSingleStep` can read heads from."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.self_attn = _DummySelfAttnMeta(num_heads, head_dim)


class _DummyExportableDecoder(nn.Module):
    """
    Minimal decoder with the attributes/method used by DecoderExportSingleStep:
      - layers: list with element exposing `.self_attn.num_heads/head_dim`
      - embed_dim: model hidden size C
      - forward_external_step(prev_output_x, encoder_out_x, encoder_padding_mask, self_attn_mask, prev_self_k_list, prev_self_v_list)
          -> returns (x_step [1,B,C], new_k_list, new_v_list) where new_k/v
             are lists of length L, each [B*H, 1, Dh]
    """

    def __init__(self, *, L=3, B=2, S=11, C=32, H=4, Dh=8):
        super().__init__()
        self.embed_dim = C
        self.layers = nn.ModuleList([_DummyLayerForExport(H, Dh) for _ in range(L)])
        self._B = B
        self._S = S
        self._H = H
        self._Dh = Dh
        self._L = L

    def forward_external_step(
        self,
        *,
        prev_output_x: torch.Tensor,  # [1,B,C]
        encoder_out_x: torch.Tensor,  # [S,B,C]
        encoder_padding_mask: torch.Tensor,  # [B,1,S] additive float
        self_attn_mask: torch.Tensor,  # [B,1,S_tot] additive float
        prev_self_k_list,
        prev_self_v_list,
        **kwargs,
    ):
        # Return a deterministic x_step and per-layer K/V with expected shapes
        x_step = torch.zeros_like(prev_output_x) + 1.23
        new_k_list = []
        new_v_list = []
        for _ in range(self._L):
            new_k = torch.zeros(self._B * self._H, 1, self._Dh) + 0.5
            new_v = torch.zeros(self._B * self._H, 1, self._Dh) + 0.7
            new_k_list.append(new_k)
            new_v_list.append(new_v)
        return x_step, new_k_list, new_v_list


# ─────────────────────────────────────────────────────────────────────────────
# Test Suite
# ─────────────────────────────────────────────────────────────────────────────
class TestDecoderExportSingleStep(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.L = 3
        self.B = 2
        self.S = 11
        self.C = 32
        self.H = 4
        self.Dh = 8
        self.Tprev = 5

        self.dec = _DummyExportableDecoder(
            L=self.L, B=self.B, S=self.S, C=self.C, H=self.H, Dh=self.Dh
        ).eval()
        self.wrapper = DecoderExportSingleStep(self.dec).eval()

    # 1) Basic forward: shapes of x and K/V per layer line up with spec
    def test_forward_shapes(self):
        prev_x = torch.randn(self.B, 1, self.C)  # [B,1,C]
        enc_x = torch.randn(self.S, self.B, self.C)  # [S,B,C]
        enc_pad_add = torch.zeros(self.B, 1, self.S)  # [B,1,S] additive
        self_attn_mask = torch.zeros(self.B, 1, self.S)  # [B,1,S_tot] here == S

        # Build caches for each layer: [B,H,Tprev,Dh]
        kv = []
        for _ in range(self.L):
            k = torch.randn(self.B, self.H, self.Tprev, self.Dh)
            v = torch.randn(self.B, self.H, self.Tprev, self.Dh)
            kv += [k, v]

        out = self.wrapper(
            prev_x, enc_x, enc_pad_add, *kv, self_attn_mask=self_attn_mask
        )
        # First tensor is x_step
        x_step = out[0]
        self.assertEqual(x_step.shape, (self.B, 1, self.C))

        # Then 2 * L tensors, each [B*H, 1, Dh]
        self.assertEqual(len(out), 1 + 2 * self.L)
        for i in range(self.L):
            nk = out[1 + 2 * i + 0]
            nv = out[1 + 2 * i + 1]
            self.assertEqual(nk.shape, (self.B * self.H, 1, self.Dh))
            self.assertEqual(nv.shape, (self.B * self.H, 1, self.Dh))

    # 2) KV arg count mismatch should assert
    def test_kv_arg_count_mismatch(self):
        prev_x = torch.randn(self.B, 1, self.C)
        enc_x = torch.randn(self.S, self.B, self.C)
        enc_pad_add = torch.zeros(self.B, 1, self.S)
        sam = torch.zeros(self.B, 1, self.S)

        # Provide too few KV tensors (expected 2*L)
        with self.assertRaises(AssertionError):
            self.wrapper(prev_x, enc_x, enc_pad_add, *[], self_attn_mask=sam)

    # 3) Meta inference from decoder (num_layers, heads, head_dim, embed_dim)
    def test_meta_inference(self):
        self.assertEqual(self.wrapper.num_layers, self.L)
        self.assertEqual(self.wrapper.num_heads, self.H)
        self.assertEqual(self.wrapper.head_dim, self.Dh)
        self.assertEqual(self.wrapper.embed_dim, self.C)

    # 4) make_example_inputs: shapes match the docstring
    def test_make_example_inputs_shapes(self):
        L, B, S, H, Dh, C, Tprev = 4, 1, 64, 8, 64, 512, 63
        args, kwargs = make_example_inputs(
            L=L, B=B, S=S, H=H, Dh=Dh, C=C, Tprev=Tprev, device="cpu"
        )
        self.assertEqual(len(args), 3 + 2 * L)  # prev_x, enc_x, enc_pad + 2L K/V
        prev_x, enc_x, enc_pad = args[0], args[1], args[2]
        self.assertEqual(prev_x.shape, (B, 1, C))
        self.assertEqual(enc_x.shape, (S, B, C))
        self.assertEqual(enc_pad.shape, (B, 1, S))
        # Each KV: [B,H,Tprev,Dh]
        for i in range(L):
            k = args[3 + i]
            v = args[3 + L + i]
            self.assertEqual(k.shape, (B, H, Tprev, Dh))
            self.assertEqual(v.shape, (B, H, Tprev, Dh))
        # self_attn_mask in kwargs: [B,1,S]
        self.assertIn("self_attn_mask", kwargs)
        self.assertEqual(kwargs["self_attn_mask"].shape, (B, 1, S))

    # 5) export_decoder_single_step calls tico.convert with proper wrapper and example I/O
    def test_export_decoder_single_step_invokes_convert(self):
        class _FakeModel:
            def __init__(self, dec):
                self.decoder = dec

        class _FakeTranslator:
            def __init__(self, dec):
                self.models = [_FakeModel(dec)]

        fake_translator = _FakeTranslator(self.dec)

        # Patch tico.convert to validate inputs and emulate a returned object with .save()
        with patch("tico.convert") as mock_convert:
            # Emulate a compiled model handle with .save()
            cmock = MagicMock()
            mock_convert.return_value = cmock

            export_decoder_single_step(fake_translator, save_path="dummy.circle")

            # Ensure convert was called once with a DecoderExportSingleStep and example tensors
            self.assertTrue(mock_convert.called)
            call_args, call_kwargs = mock_convert.call_args
            self.assertIsInstance(
                call_args[0], DecoderExportSingleStep
            )  # wrapper module

            # Verify example I/O presence and basic shapes
            ex_args = call_kwargs["args"]
            ex_kwargs = call_kwargs["kwargs"]
            self.assertIsInstance(ex_args, tuple)
            self.assertIn("self_attn_mask", ex_kwargs)
            # First two positional args are prev_x and enc_x
            self.assertTrue(
                ex_args[0].dim() == 3 and ex_args[0].size(1) == 1
            )  # prev_x [B,1,C]
            self.assertTrue(ex_args[1].dim() == 3)  # enc_x [S,B,C]
            # save() must be called with the provided path
            cmock.save.assert_called_once_with("dummy.circle")
