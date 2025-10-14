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
from tico.experimental.quantization.config.ptq import PTQConfig

from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.wrappers.fairseq.quant_encoder import (
    QuantFairseqEncoder,
)
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.ptq.wrappers.registry import register


# ────────────────────────────────────────────────────────────
#   Minimal FP building blocks used by the encoder stub
# ────────────────────────────────────────────────────────────
class DummyCfg:
    def __init__(self, no_scale_embedding: bool = False):
        self.no_scale_embedding = no_scale_embedding


class DummyPositionalEmbedding(nn.Module):
    """
    Returns zeros with the same (B, T, C) shape as embeddings.
    Exposes `max_positions` like fairseq's positional embedding.
    """

    def __init__(self, embed_dim: int, max_positions: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T)
        B, T = tokens.shape
        return torch.zeros(
            B, T, self.embed_dim, dtype=torch.float32, device=tokens.device
        )


class DummyFPEncoderLayer(nn.Module):
    """
    A light FP encoder layer: keeps an attribute `return_fc` and simply passes through.
    Its quant wrapper will be registered below so PTQWrapper can wrap it.
    """

    def __init__(self, embed_dim: int, return_fc: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.return_fc = return_fc

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError("This FP layer is wrapped and not called directly.")


# Register a tiny quant wrapper so PTQWrapper(DummyFPEncoderLayer) works.
@register(DummyFPEncoderLayer)
class QuantDummyEncoderLayer(QuantModuleBase):
    """
    Minimal quant-aware wrapper used only for tests.
    Behavior: identity transform; optionally returns (x, x) if fp_layer.return_fc is True.
    """

    def __init__(
        self,
        fp_layer: DummyFPEncoderLayer,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.embed_dim = fp_layer.embed_dim
        self._return_fc = bool(getattr(fp_layer, "return_fc", False))
        # One simple observer to exercise lifecycle
        self.obs_identity = self._make_obs("identity")

    def forward(
        self,
        x: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x = self._fq(x, self.obs_identity)
        if self._return_fc:
            return x, x  # fc_result == x (for testability)
        return x

    def _all_observers(self):
        yield from (self.obs_identity,)


class DummyFPEncoder(nn.Module):
    """
    A fairseq-like FP encoder providing the attributes used by QuantFairseqEncoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        padding_idx: int = 0,
        max_source_positions: int = 1024,
        use_positional: bool = True,
        use_embed_ln: bool = True,
        use_final_ln: bool = True,
        no_scale_embedding: bool = False,
        return_fc: bool = False,
        version_val: float = 3.0,
    ):
        super().__init__()
        self.cfg = DummyCfg(no_scale_embedding=no_scale_embedding)
        self.return_fc = return_fc
        self.padding_idx = padding_idx
        self.max_source_positions = max_source_positions

        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # Positional embedding and LNs (optional)
        self.embed_positions = (
            DummyPositionalEmbedding(embed_dim, max_source_positions)
            if use_positional
            else None
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim) if use_embed_ln else None
        self.layer_norm = nn.LayerNorm(embed_dim) if use_final_ln else None

        # Stacked layers
        self.layers = nn.ModuleList(
            [
                DummyFPEncoderLayer(embed_dim, return_fc=return_fc)
                for _ in range(num_layers)
            ]
        )

        # Version buffer (mirrors fairseq encoders)
        self.version = torch.tensor([version_val], dtype=torch.float32)


# ────────────────────────────────────────────────────────────
#   Test Suite
# ────────────────────────────────────────────────────────────
class TestQuantFairseqEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab = 50
        self.E = 32
        self.B = 4
        self.T = 7
        self.layers = 3
        self.pad_idx = 1

    def _make_src(self, pad_last_col: bool = True):
        # Random tokens in [2, vocab-1], reserve 0 for BOS-like and 1 for PAD by convention here
        tokens = torch.randint(low=2, high=self.vocab, size=(self.B, self.T))
        if pad_last_col:
            tokens[:, -1] = self.pad_idx
        return tokens

    def _build_encoder(self, **fp_kwargs):
        fp = DummyFPEncoder(
            vocab_size=self.vocab,
            embed_dim=self.E,
            num_layers=self.layers,
            padding_idx=self.pad_idx,
            **fp_kwargs,
        )
        enc = QuantFairseqEncoder(
            fp,
            qcfg=PTQConfig(),
            fp_name="enc",
            use_external_inputs=False,
            return_type="dict",
        )
        return enc, fp

    # 1) Standard path returns a dict with expected keys and shapes
    def test_forward_standard_dict_shapes(self):
        enc, _ = self._build_encoder()
        src = self._make_src(pad_last_col=True)
        out = enc(src, return_all_hiddens=True)

        # Keys existence
        for key in [
            "encoder_out",
            "encoder_padding_mask",
            "encoder_embedding",
            "encoder_states",
            "fc_results",
            "src_tokens",
            "src_lengths",
        ]:
            self.assertIn(key, out)

        # Shapes
        x = out["encoder_out"][0]
        self.assertEqual(x.shape, (self.T, self.B, self.E))
        kpm = out["encoder_padding_mask"][0]
        self.assertEqual(kpm.shape, (self.B, self.T))
        emb = out["encoder_embedding"][0]
        self.assertEqual(emb.shape, (self.B, self.T, self.E))
        # states length = layers + 1 (initial) in return_all_hiddens=True
        self.assertEqual(len(out["encoder_states"]), self.layers + 1)
        # src_lengths shape
        self.assertEqual(out["src_lengths"][0].shape, (self.B, 1))

    # 2) Padding logic: padded time steps are zeroed before the stack
    def test_padding_zeroing_before_stack(self):
        enc, _ = self._build_encoder()
        src = self._make_src(pad_last_col=True)
        out = enc(src, return_all_hiddens=True)
        # The first encoder_states entry is x right after transpose, after zeroing pads
        first_state = out["encoder_states"][0]  # (T, B, C)
        # Last time step is padded for all samples → should be all zeros
        last_timestep = first_state[-1]  # (B, C)
        self.assertTrue(
            torch.allclose(
                last_timestep, torch.zeros_like(last_timestep), atol=0, rtol=0
            )
        )

    # 3) External-inputs mode: return a single Tensor (T, B, C)
    def test_external_inputs_tensor_return(self):
        # Build encoder configured for external inputs
        enc, fp = self._build_encoder()
        enc.use_external_inputs = True
        enc.return_type = "tensor"

        # External x is already (T, B, C)
        x_ext = torch.randn(self.T, self.B, self.E)
        # Provide an encoder padding mask (B, T) to propagate through
        kpm = torch.zeros(self.B, self.T, dtype=torch.bool)

        out = enc(x_ext, encoder_padding_mask=kpm)  # src_tokens is x_ext here
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (self.T, self.B, self.E))

    # 4) External-inputs mode with dict return
    def test_external_inputs_dict_return(self):
        enc, _ = self._build_encoder()
        enc.use_external_inputs = True
        enc.return_type = "dict"

        x_ext = torch.randn(self.T, self.B, self.E)
        kpm = torch.zeros(self.B, self.T, dtype=torch.bool)
        out = enc(x_ext, encoder_padding_mask=kpm, return_all_hiddens=True)

        self.assertIn("encoder_out", out)
        self.assertEqual(out["encoder_out"][0].shape, (self.T, self.B, self.E))
        self.assertIn("encoder_states", out)
        # states length = layers (+1 if return_all_hiddens=True)
        self.assertEqual(len(out["encoder_states"]), self.layers + 1)

    # 5) Lifecycle: enable_calibration → freeze_qparams propagates to child wrappers
    def test_lifecycle_propagation(self):
        enc, _ = self._build_encoder()
        self.assertEqual(enc._mode, Mode.NO_QUANT)

        enc.enable_calibration()
        self.assertEqual(enc._mode, Mode.CALIB)
        # At least one child quant module should also be in CALIB
        child_modes = []
        for m in enc.modules():
            if isinstance(m, QuantModuleBase) and m is not enc:
                child_modes.append(m._mode)
        self.assertTrue(any(m == Mode.CALIB for m in child_modes))

        enc.freeze_qparams()
        self.assertEqual(enc._mode, Mode.QUANT)
        child_modes = []
        for m in enc.modules():
            if isinstance(m, QuantModuleBase) and m is not enc:
                child_modes.append(m._mode)
        self.assertTrue(any(m == Mode.QUANT for m in child_modes))

    # 6) reorder_encoder_out should permute the batch dimension
    def test_reorder_encoder_out(self):
        enc, _ = self._build_encoder()
        src = self._make_src(pad_last_col=False)
        out = enc(src, return_all_hiddens=True)

        new_order = torch.tensor([2, 0, 3, 1])  # permute B
        re = enc.reorder_encoder_out(out, new_order)

        # encoder_out: (T, B, C) → select(1, new_order)
        old_x = out["encoder_out"][0]
        new_x = re["encoder_out"][0]
        self.assertTrue(torch.allclose(new_x, old_x.index_select(1, new_order)))

        # encoder_padding_mask: (B, T) → select(0, new_order)
        old_kpm = out["encoder_padding_mask"][0]
        new_kpm = re["encoder_padding_mask"][0]
        self.assertTrue(torch.equal(new_kpm, old_kpm.index_select(0, new_order)))

    # 7) max_positions should respect positional embed limits if present
    def test_max_positions_with_and_without_positional(self):
        # With positional embedding (max_positions=20)
        enc, fp = self._build_encoder(max_source_positions=100, use_positional=True)
        fp.embed_positions.max_positions = 20
        self.assertEqual(enc.max_positions(), 20)

        # Without positional embedding
        enc2, _ = self._build_encoder(max_source_positions=100, use_positional=False)
        self.assertEqual(enc2.max_positions(), 100)

        # If source limit is smaller than positional limit, it clamps
        enc3, fp3 = self._build_encoder(max_source_positions=16, use_positional=True)
        fp3.embed_positions.max_positions = 50
        self.assertEqual(enc3.max_positions(), 16)

    # 8) forward_torchscript should obey presence/absence of encoder_padding_mask
    def test_forward_torchscript_wrapper(self):
        enc, _ = self._build_encoder()
        src = self._make_src(pad_last_col=True)
        kpm = src.eq(self.pad_idx)

        out1 = enc.forward_torchscript(
            {
                "src_tokens": src,
                "src_lengths": torch.sum(src != self.pad_idx, dim=1),
                "encoder_padding_mask": kpm,
            }
        )
        self.assertIn("encoder_out", out1)

        out2 = enc.forward_torchscript(
            {"src_tokens": src, "src_lengths": torch.sum(src != self.pad_idx, dim=1)}
        )
        self.assertIn("encoder_out", out2)

    # 9) upgrade_state_dict_named should set layer_norm=None if version < 2
    def test_upgrade_state_dict_named_behavior(self):
        # Build FP encoder with old version=1.0 to trigger path
        enc, fp = self._build_encoder(version_val=1.0)
        sd = {}
        name = "enc"
        sd[f"{name}.version"] = torch.tensor([1.0])
        out_sd = enc.upgrade_state_dict_named(sd, name)

        # layer_norm becomes None; version coerced to [1]
        self.assertIsNone(enc.layer_norm)
        self.assertIn(f"{name}.version", out_sd)
        self.assertEqual(float(out_sd[f"{name}.version"][0]), 1.0)

    # 10) Numerical sanity in both scaling modes
    def test_numerical_sanity_with_or_without_embed_scale(self):
        # no_scale_embedding=False → scale = sqrt(E)
        enc1, _ = self._build_encoder(no_scale_embedding=False)
        y1 = enc1(self._make_src(), return_all_hiddens=False)
        self.assertFalse(torch.isnan(y1["encoder_out"][0]).any())

        # no_scale_embedding=True → scale = 1.0
        enc2, _ = self._build_encoder(no_scale_embedding=True)
        y2 = enc2(self._make_src(), return_all_hiddens=False)
        self.assertFalse(torch.isnan(y2["encoder_out"][0]).any())
