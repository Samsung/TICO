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
from tico.experimental.quantization.ptq.wrappers.fairseq.quant_mha import (
    QuantFairseqMultiheadAttention,
)


# ────────────────────────────────────────────────────────────
#   Minimal FP attention stubs to inject into the wrapper
# ────────────────────────────────────────────────────────────
class DummySelfAttn(nn.Module):
    """
    Fairseq-like MultiheadAttention stub (self-attention flavor).
    Only provides attributes and modules required by the wrapper.
    """

    def __init__(self, embed_dim: int = 8, num_heads: int = 2, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attention = True
        self.encoder_decoder_attention = False

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


class DummyCrossAttn(nn.Module):
    """
    Fairseq-like MultiheadAttention stub (encoder-decoder cross-attention).
    """

    def __init__(self, embed_dim: int = 8, num_heads: int = 2, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attention = False
        self.encoder_decoder_attention = True

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


class TestQuantFairseqMHA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.E = 8  # embed dim
        self.H = 2  # num heads
        self.Dh = self.E // self.H
        self.B = 3  # batch
        self.T = 4  # sequence length
        self.qcfg = QuantConfig()

    def _make_inputs(self, Tq=None, Tk=None, Tv=None):
        Tq = Tq or self.T
        Tk = Tk or self.T
        Tv = Tv or self.T
        query = torch.randn(Tq, self.B, self.E)
        key = torch.randn(Tk, self.B, self.E)
        value = torch.randn(Tv, self.B, self.E)
        return query, key, value

    # 1) Self-attention basic forward: output shapes and types
    def test_forward_self_attention_shapes(self):
        fp = DummySelfAttn(self.E, self.H)
        attn = QuantFairseqMultiheadAttention(
            fp, qcfg=self.qcfg, fp_name="mha", use_static_causal=False
        )
        q, _, _ = self._make_inputs()
        out, weights = attn(
            q, None, None, need_weights=False
        )  # SA: key/value can be None
        self.assertEqual(out.shape, (self.T, self.B, self.E))
        self.assertIsNone(weights)
        self.assertEqual(attn.head_dim, self.Dh)
        self.assertEqual(attn.num_heads, self.H)

    # 2) Static causal mask: upper-triangular logits should be much smaller
    def test_static_causal_mask_upper_triangle_small(self):
        fp = DummySelfAttn(self.E, self.H)
        attn = QuantFairseqMultiheadAttention(
            fp,
            qcfg=self.qcfg,
            fp_name="mha",
            use_static_causal=True,
            mask_fill_value=-120.0,
        )
        Tq = Ts = 5
        q, _, _ = self._make_inputs(Tq, Ts, Ts)
        logits, _ = attn(q, None, None, attn_mask=None, before_softmax=True)
        self.assertEqual(logits.shape, (self.B * self.H, Tq, Ts))

        # For each row i, entries j>i should be << entries j<=i
        with torch.no_grad():
            for b in range(self.B * self.H):
                for i in range(Tq):
                    row = logits[b, i]
                    allowed = row[: i + 1]
                    blocked = row[i + 1 :]
                    if blocked.numel() == 0:
                        continue
                    self.assertTrue(blocked.max().item() < allowed.min().item())

    # 3) need_head_weights=True: per-head weight tensor shape
    def test_need_head_weights_shape(self):
        fp = DummySelfAttn(self.E, self.H)
        attn = QuantFairseqMultiheadAttention(
            fp, qcfg=self.qcfg, fp_name="mha", use_static_causal=True
        )
        q, _, _ = self._make_inputs()
        out, weights = attn(q, None, None, need_weights=True, need_head_weights=True)
        self.assertEqual(out.shape, (self.T, self.B, self.E))
        self.assertIsNotNone(weights)
        # Implementation: [H, B, Tq, Ts]
        self.assertEqual(weights.shape, (self.H, self.B, self.T, self.T))

    # 4) Incremental self-attention: cache should accumulate across calls
    def test_incremental_self_attention_accumulates(self):
        fp = DummySelfAttn(self.E, self.H)
        attn = QuantFairseqMultiheadAttention(
            fp, qcfg=self.qcfg, fp_name="mha", use_static_causal=True
        )
        inc: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

        # Step 1 (T=1)
        q1 = torch.randn(1, self.B, self.E)
        out1, _ = attn(q1, None, None, incremental_state=inc)
        assert isinstance(out1, torch.Tensor)
        self.assertEqual(out1.shape, (1, self.B, self.E))
        state = inc.get(attn._state_key)
        self.assertIsNotNone(state)
        assert state is not None
        assert "prev_key" in state and "prev_value" in state
        assert isinstance(state["prev_key"], torch.Tensor)
        assert isinstance(state["prev_value"], torch.Tensor)
        self.assertEqual(state["prev_key"].shape[-2], 1)
        self.assertEqual(state["prev_value"].shape[-2], 1)

        # Step 2 (another token)
        q2 = torch.randn(1, self.B, self.E)
        out2, _ = attn(q2, None, None, incremental_state=inc)
        assert isinstance(out2, torch.Tensor)
        self.assertEqual(out2.shape, (1, self.B, self.E))
        state = inc.get(attn._state_key)
        assert state is not None
        assert "prev_key" in state and "prev_value" in state
        assert isinstance(state["prev_key"], torch.Tensor)
        assert isinstance(state["prev_value"], torch.Tensor)
        self.assertEqual(state["prev_key"].shape[-2], 2)
        self.assertEqual(state["prev_value"].shape[-2], 2)

    # 5) return_new_kv=True should expose newly produced KV in BH layout
    def test_return_new_kv_shapes(self):
        fp = DummySelfAttn(self.E, self.H)
        attn = QuantFairseqMultiheadAttention(fp, qcfg=self.qcfg, fp_name="mha")
        inc: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

        q = torch.randn(2, self.B, self.E)
        out, weights, new_k, new_v = attn(
            q, None, None, incremental_state=inc, return_new_kv=True
        )
        self.assertEqual(out.shape, (2, self.B, self.E))
        self.assertIsNotNone(new_k)
        self.assertIsNotNone(new_v)
        self.assertEqual(new_k.shape, (self.B * self.H, 2, self.Dh))
        self.assertEqual(new_v.shape, (self.B * self.H, 2, self.Dh))

    # 6) Cross-attention + static_kv=True: always pass key/value to avoid assertion
    #    First call seeds cache with memory length 5; second call reuses it without concatenation.
    def test_cross_attention_static_kv_reuse(self):
        fp = DummyCrossAttn(self.E, self.H)
        attn = QuantFairseqMultiheadAttention(fp, qcfg=self.qcfg, fp_name="xattn")
        inc: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

        # Seed cache (memory length = 5)
        q1 = torch.randn(1, self.B, self.E)
        k_mem = torch.randn(5, self.B, self.E)
        v_mem = torch.randn(5, self.B, self.E)
        out1, _ = attn(q1, k_mem, v_mem, incremental_state=inc)
        self.assertEqual(out1.shape, (1, self.B, self.E))
        state = inc.get(attn._state_key)
        assert isinstance(state, dict)
        assert "prev_key" in state and "prev_value" in state
        assert isinstance(state["prev_key"], torch.Tensor)
        assert isinstance(state["prev_value"], torch.Tensor)
        self.assertEqual(state["prev_key"].shape[-2], 5)
        self.assertEqual(state["prev_value"].shape[-2], 5)

        # Second call with static_kv=True: still pass non-None k/v to satisfy assertion;
        # the wrapper will reuse cached KV and ignore these by setting k=v=None internally.
        q2 = torch.randn(1, self.B, self.E)
        dummy_k = torch.randn(2, self.B, self.E)
        dummy_v = torch.randn(2, self.B, self.E)
        out2, _ = attn(q2, dummy_k, dummy_v, incremental_state=inc, static_kv=True)
        self.assertEqual(out2.shape, (1, self.B, self.E))
        state2 = inc.get(attn._state_key)
        assert isinstance(state2, dict)
        assert "prev_key" in state2 and "prev_value" in state2
        assert isinstance(state2["prev_key"], torch.Tensor)
        assert isinstance(state2["prev_value"], torch.Tensor)
        self.assertEqual(state2["prev_key"].shape[-2], 5)
        self.assertEqual(state2["prev_value"].shape[-2], 5)

    # 7) Lifecycle: enable_calibration() → freeze_qparams() propagation to children
    def test_lifecycle_propagation(self):
        fp = DummySelfAttn(self.E, self.H)
        attn = QuantFairseqMultiheadAttention(fp, qcfg=self.qcfg, fp_name="mha")
        self.assertEqual(attn._mode, Mode.NO_QUANT)

        attn.enable_calibration()
        self.assertEqual(attn._mode, Mode.CALIB)
        for _, obs in attn.named_observers():
            self.assertTrue(getattr(obs, "enabled", False))

        attn.freeze_qparams()
        self.assertEqual(attn._mode, Mode.QUANT)
        for _, obs in attn.named_observers():
            self.assertFalse(getattr(obs, "enabled", True))

        q, _, _ = self._make_inputs()
        out, _ = attn(q, None, None)
        self.assertEqual(out.shape, (self.T, self.B, self.E))

    # 8) Observer enumeration and direct lookup
    def test_observer_lookup_and_uniqueness(self):
        fp = DummySelfAttn(self.E, self.H)
        attn = QuantFairseqMultiheadAttention(fp, qcfg=self.qcfg, fp_name="mha")

        names = []
        for name, obs in attn.named_observers():
            self.assertIsNotNone(obs)
            names.append(name)
            self.assertIs(attn.get_observer(name), obs)

        self.assertEqual(len(names), len(set(names)))
