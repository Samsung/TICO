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

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.experimental.quantization.custom.dtypes import DType
from tico.experimental.quantization.custom.observers import MinMaxObserver
from tico.experimental.quantization.custom.qscheme import QScheme
from tico.experimental.quantization.custom.wrappers.mode import Mode
from tico.experimental.quantization.custom.wrappers.ptq_wrapper import PTQWrapper


class QuantLlamaAttention(nn.Module):
    def __init__(self, attn_fp32: nn.Module):
        super().__init__()
        cfg = attn_fp32.config
        self.h_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        self.kv_rep = cfg.num_attention_heads // cfg.num_key_value_heads

        scale = self.h_dim**-0.5
        self.scale_t = torch.tensor(scale)
        self.obs_scale = MinMaxObserver(dtype=DType.uint(8))

        # ---- wrap Linear layers ------------------------------------
        def wrap_lin(m: nn.Linear) -> PTQWrapper:
            return PTQWrapper(
                module=m,
                act_obs=MinMaxObserver(dtype=DType.uint(8)),
                weight_obs=MinMaxObserver(
                    dtype=DType.uint(8),
                    qscheme=QScheme.PER_CHANNEL_AFFINE,
                    channel_axis=0,
                ),
            )

        self.q_proj = wrap_lin(attn_fp32.q_proj)
        self.k_proj = wrap_lin(attn_fp32.k_proj)
        self.v_proj = wrap_lin(attn_fp32.v_proj)
        self.o_proj = wrap_lin(attn_fp32.o_proj)

        # ---- observers for EACH arithmetic result ------------------
        self.obs_hidden = MinMaxObserver(dtype=DType.uint(8))

        self.obs_cos = MinMaxObserver(dtype=DType.uint(8))
        self.obs_sin = MinMaxObserver(dtype=DType.uint(8))

        # rotate-half sub-steps
        self.obs_q_x1 = MinMaxObserver(dtype=DType.uint(8))
        self.obs_q_x2 = MinMaxObserver(dtype=DType.uint(8))
        self.obs_q_x2neg = MinMaxObserver(dtype=DType.uint(8))
        self.obs_q_cat = MinMaxObserver(dtype=DType.uint(8))

        self.obs_k_x1 = MinMaxObserver(dtype=DType.uint(8))
        self.obs_k_x2 = MinMaxObserver(dtype=DType.uint(8))
        self.obs_k_x2neg = MinMaxObserver(dtype=DType.uint(8))
        self.obs_k_cat = MinMaxObserver(dtype=DType.uint(8))

        # q path
        self.obs_q_cos = MinMaxObserver(dtype=DType.uint(8))
        self.obs_q_sin = MinMaxObserver(dtype=DType.uint(8))
        self.obs_q_rot = MinMaxObserver(dtype=DType.uint(8))

        # k path
        self.obs_k_cos = MinMaxObserver(dtype=DType.uint(8))
        self.obs_k_sin = MinMaxObserver(dtype=DType.uint(8))
        self.obs_k_rot = MinMaxObserver(dtype=DType.uint(8))

        # attention logits, softmax, attn_out
        self.obs_logits_raw = MinMaxObserver(dtype=DType.uint(8))
        self.obs_logits = MinMaxObserver(dtype=DType.uint(8))
        self.obs_sm = MinMaxObserver(dtype=DType.uint(8))
        self.obs_attnout = MinMaxObserver(dtype=DType.uint(8))

        self._mode: Mode = Mode.NO_QUANT

    def _rot(self, t, obs_x1, obs_x2, obs_neg, obs_cat):
        x1, x2 = torch.chunk(t, 2, dim=-1)
        x1 = self._fq(x1, obs_x1)
        x2 = self._fq(x2, obs_x2)
        x2n = self._fq(-x2, obs_neg)
        cat = self._fq(torch.cat((x2n, x1), dim=-1), obs_cat)
        return cat

    def _fq(self, x: torch.Tensor, obs: MinMaxObserver):  # fake / collect
        if self._mode is Mode.CALIB:
            obs.collect(x.detach())
            return x
        if self._mode is Mode.QUANT:
            return obs.fake_quant(x)
        return x

    def enable_calibration(self):
        self._mode = Mode.CALIB
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            mod.enable_calibration()
        for ob in self._all_obs():
            ob.enabled, _ = True, ob.reset()

    def freeze_qparams(self):
        self._mode = Mode.QUANT
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            mod.freeze_qparams()
        for ob in self._all_obs():
            ob.enabled = False
            ob.compute_qparams()

    def _all_obs(self):
        return (
            self.obs_hidden,
            self.obs_scale,
            self.obs_cos,
            self.obs_sin,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_x2neg,
            self.obs_q_cat,
            self.obs_k_x1,
            self.obs_k_x2,
            self.obs_k_x2neg,
            self.obs_k_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
            self.obs_k_cos,
            self.obs_k_sin,
            self.obs_k_rot,
            self.obs_logits,
            self.obs_logits_raw,
            self.obs_sm,
            self.obs_attnout,
        )

    def forward(self, hidden: torch.Tensor, pos_emb: tuple[torch.Tensor, torch.Tensor]):

        # --- 1) quant / collect input --------------------------------
        hidden = self._fq(hidden, self.obs_hidden)

        B, S, _ = hidden.shape
        H = self.h_dim

        # --- 2) projections -----------------------------------------
        q = self.q_proj(hidden).view(B, S, -1, H).transpose(1, 2)  # [B,h,S,H]
        k = self.k_proj(hidden).view(B, S, -1, H).transpose(1, 2)
        v = self.v_proj(hidden).view(B, S, -1, H).transpose(1, 2)

        # --- 3) RoPE cos/sin  ---------------------------------------
        cos, sin = pos_emb  # [B,S,H]
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)

        cos_u = cos.unsqueeze(1)  # broadcast to heads
        sin_u = sin.unsqueeze(1)

        # --- 4) q_rot -----------------------------------------------
        q_half = self._rot(
            q, self.obs_q_x1, self.obs_q_x2, self.obs_q_x2neg, self.obs_q_cat
        )
        q_cos = self._fq(q * cos_u, self.obs_q_cos)
        q_sin = self._fq(q_half * sin_u, self.obs_q_sin)
        q_rot = self._fq(q_cos + q_sin, self.obs_q_rot)

        # --- 5) k_rot -----------------------------------------------
        k_half = self._rot(
            k, self.obs_k_x1, self.obs_k_x2, self.obs_k_x2neg, self.obs_k_cat
        )
        k_cos = self._fq(k * cos_u, self.obs_k_cos)
        k_sin = self._fq(k_half * sin_u, self.obs_k_sin)
        k_rot = self._fq(k_cos + k_sin, self.obs_k_rot)

        # --- 6) logits ----------------------------------------------
        k_rep = k_rot.repeat_interleave(self.kv_rep, dim=1)
        logits_raw = self._fq(q_rot @ k_rep.transpose(-2, -1), self.obs_logits_raw)
        scale = self._fq(self.scale_t, self.obs_scale)
        logits = self._fq(logits_raw * scale, self.obs_logits)

        # --- 7) soft-max (FP32 kernel but INT8 output) --------------
        sm = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        sm = self._fq(sm, self.obs_sm)

        # --- 8) attn_out --------------------------------------------
        v_rep = v.repeat_interleave(self.kv_rep, dim=1)
        attn_out = (
            self._fq(sm @ v_rep, self.obs_attnout).transpose(1, 2).reshape(B, S, -1)
        )

        # --- 9) final projection (PTQWrapper) -----------------------
        return self.o_proj(attn_out)

    # ----------------------------------------------------------------
    def extra_repr(self):
        return f"mode={self._mode.name.lower()}"
