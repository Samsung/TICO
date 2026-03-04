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

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.llama.modeling_llama.LlamaAttention",
    "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
    variant="decode",
)
class QuantLlamaAttentionDecode(QuantModuleBase):
    """
    Decode-only attention wrapper with fully static shapes.

    Expected static shapes:
      hidden_states:      (B, 1, D)
      position_embeddings:
        cos:              (B, 1, head_dim)
        sin:              (B, 1, head_dim)
      attention_mask:     (B, 1, max_seq)    # (0 or -120)
      past_key_value:
        past_key:         (B, n_kv, max_seq-1, head_dim)  # already RoPE-applied
        past_value:       (B, n_kv, max_seq-1, head_dim)

    Outputs (when use_cache=True):
      out:                (B, 1, D)
      new_kv (delta):
        new_key:          (B, n_kv, 1, head_dim)          # RoPE-applied
        new_value:        (B, n_kv, 1, head_dim)
    """

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        cfg = fp_attn.config
        self.config = cfg

        # head shapes
        assert hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads")
        assert hasattr(cfg, "num_key_value_heads")
        assert isinstance(cfg.hidden_size, int)
        assert isinstance(cfg.num_attention_heads, int)
        assert isinstance(cfg.num_key_value_heads, int)

        self.head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        self.kv_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        self.n_kv = cfg.num_key_value_heads
        self.n_h = cfg.num_attention_heads

        assert hasattr(cfg, "max_position_embeddings")
        assert isinstance(cfg.max_position_embeddings, int)
        self.max_seq = cfg.max_position_embeddings

        # ---- Wrap q k v o projections via PTQWrapper ---------------
        q_cfg = qcfg.child("q_proj") if qcfg else None
        k_cfg = qcfg.child("k_proj") if qcfg else None
        v_cfg = qcfg.child("v_proj") if qcfg else None
        o_cfg = qcfg.child("o_proj") if qcfg else None

        assert hasattr(fp_attn, "q_proj") and isinstance(
            fp_attn.q_proj, torch.nn.Module
        )
        assert hasattr(fp_attn, "k_proj") and isinstance(
            fp_attn.k_proj, torch.nn.Module
        )
        assert hasattr(fp_attn, "v_proj") and isinstance(
            fp_attn.v_proj, torch.nn.Module
        )
        assert hasattr(fp_attn, "o_proj") and isinstance(
            fp_attn.o_proj, torch.nn.Module
        )

        self.q_proj = PTQWrapper(
            fp_attn.q_proj, qcfg=q_cfg, fp_name=f"{fp_name}.q_proj"
        )
        self.k_proj = PTQWrapper(
            copy.deepcopy(fp_attn.k_proj), qcfg=k_cfg, fp_name=f"{fp_name}.k_proj"
        )
        self.v_proj = PTQWrapper(
            fp_attn.v_proj, qcfg=v_cfg, fp_name=f"{fp_name}.v_proj"
        )
        self.o_proj = PTQWrapper(
            fp_attn.o_proj, qcfg=o_cfg, fp_name=f"{fp_name}.o_proj"
        )

        # Constant scale (1/√d) folded into k_proj
        scale_t = torch.tensor(
            float(getattr(fp_attn, "scaling", self.head_dim**-0.5))
        )
        with torch.no_grad():
            lin = self.k_proj.wrapped.module
            lin.weight.mul_(scale_t)
            if lin.bias is not None:
                lin.bias.mul_(scale_t)

        mk = self._make_obs
        self.obs_hidden = mk("hidden")

        # RoPE tables (for current token only)
        self.obs_cos = mk("cos")
        self.obs_sin = mk("sin")

        # rotate_half sub-steps (q)
        self.obs_q_x1 = mk("q_x1")
        self.obs_q_x2 = mk("q_x2")
        self.obs_q_cat = mk("q_cat")

        # rotate_half sub-steps (k)
        self.obs_k_x1 = mk("k_x1")
        self.obs_k_x2 = mk("k_x2")
        self.obs_k_cat = mk("k_cat")

        # RoPE combine
        self.obs_q_cos = mk("q_cos")
        self.obs_q_sin = mk("q_sin")
        self.obs_q_rot = mk("q_rot")
        self.obs_k_cos = mk("k_cos")
        self.obs_k_sin = mk("k_sin")
        self.obs_k_rot = mk("k_rot")

        # Masking & attention math
        self.obs_attn_mask = mk("attn_mask")
        self.obs_logits = mk("logits")
        self.obs_mask_add = mk("mask_add")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")
        self.obs_attn_weights = mk("attn_weights")
        self.obs_attn_out_h = mk("attn_out_h")

    def _rot(self, t: torch.Tensor, o_x1, o_x2, o_cat):
        # t: (..., head_dim)
        x1, x2 = torch.chunk(t, 2, dim=-1)
        x1 = self._fq(x1, o_x1)
        x2 = self._fq(x2, o_x2)
        x2n = x2
        return self._fq(torch.cat((x2n, x1), dim=-1), o_cat)

    def _apply_rope(
        self,
        t: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        obs_x1,
        obs_x2,
        obs_cat,
        obs_cos,
        obs_sin,
        obs_rot,
    ):
        # t: (B, 1, head_dim)
        t_half = self._rot(t, obs_x1, obs_x2, obs_cat)
        t_cos = self._fq(t * cos, obs_cos)
        t_sin = self._fq(t_half * sin, obs_sin)
        t_rot = self._fq(t_cos + t_sin, obs_rot)
        return t_rot

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, 1, D)
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (B, 1, H), (B, 1, H)
        attention_mask: torch.Tensor,  # (B, 1, max_seq)
        past_key_value: Tuple[torch.Tensor, torch.Tensor],  # (B, n_kv, max_seq-1, H)
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        hidden = self._fq(hidden_states, self.obs_hidden)
        assert hidden.dim() == 3 and hidden.size(1) == 1, "Decode expects (B,1,D)"
        B, _, _ = hidden.shape
        H = self.head_dim
        max_seq = self.max_seq

        past_k, past_v = past_key_value
        # Past KV must be static-sized and already RoPE-applied.
        assert past_k.shape == (B, self.n_kv, max_seq - 1, H), past_k.shape
        assert past_v.shape == (B, self.n_kv, max_seq - 1, H), past_v.shape

        # attention_mask: (0 for keep, -120 for mask)
        assert attention_mask is not None, "Decode expects attention_mask input"
        assert attention_mask.shape == (B, 1, max_seq), attention_mask.shape
        assert (
            attention_mask.dtype != torch.bool
        ), "Please pass additive int mask, not bool"
        attn_mask = self._fq(attention_mask, self.obs_attn_mask)

        # Projections (q_len=1)
        q = self.q_proj(hidden).view(B, 1, -1, H)  # (B, 1, n_h, H)
        k_new = self.k_proj(hidden).view(B, 1, -1, H)  # (B, 1, n_kv, H)
        v_new = self.v_proj(hidden).view(B, 1, -1, H)  # (B, 1, n_kv, H)

        # RoPE (current token only)
        cos, sin = position_embeddings
        assert cos.shape == (B, 1, H) and sin.shape == (B, 1, H)
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)

        attn_weights_parts: List[torch.Tensor] = []
        attn_out_parts: List[torch.Tensor] = []

        new_k_parts: List[torch.Tensor] = []
        new_v_parts: List[torch.Tensor] = []

        for kv_i in range(self.n_kv):
            # new K/V for this kv head: (B,1,H)
            k_i_new = k_new[:, :, kv_i, :]
            v_i_new = v_new[:, :, kv_i, :]

            k_i_new = self._apply_rope(
                k_i_new,
                cos,
                sin,
                self.obs_k_x1,
                self.obs_k_x2,
                self.obs_k_cat,
                self.obs_k_cos,
                self.obs_k_sin,
                self.obs_k_rot,
            )
            new_k_parts.append(k_i_new)
            new_v_parts.append(v_i_new)

            # Build total KV: (B, max_seq, H)
            # past: (B, max_seq-1, H)
            k_i_past = past_k[:, kv_i, :, :]
            v_i_past = past_v[:, kv_i, :, :]
            k_i = torch.cat([k_i_past, k_i_new], dim=1)
            v_i = torch.cat([v_i_past, v_i_new], dim=1)

            for rep_i in range(self.kv_rep):
                q_idx = kv_i * self.kv_rep + rep_i
                q_i = q[:, :, q_idx, :]  # (B, 1, H)

                q_i = self._apply_rope(
                    q_i,
                    cos,
                    sin,
                    self.obs_q_x1,
                    self.obs_q_x2,
                    self.obs_q_cat,
                    self.obs_q_cos,
                    self.obs_q_sin,
                    self.obs_q_rot,
                )

                # logits: (B, 1, max_seq)
                logits_i = self._fq(q_i @ k_i.transpose(-2, -1), self.obs_logits)

                # mask add: (B, 1, max_seq)
                logits_i = self._fq(logits_i + attn_mask, self.obs_mask_add)

                # softmax
                attn_i = torch.softmax(logits_i, dim=-1, dtype=torch.float32).to(
                    q_i.dtype
                )
                attn_i = self._fq(attn_i, self.obs_softmax)

                # out: (B, 1, H)
                out_i = self._fq(attn_i @ v_i, self.obs_attn_out)

                attn_weights_parts.append(attn_i)  # (B, 1, max_seq)
                attn_out_parts.append(out_i)  # (B, 1, H)

        # Stack heads back
        # attn_out_h: (B, n_h, 1, H)
        attn_out_h = self._fq(torch.stack(attn_out_parts, dim=1), self.obs_attn_out_h)

        # Merge heads: (B, 1, n_h*H)
        attn_out = attn_out_h.transpose(1, 2).reshape(B, 1, -1)

        # Final projection: (B, 1, D)
        out = self.o_proj(attn_out)

        # new kv delta: (B, n_kv, 1, H)
        new_k = torch.stack(new_k_parts, dim=1)
        new_v = torch.stack(new_v_parts, dim=1)
        new_key_value = (new_k, new_v)

        if use_cache:
            return out, new_key_value
        else:
            return out

    def _all_observers(self):
        yield from (
            self.obs_hidden,
            self.obs_cos,
            self.obs_sin,
            self.obs_attn_mask,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_cat,
            self.obs_k_x1,
            self.obs_k_x2,
            self.obs_k_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
            self.obs_k_cos,
            self.obs_k_sin,
            self.obs_k_rot,
            self.obs_logits,
            self.obs_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
            self.obs_attn_weights,
            self.obs_attn_out_h,
        )
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            yield from m._all_observers()
