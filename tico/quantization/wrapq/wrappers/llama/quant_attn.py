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

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.llama.modeling_llama.LlamaAttention",
    "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
)
class QuantLlamaAttention(QuantModuleBase):
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
        assert isinstance(cfg.hidden_size, int) and isinstance(
            cfg.num_attention_heads, int
        )
        assert isinstance(cfg.num_key_value_heads, int)
        self.head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        self.kv_rep = cfg.num_attention_heads // cfg.num_key_value_heads

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

        # Constant scale (1/√d)
        scale_t = torch.tensor(
            float(getattr(fp_attn, "scaling", self.head_dim**-0.5))
        )
        # merge scale_t to k_proj, (otherwise merge it to q_proj)
        with torch.no_grad():
            lin = self.k_proj.wrapped.module
            lin.weight.mul_(scale_t)
            if lin.bias is not None:
                lin.bias.mul_(scale_t)

        mk = self._make_obs
        self.obs_hidden = mk("hidden")

        # RoPE tables
        self.obs_cos = mk("cos")
        self.obs_sin = mk("sin")

        # rotate_half sub-steps (q)
        self.obs_q_x1 = mk("q_x1")
        self.obs_q_x2 = mk("q_x2")
        self.obs_q_neg = mk("q_neg")
        self.obs_q_cat = mk("q_cat")

        # rotate_half sub-steps (k)
        self.obs_k_x1 = mk("k_x1")
        self.obs_k_x2 = mk("k_x2")
        self.obs_k_neg = mk("k_neg")
        self.obs_k_cat = mk("k_cat")

        # RoPE combine
        self.obs_q_cos = mk("q_cos")
        self.obs_q_sin = mk("q_sin")
        self.obs_q_rot = mk("q_rot")
        self.obs_k_cos = mk("k_cos")
        self.obs_k_sin = mk("k_sin")
        self.obs_k_rot = mk("k_rot")

        # Masking & attention math
        self.obs_causal_mask = mk("causal_mask")
        self.obs_logits = mk("logits")
        self.obs_mask_add = mk("mask_add")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")
        self.obs_attn_weights = mk("attn_weights")
        self.obs_attn_out_h = mk("attn_out_h")

        # Static causal mask template
        assert hasattr(cfg, "max_position_embeddings")
        max_seq = cfg.max_position_embeddings
        mask = torch.full((1, 1, max_seq, max_seq), float("-120"))  # type: ignore[arg-type]
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

    def _rot(self, t, o_x1, o_x2, o_neg, o_cat):
        x1, x2 = torch.chunk(t, 2, dim=-1)
        x1 = self._fq(x1, o_x1)
        x2 = self._fq(x2, o_x2)
        x2n = x2
        return self._fq(torch.cat((x2n, x1), -1), o_cat)

    @staticmethod
    def _concat_kv(
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concat along sequence dim (dim=2): (B, n_kv, S, H)."""
        if past is None:
            return k_new, v_new
        past_k, past_v = past
        k = torch.cat([past_k, k_new], dim=2)
        v = torch.cat([past_v, v_new], dim=2)
        return k, v

    def _apply_rope(self, q, k, cos, sin, unsqueeze_dim: int = 1):
        cos_u = cos.unsqueeze(unsqueeze_dim)
        sin_u = sin.unsqueeze(unsqueeze_dim)

        q_half = self._rot(
            q, self.obs_q_x1, self.obs_q_x2, self.obs_q_neg, self.obs_q_cat
        )
        q_cos = self._fq(q * cos_u, self.obs_q_cos)
        q_sin = self._fq(q_half * sin_u, self.obs_q_sin)
        q_rot = self._fq(q_cos + q_sin, self.obs_q_rot)

        k_half = self._rot(
            k, self.obs_k_x1, self.obs_k_x2, self.obs_k_neg, self.obs_k_cat
        )
        k_cos = self._fq(k * cos_u, self.obs_k_cos)
        k_sin = self._fq(k_half * sin_u, self.obs_k_sin)
        k_rot = self._fq(k_cos + k_sin, self.obs_k_rot)

        return q_rot, k_rot

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,  # tuple(k, v) or HF Cache-like object
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        hidden = self._fq(hidden_states, self.obs_hidden)
        B, S, _ = hidden.shape
        H = self.head_dim

        # Projections
        q = self.q_proj(hidden).view(B, S, -1, H).transpose(1, 2)  # (B, n_h, S, H)
        k = self.k_proj(hidden).view(B, S, -1, H).transpose(1, 2)  # (B, n_kv, S, H)
        v = self.v_proj(hidden).view(B, S, -1, H).transpose(1, 2)  # (B, n_kv, S, H)

        # Rope tables
        cos, sin = position_embeddings
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)
        q_rot, k_rot = self._apply_rope(q, k, cos, sin, unsqueeze_dim=1)

        # --- build/update KV for attention & present_key_value -------------
        present_key_value: Tuple[torch.Tensor, torch.Tensor]

        # TODO Revisit cache logic
        # HF Cache path (if available)
        if use_cache and hasattr(past_key_value, "update"):
            k_total, v_total = past_key_value.update(k_rot, v)
            present_key_value = (k_total, v_total)
            k_for_attn, v_for_attn = k_total, v_total
        else:
            # Tuple or None path
            pkv_tuple = past_key_value if isinstance(past_key_value, tuple) else None
            k_for_attn, v_for_attn = self._concat_kv(pkv_tuple, k_rot, v)
            present_key_value = (k_for_attn, v_for_attn)

        # Build causal mask if needed
        if attention_mask is None or attention_mask.dtype == torch.bool:
            q_len = q_rot.size(2)
            k_len = k_for_attn.size(2)
            assert isinstance(self.causal_mask_template, torch.Tensor)
            attention_mask = self.causal_mask_template[..., :q_len, :k_len].to(
                hidden_states.device
            )
        attention_mask = self._fq(attention_mask, self.obs_causal_mask)

        attn_weights_parts = []
        attn_out_parts = []

        n_kv = k_for_attn.size(1)  # num_key_value_heads
        kv_rep = self.kv_rep  # num_key_value_groups

        # TODO Consider attaching a separate observer to each computation.
        for i in range(n_kv):
            # (B, 1, K, H)
            k_i = k_for_attn[:, i : i + 1, :, :]
            v_i = v_for_attn[:, i : i + 1, :, :]

            # (B, G, S, H) where G=kv_rep
            h0 = i * kv_rep
            h1 = (i + 1) * kv_rep
            q_i = q_rot[:, h0:h1, :, :]

            # logits: (B, G, S, K)
            logits_i = self._fq(q_i @ k_i.transpose(-2, -1), self.obs_logits)

            # mask add: broadcast on head axis (1 -> G).
            logits_i = self._fq(logits_i + attention_mask, self.obs_mask_add)

            # softmax
            attn_i = torch.softmax(logits_i, -1, dtype=torch.float32).to(q_i.dtype)
            attn_i = self._fq(attn_i, self.obs_softmax)

            # out: (B, G, S, H)
            out_i = self._fq(attn_i @ v_i, self.obs_attn_out)

            attn_weights_parts.append(attn_i)
            attn_out_parts.append(out_i)

        # concat heads back: (B, n_h, S, K) / (B, n_h, S, H)
        attn_weights = self._fq(
            torch.cat(attn_weights_parts, dim=1), self.obs_attn_weights
        )
        attn_out_h = self._fq(torch.cat(attn_out_parts, dim=1), self.obs_attn_out_h)

        # Attention output
        attn_out = attn_out_h.transpose(1, 2).reshape(B, S, -1)  # (B, S, n_h * H)

        # Final projection
        out = self.o_proj(attn_out)

        # return with/without cache
        if use_cache:
            return out, attn_weights, present_key_value
        else:
            return out, attn_weights

    def _all_observers(self):
        # local first
        yield from (
            self.obs_hidden,
            self.obs_cos,
            self.obs_sin,
            self.obs_causal_mask,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_neg,
            self.obs_q_cat,
            self.obs_k_x1,
            self.obs_k_x2,
            self.obs_k_neg,
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
        # recurse into children that are QuantModuleBase
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            yield from m._all_observers()
