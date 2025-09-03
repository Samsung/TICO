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

from typing import Optional

import torch
import torch.nn as nn

from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.ptq.wrappers.registry import try_register


@try_register(
    "transformers.models.llama.modeling_llama.LlamaAttention",
    "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
)
class QuantLlamaAttention(QuantModuleBase):
    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[QuantConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        cfg = fp_attn.config
        assert hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads")
        assert hasattr(cfg, "num_key_value_heads")
        assert isinstance(cfg.hidden_size, int) and isinstance(
            cfg.num_attention_heads, int
        )
        assert isinstance(cfg.num_key_value_heads, int)
        self.hdim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.kv_rep = cfg.num_attention_heads // cfg.num_key_value_heads

        # constant scale (1/√d)
        self.scale_t = torch.tensor(self.hdim**-0.5)
        self.obs_scale = self._make_obs("scale")

        # ---- wrap q k v o projections via PTQWrapper ---------------
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
            fp_attn.k_proj, qcfg=k_cfg, fp_name=f"{fp_name}.k_proj"
        )
        self.v_proj = PTQWrapper(
            fp_attn.v_proj, qcfg=v_cfg, fp_name=f"{fp_name}.v_proj"
        )
        self.o_proj = PTQWrapper(
            fp_attn.o_proj, qcfg=o_cfg, fp_name=f"{fp_name}.o_proj"
        )

        # ---- create arithmetic observers ---------------------------
        mk = self._make_obs
        self.obs_hidden = mk("hidden")

        self.obs_cos = mk("cos")
        self.obs_sin = mk("sin")

        self.obs_causal_mask = mk("causal_mask")

        # rotate-half sub-steps
        self.obs_q_x1 = mk("q_x1")
        self.obs_q_x2 = mk("q_x2")
        self.obs_q_neg = mk("q_neg")
        self.obs_q_cat = mk("q_cat")
        self.obs_k_x1 = mk("k_x1")
        self.obs_k_x2 = mk("k_x2")
        self.obs_k_neg = mk("k_neg")
        self.obs_k_cat = mk("k_cat")

        # q / k paths
        self.obs_q_cos = mk("q_cos")
        self.obs_q_sin = mk("q_sin")
        self.obs_q_rot = mk("q_rot")
        self.obs_k_cos = mk("k_cos")
        self.obs_k_sin = mk("k_sin")
        self.obs_k_rot = mk("k_rot")

        # logits / softmax / out
        self.obs_logits_raw = mk("logits_raw")
        self.obs_logits = mk("logits")
        self.obs_mask_add = mk("mask_add")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")

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
        x2n = self._fq(-x2, o_neg)
        return self._fq(torch.cat((x2n, x1), -1), o_cat)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,  # not supported yet
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if past_key_value is not None:
            raise NotImplementedError(
                "QuantLlamaAttention does not support KV cache yet."
            )

        hidden = self._fq(hidden_states, self.obs_hidden)
        B, S, _ = hidden.shape
        H = self.hdim

        # projections
        q = self.q_proj(hidden).view(B, S, -1, H).transpose(1, 2)
        k = self.k_proj(hidden).view(B, S, -1, H).transpose(1, 2)
        v = self.v_proj(hidden).view(B, S, -1, H).transpose(1, 2)

        # rope tables
        cos, sin = position_embeddings
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)
        cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)

        # q_rot
        q_half = self._rot(
            q, self.obs_q_x1, self.obs_q_x2, self.obs_q_neg, self.obs_q_cat
        )
        q_cos = self._fq(q * cos_u, self.obs_q_cos)
        q_sin = self._fq(q_half * sin_u, self.obs_q_sin)
        q_rot = self._fq(q_cos + q_sin, self.obs_q_rot)

        # k_rot
        k_half = self._rot(
            k, self.obs_k_x1, self.obs_k_x2, self.obs_k_neg, self.obs_k_cat
        )
        k_cos = self._fq(k * cos_u, self.obs_k_cos)
        k_sin = self._fq(k_half * sin_u, self.obs_k_sin)
        k_rot = self._fq(k_cos + k_sin, self.obs_k_rot)

        # logits
        k_rep = k_rot.repeat_interleave(self.kv_rep, dim=1)
        logits_raw = self._fq(q_rot @ k_rep.transpose(-2, -1), self.obs_logits_raw)
        scale = self._fq(self.scale_t, self.obs_scale)
        logits = self._fq(logits_raw * scale, self.obs_logits)

        if attention_mask is None or attention_mask.dtype == torch.bool:
            _, _, q_len, k_len = logits.shape
            assert isinstance(self.causal_mask_template, torch.Tensor)
            attention_mask = self.causal_mask_template[..., :q_len, :k_len].to(
                hidden_states.device
            )
        attention_mask = self._fq(attention_mask, self.obs_causal_mask)
        logits = self._fq(logits + attention_mask, self.obs_mask_add)

        # softmax
        attn_weights = torch.softmax(logits, -1, dtype=torch.float32).to(q.dtype)
        attn_weights = self._fq(attn_weights, self.obs_softmax)

        # attn out
        v_rep = v.repeat_interleave(self.kv_rep, dim=1)
        attn_out = (
            self._fq(attn_weights @ v_rep, self.obs_attn_out)
            .transpose(1, 2)
            .reshape(B, S, -1)
        )

        # final projection
        return self.o_proj(attn_out), attn_weights

    def _all_observers(self):
        # local first
        yield from (
            self.obs_hidden,
            self.obs_scale,
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
            self.obs_logits_raw,
            self.obs_logits,
            self.obs_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
        )
        # recurse into children that are QuantModuleBase
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            yield from m._all_observers()
