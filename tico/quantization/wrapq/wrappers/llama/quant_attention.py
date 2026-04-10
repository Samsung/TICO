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

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaAttentionDecodeExportAdapter,
    LlamaAttentionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.llama.modeling_llama.LlamaAttention",
    "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
)
class QuantLlamaAttention(QuantModuleBase):
    """
    Unified quantized Llama attention wrapper.

    A single HF-compatible forward is used for both runtime modes:
      - prefill: `past_key_value is None`
      - decode : `past_key_value is not None`

    Export specialization is provided by thin adapter modules.

    Behavior
    --------
    - If `past_key_value` is None, this behaves like a regular prefill step.
    - If `past_key_value` is not None, current K/V are concatenated to the past.
    - If `use_cache=True`, the returned cache is always the full present K/V.
      Export adapters may post-process this into a delta-only form if needed.
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

        assert hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads")
        assert hasattr(cfg, "num_key_value_heads") and hasattr(
            cfg, "max_position_embeddings"
        )
        assert isinstance(cfg.hidden_size, int)
        assert isinstance(cfg.num_attention_heads, int)
        assert isinstance(cfg.num_key_value_heads, int)
        assert isinstance(cfg.max_position_embeddings, int)

        self.head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.kv_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        self.max_seq = cfg.max_position_embeddings

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
            fp_attn.q_proj, qcfg=q_cfg, fp_name=f"{fp_name}.q_proj" if fp_name else None
        )
        self.k_proj = PTQWrapper(
            fp_attn.k_proj, qcfg=k_cfg, fp_name=f"{fp_name}.k_proj" if fp_name else None
        )
        self.v_proj = PTQWrapper(
            fp_attn.v_proj, qcfg=v_cfg, fp_name=f"{fp_name}.v_proj" if fp_name else None
        )
        self.o_proj = PTQWrapper(
            fp_attn.o_proj, qcfg=o_cfg, fp_name=f"{fp_name}.o_proj" if fp_name else None
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

        # kv cache
        self.obs_past_key = mk("past_key")
        self.obs_past_value = mk("past_value")

        # New kv delta``
        self.obs_new_k = mk("new_k")  # (B, n_kv, 1, H)
        self.obs_new_v = mk("new_v")  # (B, n_kv, 1, H)

        # Total KV after concat (used for matmul/attn)
        self.obs_present_key = mk("present_key")  # (B, max_seq, H)
        self.obs_present_value = mk("present_value")  # (B, max_seq, H)

        # Static causal mask template
        mask = torch.full((1, 1, self.max_seq, self.max_seq), float("-120"))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

    def _rot(self, t: torch.Tensor, o_x1, o_x2, o_cat):
        x1, x2 = torch.chunk(t, 2, dim=-1)
        x1 = self._fq(x1, o_x1)
        x2 = self._fq(x2, o_x2)
        return self._fq(torch.cat((x2, x1), dim=-1), o_cat)

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
        t_half = self._rot(t, obs_x1, obs_x2, obs_cat)
        t_cos = self._fq(t * cos, obs_cos)
        t_sin = self._fq(t_half * sin, obs_sin)
        return self._fq(t_cos + t_sin, obs_rot)

    def _concat_kv(
        self,
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        kv_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concat along sequence dim for one kv head."""
        if past is None:
            return k_new, v_new
        past_k, past_v = past
        past_k = self._fq(past_k, self.obs_past_key)
        past_v = self._fq(past_v, self.obs_past_value)
        k = torch.cat([past_k[:, kv_idx, :, :], k_new], dim=1)
        v = torch.cat([past_v[:, kv_idx, :, :], v_new], dim=1)
        return k, v

    def _build_attention_mask(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return an additive attention mask usable with per-head logits.

        Supported cases:
        - None: build a causal mask slice.
        - bool mask: convert to additive mask using 0 / -120.
        - additive mask: use as-is.
        """
        q_len = hidden_states.size(1)
        past_len = 0 if past_key_value is None else int(past_key_value[0].shape[2])
        k_len = past_len + q_len

        if attention_mask is None:
            assert isinstance(self.causal_mask_template, torch.Tensor)
            mask = self.causal_mask_template[..., :q_len, :k_len].to(device)
            mask = mask.squeeze(0)
            return self._fq(mask, self.obs_attn_mask)

        if attention_mask.dtype == torch.bool:
            additive = torch.zeros_like(attention_mask, dtype=torch.float32)
            additive = additive.masked_fill(~attention_mask, float("-120"))
            return self._fq(additive, self.obs_attn_mask)

        return self._fq(attention_mask, self.obs_attn_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        hidden = self._fq(hidden_states, self.obs_hidden)
        B, S, _ = hidden.shape
        H = self.head_dim

        q = self.q_proj(hidden).view(B, S, -1, H)  # (B, S, n_h, H)
        k = self.k_proj(hidden).view(B, S, -1, H)  # (B, K, n_kv, H)
        v = self.v_proj(hidden).view(B, S, -1, H)  # (B, K, n_kv, H)

        # Rope tables
        cos, sin = position_embeddings
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)

        past_len = 0 if past_key_value is None else int(past_key_value[0].shape[2])
        key_len = past_len + S

        attn_mask = self._build_attention_mask(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            device=hidden.device,
        )

        attn_weights_parts: List[torch.Tensor] = []
        attn_out_parts: List[torch.Tensor] = []
        new_k_parts: List[torch.Tensor] = []
        new_v_parts: List[torch.Tensor] = []

        for kv_i in range(self.num_kv_heads):
            # k_h, v_h: (B, K, H)
            k_i_new = k[:, :, kv_i, :]
            v_i_new = v[:, :, kv_i, :]

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

            k_i, v_i = self._concat_kv(past_key_value, k_i_new, v_i_new, kv_i)
            k_i = self._fq(k_i, self.obs_present_key)
            v_i = self._fq(v_i, self.obs_present_value)

            for rep_i in range(self.kv_rep):
                q_idx = kv_i * self.kv_rep + rep_i
                # q_h: (B, S, H)
                q_i = q[:, :, q_idx, :]
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

                # logits: (B, S, K)
                logits_i = self._fq(q_i @ k_i.transpose(-2, -1), self.obs_logits)

                assert attn_mask.shape[-2:] == logits_i.shape[-2:], (
                    attn_mask.shape,
                    logits_i.shape,
                )
                logits_i = self._fq(logits_i + attn_mask, self.obs_mask_add)

                # softmax
                attn_i = torch.softmax(logits_i, dim=-1, dtype=torch.float32).to(
                    q_i.dtype
                )
                attn_i = self._fq(attn_i, self.obs_softmax)

                # out: (B, S, H)
                out_i = self._fq(attn_i @ v_i, self.obs_attn_out)

                attn_weights_parts.append(attn_i)
                attn_out_parts.append(out_i)

        # Concat heads back
        # (B, n_h, S, K)
        attn_weights = self._fq(
            torch.stack(attn_weights_parts, dim=1), self.obs_attn_weights
        )
        # (B, n_h, S, H)
        attn_out_h = self._fq(torch.stack(attn_out_parts, dim=1), self.obs_attn_out_h)
        # Attention output: (B, S, n_h * H)
        attn_out = attn_out_h.transpose(1, 2).reshape(B, S, -1)
        # Final projection: (B, 1, D)
        out = self.o_proj(attn_out)

        outputs = (out, attn_weights)

        if use_cache:
            # New kv delta: (B, n_kv, S, H)
            new_k = torch.stack(new_k_parts, dim=1)
            new_v = torch.stack(new_v_parts, dim=1)
            new_k = self._fq(new_k, self.obs_new_k)
            new_v = self._fq(new_v, self.obs_new_v)
            outputs += ((new_k, new_v),)  # type: ignore[assignment]

        return outputs

    def _all_observers(self):
        # local first
        yield from (
            self.obs_hidden,
            self.obs_cos,
            self.obs_sin,
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
            self.obs_attn_mask,
            self.obs_logits,
            self.obs_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
            self.obs_attn_weights,
            self.obs_attn_out_h,
            self.obs_past_key,
            self.obs_past_value,
            self.obs_new_k,
            self.obs_new_v,
            self.obs_present_key,
            self.obs_present_value,
        )
        # recurse into children that are QuantModuleBase
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            yield from m._all_observers()

    def as_export_module(
        self, mode: ExportMode = "prefill", *, return_kv: bool = True
    ) -> nn.Module:
        if mode == "prefill":
            return LlamaAttentionPrefillExportAdapter(self, return_kv=return_kv)
        if mode == "decode":
            return LlamaAttentionDecodeExportAdapter(self, return_kv=return_kv)
        raise ValueError(f"Unsupported export mode: {mode!r}")
