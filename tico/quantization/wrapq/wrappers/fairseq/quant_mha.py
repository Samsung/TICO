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
#
# -----------------------------------------------------------------------------
# This file includes modifications based on fairseq
#  (https://github.com/facebookresearch/fairseq), originally licensed under
# the MIT License. See the LICENSE file in the fairseq repository for details.
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("fairseq.modules.multihead_attention.MultiheadAttention")
class QuantFairseqMultiheadAttention(QuantModuleBase):
    """
    Quant-aware drop-in for Fairseq MultiheadAttention.

    - No xFormers / no torch F.multi_head_attention_forward fast-path.
    - Self/cross attention + minimal incremental KV cache.
    - Causal mask is pre-built statically; `key_padding_mask` is additive float.
    - I/O shape: [T, B, C]

    Runtime optimization flags
    --------------------------
    use_static_causal : bool
        If True, reuse a precomputed upper-triangular causal mask template
        instead of rebuilding it each forward step. Reduces per-step mask
        construction overhead during incremental decoding.

    assume_additive_key_padding : bool
        If True, assume the `key_padding_mask` is already an additive float
        tensor (large negative values at padded positions). Skips conversion
        from boolean masks, reducing runtime overhead.
    """

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
        max_seq: int = 4096,
        use_static_causal: bool = False,
        mask_fill_value: float = -120.0,
        assume_additive_key_padding: bool = False,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        self.use_static_causal = use_static_causal
        self.mask_fill_value = mask_fill_value
        self.assume_additive_key_padding = assume_additive_key_padding
        self.embed_dim: int = int(fp_attn.embed_dim)  # type: ignore[arg-type]
        self.num_heads: int = int(fp_attn.num_heads)  # type: ignore[arg-type]
        self.head_dim: int = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim

        self.self_attention: bool = bool(getattr(fp_attn, "self_attention", False))
        self.encoder_decoder_attention: bool = bool(
            getattr(fp_attn, "encoder_decoder_attention", False)
        )
        assert self.self_attention != self.encoder_decoder_attention

        # PTQ-wrapped projections
        qc = qcfg.child("q_proj") if qcfg else None
        kc = qcfg.child("k_proj") if qcfg else None
        vc = qcfg.child("v_proj") if qcfg else None
        oc = qcfg.child("out_proj") if qcfg else None
        assert hasattr(fp_attn, "q_proj") and hasattr(fp_attn, "k_proj")
        assert hasattr(fp_attn, "v_proj") and hasattr(fp_attn, "out_proj")
        assert isinstance(fp_attn.q_proj, nn.Module) and isinstance(
            fp_attn.k_proj, nn.Module
        )
        assert isinstance(fp_attn.v_proj, nn.Module) and isinstance(
            fp_attn.out_proj, nn.Module
        )
        self.q_proj = PTQWrapper(fp_attn.q_proj, qcfg=qc, fp_name=f"{fp_name}.q_proj")
        self.k_proj = PTQWrapper(fp_attn.k_proj, qcfg=kc, fp_name=f"{fp_name}.k_proj")
        self.v_proj = PTQWrapper(fp_attn.v_proj, qcfg=vc, fp_name=f"{fp_name}.v_proj")
        self.out_proj = PTQWrapper(
            fp_attn.out_proj, qcfg=oc, fp_name=f"{fp_name}.out_proj"
        )

        # scale & static causal mask
        self.register_buffer(
            "scale_const", torch.tensor(self.head_dim**-0.5), persistent=False
        )
        mask = torch.full((1, 1, max_seq, max_seq), float(self.mask_fill_value))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

        # observers (no *_proj_out here; PTQWrapper handles module outputs)
        mk = self._make_obs
        self.obs_query_in = mk("query_in")
        self.obs_key_in = mk("key_in")
        self.obs_value_in = mk("value_in")
        self.obs_kpm_in = mk("kpm_in")
        self.obs_causal_mask = mk("causal_mask")
        self.obs_q_fold = mk("q_fold")
        self.obs_k_fold = mk("k_fold")
        self.obs_v_fold = mk("v_fold")
        self.obs_scale = mk("scale")
        self.obs_logits_raw = mk("logits_raw")
        self.obs_logits = mk("logits_scaled")
        self.obs_attn_mask_add = mk("obs_attn_mask_add")
        self.obs_kp_mask_add = mk("obs_kp_mask_add")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")

        safe_name = (
            fp_name if (fp_name not in (None, "", "None")) else f"QuantFsMHA_{id(self)}"
        )
        assert safe_name is not None
        self._state_key = safe_name + ".attn_state"

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Return saved KV/mask dict or None."""
        if incremental_state is None:
            return None
        return incremental_state.get(self._state_key, None)

    def _set_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        buffer: Dict[str, Optional[torch.Tensor]],
    ):
        """Store KV/mask dict in incremental_state."""
        if incremental_state is not None:
            incremental_state[self._state_key] = buffer
        return incremental_state

    # ---- utils ----
    def _fold_heads(self, x: torch.Tensor, B: int) -> torch.Tensor:
        # [T,B,E] -> [B*H, T, Dh]
        T = x.size(0)
        x = x.view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()
        return x.view(B * self.num_heads, T, self.head_dim)

    def _unfold_heads(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # [B*H, T, Dh] -> [T,B,E]
        x = x.view(B, self.num_heads, T, self.head_dim).permute(2, 0, 1, 3).contiguous()
        return x.view(T, B, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,  # [Tq,B,C]
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[
            torch.Tensor
        ] = None,  # additive float (e.g. -120 at pads)
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[torch.Tensor]]]
        ] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,  # if None -> internal causal
        before_softmax: bool = False,
        need_head_weights: bool = False,
        return_new_kv: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ],
    ]:

        if need_head_weights:
            need_weights = True

        Tq, B, _ = query.shape
        if self.self_attention:
            key = query if key is None else key
            value = query if value is None else value
        else:
            assert key is not None and value is not None

        Tk, Bk, _ = key.shape
        Tv, Bv, _ = value.shape
        assert B == Bk == Bv

        q = self.q_proj(self._fq(query, self.obs_query_in))
        k = self.k_proj(self._fq(key, self.obs_key_in))
        v = self.v_proj(self._fq(value, self.obs_value_in))

        state = self._get_input_buffer(incremental_state)
        if incremental_state is not None and state is None:
            state = {}

        # Capture "new" K/V for this call BEFORE concatenating with cache
        new_k_bh: Optional[torch.Tensor] = None
        new_v_bh: Optional[torch.Tensor] = None

        # Fold heads
        q = self._fq(self._fold_heads(q, B), self.obs_q_fold)
        if state is not None and "prev_key" in state and static_kv:
            # Cross-attention static_kv path: reuse cached KV; there is no new KV this call.
            k = None
            v = None
        if k is not None:
            k = self._fq(self._fold_heads(k, B), self.obs_k_fold)  # [B*H, Tnew, Dh]
            if return_new_kv:
                new_k_bh = k.contiguous()
        if v is not None:
            v = self._fq(self._fold_heads(v, B), self.obs_v_fold)  # [B*H, Tnew, Dh]
            if return_new_kv:
                new_v_bh = v.contiguous()

        # Append/reuse cache
        if state is not None:
            pk = state.get("prev_key")
            pv = state.get("prev_value")
            if pk is not None:
                pk = pk.view(B * self.num_heads, -1, self.head_dim)
                k = pk if static_kv else torch.cat([pk, k], dim=1)
            if pv is not None:
                pv = pv.view(B * self.num_heads, -1, self.head_dim)
                v = pv if static_kv else torch.cat([pv, v], dim=1)

        assert k is not None and v is not None
        Ts = k.size(1)

        # Scaled dot-product
        scale = self._fq(self.scale_const, self.obs_scale).to(q.dtype)
        logits_raw = self._fq(
            torch.bmm(q, k.transpose(1, 2)), self.obs_logits_raw
        )  # [B*H,Tq,Ts]
        logits = self._fq(logits_raw * scale, self.obs_logits)

        assert isinstance(self.causal_mask_template, torch.Tensor)
        # Masks
        device = logits.device
        if attn_mask is None and self.use_static_causal:
            # Incremental decoding aware slicing:
            # align the causal row(s) to the current time indices
            start_q = max(Ts - Tq, 0)
            cm = self.causal_mask_template[..., start_q : start_q + Tq, :Ts].to(
                device=device, dtype=logits.dtype
            )
            attn_mask = cm.squeeze(0).squeeze(0)  # [Tq,Ts]

        if attn_mask is not None:
            # Bool/byte mask -> additive float with large negatives
            if not torch.is_floating_point(attn_mask):
                fill = self.causal_mask_template.new_tensor(self.mask_fill_value)
                attn_mask = torch.where(
                    attn_mask.to(torch.bool), fill, fill.new_zeros(())
                )
            attn_mask = self._fq(attn_mask, self.obs_causal_mask)
            assert isinstance(attn_mask, torch.Tensor)

            if not self.assume_additive_key_padding:
                # attn_mask -> [B*H,Tq,Ts]
                if attn_mask.dim() == 2:
                    add_mask = attn_mask.unsqueeze(0).expand(logits.size(0), -1, -1)
                elif attn_mask.dim() == 3:
                    add_mask = (
                        attn_mask.unsqueeze(1)
                        .expand(B, self.num_heads, Tq, Ts)
                        .contiguous()
                    )
                    add_mask = add_mask.view(B * self.num_heads, Tq, Ts)
                else:
                    raise RuntimeError("attn_mask must be [T,S] or [B,T,S]")
            else:
                add_mask = attn_mask
            logits = self._fq(logits + add_mask, self.obs_attn_mask_add)

        if key_padding_mask is not None:
            if not torch.is_floating_point(key_padding_mask):
                fill = self.causal_mask_template.new_tensor(self.mask_fill_value)
                kpm = torch.where(
                    key_padding_mask.to(torch.bool), fill, fill.new_zeros(())
                )
            else:
                kpm = key_padding_mask
            kpm = self._fq(kpm, self.obs_kpm_in)

            if not self.assume_additive_key_padding:
                # key_padding_mask: additive float already
                kpm = kpm.to(dtype=logits.dtype, device=device)
                if kpm.dim() == 2:  # [B,S]
                    kpm = (
                        kpm.view(B, 1, 1, Ts)
                        .expand(B, self.num_heads, Tq, Ts)
                        .contiguous()
                    )
                    kpm = kpm.view(B * self.num_heads, Tq, Ts)
                elif kpm.dim() == 3:  # [B,T,S]
                    kpm = (
                        kpm.unsqueeze(1).expand(B, self.num_heads, Tq, Ts).contiguous()
                    )
                    kpm = kpm.view(B * self.num_heads, Tq, Ts)
                else:
                    raise RuntimeError(
                        "key_padding_mask must be [B,S] or [B,T,S] (additive)"
                    )
            logits = self._fq(logits + kpm, self.obs_kp_mask_add)

        if before_softmax:
            if return_new_kv:
                return logits, v, new_k_bh, new_v_bh
            return logits, v

        # Softmax (float32) -> back to q.dtype
        attn_probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_probs = self._fq(attn_probs, self.obs_softmax)

        # Context + output proj
        ctx = self._fq(torch.bmm(attn_probs, v), self.obs_attn_out)  # [B*H,Tq,Dh]
        ctx = self._unfold_heads(ctx, B, Tq)  # [Tq,B,E]
        out = self.out_proj(ctx)

        # Weights (optional)
        attn_weights_out: Optional[torch.Tensor] = None
        if need_weights:
            aw = (
                torch.softmax(logits, dim=-1, dtype=torch.float32)
                .view(B, self.num_heads, Tq, Ts)
                .transpose(1, 0)
            )
            if not need_head_weights:
                aw = aw.mean(dim=1)  # [B,Tq,Ts]
            attn_weights_out = aw

        # Cache write
        if state is not None:
            state["prev_key"] = k.view(B, self.num_heads, -1, self.head_dim).detach()
            state["prev_value"] = v.view(B, self.num_heads, -1, self.head_dim).detach()
            self._set_input_buffer(incremental_state, state)

        if return_new_kv:
            return out, attn_weights_out, new_k_bh, new_v_bh
        return out, attn_weights_out

    def _all_observers(self):
        yield from (
            self.obs_query_in,
            self.obs_key_in,
            self.obs_value_in,
            self.obs_kpm_in,
            self.obs_causal_mask,
            self.obs_q_fold,
            self.obs_k_fold,
            self.obs_v_fold,
            self.obs_scale,
            self.obs_logits_raw,
            self.obs_logits,
            self.obs_attn_mask_add,
            self.obs_kp_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
        )
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if isinstance(m, QuantModuleBase):
                yield from m._all_observers()
