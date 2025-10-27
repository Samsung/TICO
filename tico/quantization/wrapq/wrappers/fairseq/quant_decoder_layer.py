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

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn, Tensor

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.fairseq.quant_mha import (
    QuantFairseqMultiheadAttention,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("fairseq.modules.transformer_layer.TransformerDecoderLayerBase")
class QuantFairseqDecoderLayer(QuantModuleBase):
    """
    Quant-aware drop-in replacement for Fairseq TransformerDecoderLayerBase.

    Design (inference-only):
    - Keep LayerNorms and scalar head/residual scalers in FP.
    - PTQ-wrap: self_attn, (optional) encoder_attn, fc1, fc2.
    - Preserve Fairseq tensor contracts and incremental state handling.
    - Remove training-time behaviors: dropout, activation-dropout, quant-noise, onnx_trace.

    I/O:
    - Input/Output use Fairseq shapes: [T, B, C].
    - Forward returns: (x, attn, None) to match the original call sites in decoder.
      * `attn` is from encoder-attention when requested (alignment).
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # --- read-only metadata copied from FP layer -----------------------
        assert hasattr(fp_layer, "embed_dim")
        assert hasattr(fp_layer, "normalize_before")
        self.embed_dim: int = int(fp_layer.embed_dim)  # type: ignore[arg-type]
        self.normalize_before: bool = bool(fp_layer.normalize_before)

        # Cross-self attention flag (when True, key/value can include encoder_out)
        self.cross_self_attention: bool = bool(
            getattr(fp_layer, "cross_self_attention", False)
        )

        # Generate prefix
        def _safe_prefix(name: Optional[str]) -> str:
            # Avoid "None.*" strings causing collisions
            return (
                name
                if (name is not None and name != "None" and name != "")
                else f"{self.__class__.__name__}_{id(self)}"
            )

        prefix = _safe_prefix(fp_name)
        # Self-attn (PTQ) ---------------------------------------------------
        # Use our MHA wrapper with identical API to the FP module.
        attn_cfg = qcfg.child("self_attn") if qcfg else None
        assert hasattr(fp_layer, "self_attn") and isinstance(
            fp_layer.self_attn, nn.Module
        )
        self.self_attn = QuantFairseqMultiheadAttention(
            fp_layer.self_attn, qcfg=attn_cfg, fp_name=f"{prefix}.self_attn"
        )

        # Optional attention LayerNorm applied to self-attn output (scale_attn)
        # Kept in FP; reuse original instance for weight parity.
        self.attn_ln = getattr(fp_layer, "attn_ln", None)

        # Optional per-head scaling after self-attn output (scale_heads)
        # Keep exact Parameter reference if present (shape: [num_heads])
        self.c_attn = getattr(fp_layer, "c_attn", None)

        # Cache head meta for c_attn path
        self.nh = int(getattr(self.self_attn, "num_heads"))
        self.head_dim = int(getattr(self.self_attn, "head_dim"))

        # Encoder-attn (PTQ) ------------------------------------------------
        # Only present if the original layer was constructed with encoder_attn.
        enc_attn_mod = getattr(fp_layer, "encoder_attn", None)
        assert enc_attn_mod is not None
        enc_cfg = qcfg.child("encoder_attn") if qcfg else None
        self.encoder_attn = QuantFairseqMultiheadAttention(
            enc_attn_mod, qcfg=enc_cfg, fp_name=f"{prefix}.encoder_attn"
        )

        # Feed-forward (PTQ) ------------------------------------------------
        fc1_cfg = qcfg.child("fc1") if qcfg else None
        fc2_cfg = qcfg.child("fc2") if qcfg else None
        assert hasattr(fp_layer, "fc1") and isinstance(fp_layer.fc1, nn.Module)
        assert hasattr(fp_layer, "fc2") and isinstance(fp_layer.fc2, nn.Module)
        self.fc1 = PTQWrapper(fp_layer.fc1, qcfg=fc1_cfg, fp_name=f"{fp_name}.fc1")
        self.fc2 = PTQWrapper(fp_layer.fc2, qcfg=fc2_cfg, fp_name=f"{fp_name}.fc2")

        # LayerNorms
        enc_attn_ln_cfg = qcfg.child("encoder_attn_layer_norm") if qcfg else None
        attn_ln_cfg = qcfg.child("self_attn_layer_norm") if qcfg else None
        final_ln_cfg = qcfg.child("final_layer_norm") if qcfg else None
        assert hasattr(fp_layer, "encoder_attn_layer_norm") and isinstance(
            fp_layer.encoder_attn_layer_norm, nn.Module
        )
        assert hasattr(fp_layer, "self_attn_layer_norm") and isinstance(
            fp_layer.self_attn_layer_norm, nn.Module
        )
        assert hasattr(fp_layer, "final_layer_norm") and isinstance(
            fp_layer.final_layer_norm, nn.Module
        )
        self.encoder_attn_layer_norm = PTQWrapper(
            fp_layer.encoder_attn_layer_norm,
            qcfg=enc_attn_ln_cfg,
            fp_name=f"{fp_name}.encoder_attn_layer_norm",
        )
        self.self_attn_layer_norm = PTQWrapper(
            fp_layer.self_attn_layer_norm,
            qcfg=attn_ln_cfg,
            fp_name=f"{fp_name}.self_attn_layer_norm",
        )
        self.final_layer_norm = PTQWrapper(
            fp_layer.final_layer_norm,
            qcfg=final_ln_cfg,
            fp_name=f"{fp_name}.final_layer_norm",
        )

        # Optional FFN intermediate LayerNorm (scale_fc), FP
        self.ffn_layernorm = getattr(fp_layer, "ffn_layernorm", None)

        # Optional residual scaling (scale_resids), keep Parameter reference
        self.w_resid = getattr(fp_layer, "w_resid", None)

        # Activation function
        self.activation_fn = fp_layer.activation_fn  # type: ignore[operator]
        self.obs_activation_fn = self._make_obs("activation_fn")

        # Alignment flag used by Fairseq (kept for API parity)
        self.need_attn: bool = bool(getattr(fp_layer, "need_attn", True))

        # No dropout / activation-dropout in inference wrapper
        # (intentionally omitted)

        # --- observers for external/self-attn KV cache inputs --------------
        self.obs_prev_self_k_in = self._make_obs("prev_self_k_in")
        self.obs_prev_self_v_in = self._make_obs("prev_self_v_in")

    # ----------------------------------------------------------------------
    def _maybe_apply_head_scale(self, x: Tensor) -> Tensor:
        """
        Optional per-head scaling (scale_heads) after self-attention.
        x: [T, B, C]
        """
        if self.c_attn is None:
            return x
        T, B, _ = x.shape
        x = x.view(T, B, self.nh, self.head_dim)  # [T,B,H,Dh]
        # einsum over head dim: scales each head independently
        x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)  # [T,B,H,Dh]
        return x.reshape(T, B, self.nh * self.head_dim)  # [T,B,C]

    # ----------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,  # [T,B,C]
        encoder_out: Optional[Tensor] = None,  # [S,B,Ce] or None
        encoder_padding_mask: Optional[Tensor] = None,  # [B,S] bool or additive float
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[Tensor]] = None,
        prev_attn_state: Optional[List[Tensor]] = None,
        self_attn_mask: Optional[Tensor] = None,  # [T,T] or [B,T,T] or None
        self_attn_padding_mask: Optional[Tensor] = None,  # [B,T] or [B,T,T] or None
        need_attn: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], None]:
        """
        Mirrors the original forward, minus training-only logic.
        Returns:
            x': [T,B,C], attn (from encoder-attn when requested), None
        """
        if need_head_weights:
            need_attn = True

        # ---- (1) Self-Attention block ------------------------------------
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Load provided cached self-attn state (for incremental decoding)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        # Cross-self-attention: prepend encoder_out to K/V at the first step
        y = x
        if self.cross_self_attention:
            _buf = self.self_attn._get_input_buffer(incremental_state)
            no_cache_yet = not (
                incremental_state is not None
                and _buf is not None
                and "prev_key" in _buf
            )
            if no_cache_yet:
                if self_attn_mask is not None:
                    assert encoder_out is not None
                    # Grow attn mask to cover encoder timesteps (no autoregressive penalty for them)
                    self_attn_mask = torch.cat(
                        (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask),
                        dim=1,
                    )
                if self_attn_padding_mask is not None:
                    if encoder_padding_mask is None:
                        assert encoder_out is not None
                        encoder_padding_mask = self_attn_padding_mask.new_zeros(
                            encoder_out.size(1), encoder_out.size(0)
                        )
                    # Concatenate encoder pad-mask in front of target pad-mask
                    self_attn_padding_mask = torch.cat(
                        (encoder_padding_mask, self_attn_padding_mask), dim=1
                    )
                assert encoder_out is not None
                y = torch.cat((encoder_out, x), dim=0)  # [S+T, B, C]

        # Self-attn; Fairseq never consumes self-attn weights for alignment here
        x, _ = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        # Optional per-head scaling and attn LayerNorm on self-attn output
        x = self._maybe_apply_head_scale(x)
        if self.attn_ln is not None:
            x = self.attn_ln(x)

        # Residual + (post-norm if applicable)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # ---- (2) Encoder-Decoder Attention block --------------------------
        attn_out: Optional[Tensor] = None
        assert encoder_out is not None
        residual = x
        assert self.encoder_attn_layer_norm is not None
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Load provided cached cross-attn state
        if prev_attn_state is not None:
            prev_key, prev_value = prev_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            assert incremental_state is not None
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)

        # Cross-attn (static_kv=True to reuse encoder K/V across steps)
        assert self.encoder_attn is not None
        x, attn_out = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or self.need_attn,
            need_head_weights=need_head_weights,
        )

        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # ---- (3) Feed-Forward block --------------------------------------
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        # FFN: fc1 -> activation -> (optional LN) -> fc2
        x = self.fc1(x)
        x = self.activation_fn(x)  # type: ignore[operator]
        x = self._fq(x, self.obs_activation_fn)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)

        # Optional residual scaling (scale_resids)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)

        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # Return attn from encoder-attn branch when requested; self-attn weights are not returned.
        return x, attn_out, None

    def forward_external(
        self,
        x: Tensor,  # [1, B, C] (embedded current-step token)
        *,
        encoder_out: Optional[Tensor],  # [S, B, Ce]
        encoder_padding_mask: Optional[
            Tensor
        ] = None,  # [B,S] bool or additive-float or [B,1,S] additive-float
        prev_self_k: Optional[Tensor] = None,  # [B, H, Tprev, Dh]
        prev_self_v: Optional[Tensor] = None,  # [B, H, Tprev, Dh]
        self_attn_mask: Optional[
            Tensor
        ] = None,  # [1, 1, S_hist+1] or [B,1,S_hist+1] additive-float
        need_attn: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Export-only single-step:
          Returns (x_out[1,B,C], attn_from_cross, new_self_k[B,H,1,Dh], new_self_v[B,H,1,Dh]).
        """
        if need_head_weights:
            need_attn = True

        assert x.dim() == 3 and x.size(0) == 1, "x must be [1,B,C]"
        B = x.size(1)

        # ---- Self-Attention (uses MHA return_new_kv) ----------------------
        x_tbc = x
        if self.normalize_before:
            x_tbc = self.self_attn_layer_norm(x_tbc)

        # Provide prev KV via incremental_state so wrapper appends internally
        incr: Dict[str, Dict[str, Optional[Tensor]]] = {}
        if prev_self_k is not None and prev_self_v is not None:
            # Attach observers to incoming caches
            prev_self_k = self._fq(prev_self_k, self.obs_prev_self_k_in)
            prev_self_v = self._fq(prev_self_v, self.obs_prev_self_v_in)
            assert isinstance(prev_self_k, Tensor) and isinstance(prev_self_v, Tensor)
            saved = {
                "prev_key": prev_self_k.detach(),
                "prev_value": prev_self_v.detach(),
            }
            self.self_attn._set_input_buffer(incr, saved)  # type: ignore[arg-type]

        # Normalize self-attn additive mask to shapes wrapper accepts: [T,S] or [B,T,S]
        attn_mask_for_wrapper = None
        if self_attn_mask is not None:
            if (
                self_attn_mask.dim() == 3
                and self_attn_mask.size(0) == B
                and self_attn_mask.size(1) == 1
            ):
                attn_mask_for_wrapper = self_attn_mask  # [B,1,S]
            elif (
                self_attn_mask.dim() == 3
                and self_attn_mask.size(0) == 1
                and self_attn_mask.size(1) == 1
            ):
                attn_mask_for_wrapper = self_attn_mask[0]  # -> [1,S]
            elif self_attn_mask.dim() == 2 and self_attn_mask.size(0) == 1:
                attn_mask_for_wrapper = self_attn_mask  # [1,S]
            else:
                raise RuntimeError(
                    "self_attn_mask must be [1,S] or [B,1,S] additive-float."
                )
            attn_mask_for_wrapper = attn_mask_for_wrapper.to(
                dtype=x_tbc.dtype, device=x_tbc.device
            )

        x_sa, _, new_k_bh, new_v_bh = self.self_attn(
            query=x_tbc,
            key=x_tbc,
            value=x_tbc,
            key_padding_mask=None,
            incremental_state=incr,
            need_weights=False,
            attn_mask=attn_mask_for_wrapper,
            return_new_kv=True,  # <<< NEW: ask wrapper to return this step's K/V
        )  # x_sa: [1,B,C]; new_k_bh/new_v_bh: [B*H, Tnew, Dh]

        x_sa = self._maybe_apply_head_scale(x_sa)
        if self.attn_ln is not None:
            x_sa = self.attn_ln(x_sa)

        x_tbc = x_tbc + x_sa
        if not self.normalize_before:
            x_tbc = self.self_attn_layer_norm(x_tbc)

        # ---- Encoder-Decoder Attention -----------------------------------
        assert encoder_out is not None, "encoder_out is required in export path"
        residual = x_tbc
        if self.normalize_before:
            assert self.encoder_attn_layer_norm is not None
            x_tbc = self.encoder_attn_layer_norm(x_tbc)

        enc_kpm = encoder_padding_mask  # pass-through; wrapper handles bool/additive
        x_ed, attn_out = self.encoder_attn(
            query=x_tbc,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=enc_kpm,
            incremental_state=None,
            static_kv=True,
            need_weights=need_attn,
            need_head_weights=need_head_weights,
        )

        x_tbc = residual + x_ed
        if not self.normalize_before:
            assert self.encoder_attn_layer_norm is not None
            x_tbc = self.encoder_attn_layer_norm(x_tbc)

        # ---- Feed-Forward -------------------------------------------------
        residual = x_tbc
        if self.normalize_before:
            x_tbc = self.final_layer_norm(x_tbc)

        x_tbc = self.fc1(x_tbc)
        x_tbc = self.activation_fn(x_tbc)  # type: ignore[operator]
        x_tbc = self._fq(x_tbc, self.obs_activation_fn)
        if self.ffn_layernorm is not None:
            x_tbc = self.ffn_layernorm(x_tbc)
        x_tbc = self.fc2(x_tbc)

        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)

        x_tbc = residual + x_tbc
        if not self.normalize_before:
            x_tbc = self.final_layer_norm(x_tbc)

        return (
            x_tbc,
            attn_out,
            new_k_bh,
            new_v_bh,
        )  # [1,B,C], attn, [B*H, Tnew, Dh], [B*H, Tnew, Dh]

    def _all_observers(self) -> Iterable:
        """
        Expose all observers from child PTQ-wrapped modules.
        This layer itself does not add extra per-tensor observers.
        """
        # local observers
        yield from (
            self.obs_activation_fn,
            self.obs_prev_self_k_in,
            self.obs_prev_self_v_in,
        )

        for m in (
            self.self_attn,
            self.encoder_attn,
            self.fc1,
            self.fc2,
            self.encoder_attn_layer_norm,
            self.self_attn_layer_norm,
            self.final_layer_norm,
        ):
            if isinstance(m, QuantModuleBase) and m is not None:
                yield from m._all_observers()
