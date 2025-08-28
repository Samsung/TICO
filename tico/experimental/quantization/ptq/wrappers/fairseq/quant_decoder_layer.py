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

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn, Tensor

from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.fairseq.quant_mha import (
    QuantFairseqMultiheadAttention,
)
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.ptq.wrappers.registry import try_register


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
        qcfg: Optional[QuantConfig] = None,
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
        # Corresponding LayerNorm in FP (reuse instance)
        self.encoder_attn_layer_norm = getattr(
            fp_layer, "encoder_attn_layer_norm", None
        )

        # Feed-forward (PTQ) ------------------------------------------------
        fc1_cfg = qcfg.child("fc1") if qcfg else None
        fc2_cfg = qcfg.child("fc2") if qcfg else None
        assert hasattr(fp_layer, "fc1") and isinstance(fp_layer.fc1, nn.Module)
        assert hasattr(fp_layer, "fc2") and isinstance(fp_layer.fc2, nn.Module)
        self.fc1 = PTQWrapper(fp_layer.fc1, qcfg=fc1_cfg, fp_name=f"{fp_name}.fc1")
        self.fc2 = PTQWrapper(fp_layer.fc2, qcfg=fc2_cfg, fp_name=f"{fp_name}.fc2")

        # LayerNorms (FP, reuse instances)
        attn_ln_cfg = qcfg.child("self_attn_layer_norm") if qcfg else None
        final_ln_cfg = qcfg.child("final_layer_norm") if qcfg else None
        assert hasattr(fp_layer, "self_attn_layer_norm") and isinstance(
            fp_layer.self_attn_layer_norm, nn.Module
        )
        assert hasattr(fp_layer, "final_layer_norm") and isinstance(
            fp_layer.final_layer_norm, nn.Module
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

        # Activation function (FP; reuse)
        self.activation_fn = fp_layer.activation_fn  # type: ignore[operator]

        # Alignment flag used by Fairseq (kept for API parity)
        self.need_attn: bool = bool(getattr(fp_layer, "need_attn", True))

        # No dropout / activation-dropout in inference wrapper
        # (intentionally omitted)

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

    # ----------------------------------------------------------------------
    def _all_observers(self) -> Iterable:
        """
        Expose all observers from child PTQ-wrapped modules.
        This layer itself does not add extra per-tensor observers.
        """
        for m in (self.self_attn, self.encoder_attn, self.fc1, self.fc2):
            if isinstance(m, QuantModuleBase) and m is not None:
                yield from m._all_observers()
