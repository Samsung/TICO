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

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("fairseq.models.transformer.TransformerDecoderBase")
class QuantFairseqDecoder(QuantModuleBase):
    """
    Quant-aware drop-in replacement for Fairseq TransformerDecoderBase.

    Design (inference-only):
    - Keep embeddings, positional embeddings, LayerNorms, output_projection in FP.
    - PTQ-wrap all TransformerDecoderLayerBase items via PTQWrapper (uses QuantFairseqDecoderLayer).
    - Drop training-only logic (dropout, activation-dropout, quant-noise, checkpoint wrappers).
    - Preserve Fairseq forward/extract_features contract, shapes, and incremental decoding behavior.

    I/O:
    - Forward(prev_output_tokens, encoder_out, incremental_state, ...) -> (logits, extra) like the original.
    - `features_only=True` returns features without output projection.
    """

    def __init__(
        self,
        fp_decoder: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # ---- carry config/meta (read-only views) --------------------------
        assert hasattr(fp_decoder, "cfg")
        self.cfg = fp_decoder.cfg
        self.share_input_output_embed: bool = bool(
            getattr(fp_decoder, "share_input_output_embed", False)
        )

        # Version buffer (parity with original)
        version = getattr(fp_decoder, "version", None)
        if isinstance(version, torch.Tensor):
            self.register_buffer("version", version.clone(), persistent=False)
        else:
            self.register_buffer("version", torch.tensor([3.0]), persistent=False)

        # Embeddings / positional encodings (FP; reuse modules)
        assert hasattr(fp_decoder, "embed_tokens") and isinstance(
            fp_decoder.embed_tokens, nn.Module
        )
        self.embed_tokens = fp_decoder.embed_tokens  # (B,T)->(B,T,C)

        self.padding_idx: int = int(fp_decoder.padding_idx)  # type: ignore[arg-type]
        self.max_target_positions: int = int(fp_decoder.max_target_positions)  # type: ignore[arg-type]

        self.embed_positions = getattr(fp_decoder, "embed_positions", None)
        self.layernorm_embedding = getattr(fp_decoder, "layernorm_embedding", None)

        # Dimensions / projections (reuse)
        self.embed_dim: int = int(getattr(fp_decoder, "embed_dim"))
        self.output_embed_dim: int = int(getattr(fp_decoder, "output_embed_dim"))
        self.project_in_dim = getattr(fp_decoder, "project_in_dim", None)
        self.project_out_dim = getattr(fp_decoder, "project_out_dim", None)

        # Scale factor (sqrt(embed_dim) unless disabled)
        no_scale = bool(getattr(self.cfg, "no_scale_embedding", False))
        self.embed_scale: float = 1.0 if no_scale else math.sqrt(self.embed_dim)

        # Final decoder LayerNorm (may be None depending on cfg)
        self.layer_norm = getattr(fp_decoder, "layer_norm", None)

        # Output projection / adaptive softmax (reuse FP modules)
        self.adaptive_softmax = getattr(fp_decoder, "adaptive_softmax", None)
        self.output_projection = getattr(fp_decoder, "output_projection", None)

        # ---- wrap decoder layers ------------------------------------------
        assert hasattr(fp_decoder, "layers")
        fp_layers = list(fp_decoder.layers)  # type: ignore[arg-type]
        self.layers = nn.ModuleList()

        # Safe prefix to avoid None-based name collisions in KV cache keys
        def _safe_prefix(name: Optional[str]) -> str:
            return (
                name
                if (name is not None and name != "" and name != "None")
                else f"{self.__class__.__name__}_{id(self)}"
            )

        prefix = _safe_prefix(fp_name)

        # Prepare child PTQConfig namespaces: layers/<idx>
        layers_qcfg = qcfg.child("layers") if qcfg else None
        for i, layer in enumerate(fp_layers):
            child_cfg = layers_qcfg.child(str(i)) if layers_qcfg else None
            # Not every item is necessarily a TransformerDecoderLayerBase (e.g., BaseLayer).
            # If there's no registered wrapper for a layer type, keep it FP.
            try:
                wrapped = PTQWrapper(
                    layer, qcfg=child_cfg, fp_name=f"{prefix}.layers.{i}"
                )
            except NotImplementedError:
                wrapped = layer  # keep as-is (FP)
            self.layers.append(wrapped)
        self.num_layers = len(self.layers)

        # choose a generous upper-bound; you can wire this from cfg if you like
        self.mask_fill_value: float = -120.0
        max_tgt = int(getattr(self.cfg, "max_target_positions", 2048))  # fallback: 2048

        mask = torch.full((1, 1, max_tgt, max_tgt), float(self.mask_fill_value))
        mask.triu_(1)  # upper triangle set to fill_value; diagonal/lower are zeros
        self.register_buffer("causal_mask_template", mask, persistent=False)

    def forward(
        self,
        prev_output_tokens: Tensor,  # [B, T]
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Match the original API.
        Returns:
            (logits_or_features, extra_dict)
        """
        x, extra = self.extract_features_scriptable(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features_scriptable(
        self,
        prev_output_tokens: Tensor,  # [B,T]
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, List[Optional[Tensor]]]]:
        """
        Feature path that mirrors Fairseq's implementation (minus training-only code).

        Returns:
            x: [B, T, C]
            extra: {"attn": [attn or None], "inner_states": [T x B x C tensors]}
        """
        B, T = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # Unpack encoder outputs in Fairseq dict format
        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out.get("encoder_out", [])) > 0:
            enc = encoder_out["encoder_out"][0]  # [S,B,Ce]
        if (
            encoder_out is not None
            and len(encoder_out.get("encoder_padding_mask", [])) > 0
        ):
            padding_mask = encoder_out["encoder_padding_mask"][0]  # [B,S] (bool)

        # Positional embeddings (support incremental decoding)
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        # In incremental mode, only the last step is consumed
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent view quirks (TorchScript parity in original)
        prev_output_tokens = prev_output_tokens.contiguous()

        # Token embeddings (+ optional proj-in), + positions, + optional LN
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)  # [B,T,C]
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x = x + positions
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        # No dropout / quant_noise (inference-only)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Build self-attn masks
        self_attn_padding_mask: Optional[Tensor] = None
        if (
            getattr(self.cfg, "cross_self_attention", False)
            or prev_output_tokens.eq(self.padding_idx).any()
        ):
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)  # [B,T]

        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        for idx, layer in enumerate(self.layers):
            # Causal mask unless full-context alignment or incremental decoding
            if incremental_state is None and not full_context_alignment:
                Tq = x.size(0)
                self_attn_mask = self.buffered_future_mask(
                    Tq, Tq, x=x
                )  # [Tq,Tq] additive float
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool(idx == alignment_layer),
                need_head_weights=bool(idx == alignment_layer),
            )

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        # Average heads if needed
        if attn is not None and alignment_heads is not None:
            attn = attn[:alignment_heads]
        if attn is not None:
            attn = attn.mean(dim=0)  # [B,T,S]

        # Optional final layer norm
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # Optional proj-out
        if self.project_out_dim is not None:
            assert self.project_out_dim is not None
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features: Tensor) -> Tensor:
        """Project features to vocabulary size (or return features with adaptive softmax)."""
        if self.adaptive_softmax is None:
            assert self.output_projection is not None
            return self.output_projection(features)  # type: ignore[operator]
        else:
            return features

    def buffered_future_mask(
        self, Tq: int, Ts: int, *, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Return additive float mask [Tq, Ts]: zeros on allowed, large-neg on disallowed.
        Uses the prebuilt template; will re-build if you exceed template size.
        """
        assert isinstance(self.causal_mask_template, torch.Tensor)
        Mmax = self.causal_mask_template.size(-1)
        assert Tq <= Mmax and Ts <= Mmax
        cm = self.causal_mask_template[..., :Tq, :Ts].to(device=x.device, dtype=x.dtype)
        return cm.squeeze(0).squeeze(0)  # [Tq, Ts]

    def max_positions(self) -> int:
        """Maximum output length supported by the decoder (same policy as the original)."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1, dtype=torch.float32)
        else:
            return F.softmax(logits, dim=-1, dtype=torch.float32)

    def reorder_incremental_state_scripting(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Main entry point for reordering the incremental state.

        Due to limitations in TorchScript, we call this function in
        :class:`fairseq.sequence_generator.SequenceGenerator` instead of
        calling :func:`reorder_incremental_state` directly.
        """
        for module in self.modules():
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, new_order)  # type: ignore[operator]
                if result is not None:
                    incremental_state = result

    def forward_external_step(
        self,
        prev_output_x: Tensor,  # [1, B, C]
        *,
        encoder_out_x: Tensor,  # [S, B, Ce]
        encoder_padding_mask: Optional[
            Tensor
        ] = None,  # [B,S] or [B,1,S] additive-float
        self_attn_mask: Optional[
            Tensor
        ] = None,  # [1,S_hist+1] or [B,1,S_hist+1] additive-float
        prev_self_k_list: Optional[
            List[Tensor]
        ] = None,  # length=L; each [B,H,Tprev,Dh]
        prev_self_v_list: Optional[
            List[Tensor]
        ] = None,  # length=L; each [B,H,Tprev,Dh]
        need_attn: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Export-only single-step decoder.
        Returns:
          - x_out: [1, B, C]
          - new_self_k_list/new_self_v_list: lists of length L; each [B*H, Tnew, Dh]
        """
        assert (
            prev_output_x.dim() == 3 and prev_output_x.size(0) == 1
        ), "prev_output_x must be [1,B,C]"
        L = self.num_layers
        if prev_self_k_list is None:
            prev_self_k_list = [None] * L  # type: ignore[list-item]
        if prev_self_v_list is None:
            prev_self_v_list = [None] * L  # type: ignore[list-item]
        assert len(prev_self_k_list) == L and len(prev_self_v_list) == L

        assert encoder_out_x.dim() == 3, "encoder_out_x must be [S,B,C]"
        x = prev_output_x  # [1,B,C]
        enc = encoder_out_x

        new_k_list: List[Tensor] = []
        new_v_list: List[Tensor] = []

        for li, layer in enumerate(self.layers):
            assert isinstance(layer, PTQWrapper)
            x, _, k_new, v_new = layer.wrapped.forward_external(  # type: ignore[attr-defined, operator]
                x,
                encoder_out=enc,
                encoder_padding_mask=encoder_padding_mask,
                prev_self_k=prev_self_k_list[li],
                prev_self_v=prev_self_v_list[li],
                self_attn_mask=self_attn_mask,
                need_attn=need_attn and (li == L - 1),
                need_head_weights=need_head_weights and (li == L - 1),
            )
            new_k_list.append(k_new)  # [B*H, Tnew, Dh]
            new_v_list.append(v_new)  # [B*H, Tnew, Dh]

        if self.layer_norm is not None:
            x = self.layer_norm(x.transpose(0, 1)).transpose(0, 1)

        return x, new_k_list, new_v_list  # [1,B,C], lists of [B*H, Tnew, Dh]

    def _all_observers(self) -> Iterable:
        """Yield all observers from wrapped decoder layers (if any)."""
        for m in self.layers:
            if isinstance(m, QuantModuleBase):
                yield from m._all_observers()
