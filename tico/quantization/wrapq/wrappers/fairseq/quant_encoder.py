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
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("fairseq.models.transformer.TransformerEncoderBase")
class QuantFairseqEncoder(QuantModuleBase):
    """
    Quant-aware drop-in replacement for Fairseq TransformerEncoderBase.

    Key design choices:
    - Keep embeddings and LayerNorms in FP.
    - Remove training-time logic (dropout, activation-dropout, quant_noise).
    - Attention masks are handled statically inside the layer wrapper; this
      encoder only does the original padding zero-out before the stack.

    I/O contracts:
    - Forward signature and returned dictionary are identical to the original
      when `use_external_inputs=False`.
    - When `use_external_inputs=True`, forward returns a single Tensor (T,B,C)
      and completely skips embedding/positional/LN/mask-creation paths.
    - Tensor shapes follow Fairseq convention.
    """

    def __init__(
        self,
        fp_encoder: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
        use_external_inputs: bool = False,  # export-mode flag
        return_type: Literal["tensor", "dict"] = "dict",
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.use_external_inputs = use_external_inputs
        self.return_type: Literal["tensor", "dict"] = return_type

        # --- carry basic config / metadata (read-only copies) ---------------
        assert hasattr(fp_encoder, "cfg")
        self.cfg = fp_encoder.cfg
        self.return_fc: bool = bool(getattr(fp_encoder, "return_fc", False))

        # Embedding stack ----------------------------------------------------
        assert hasattr(fp_encoder, "embed_tokens") and isinstance(
            fp_encoder.embed_tokens, nn.Module
        )
        self.embed_tokens = fp_encoder.embed_tokens  # keep FP embeddings

        assert hasattr(fp_encoder, "padding_idx")
        self.padding_idx: int = int(fp_encoder.padding_idx)  # type: ignore[arg-type]

        # scale = sqrt(embed_dim) unless disabled
        embed_dim = int(self.embed_tokens.embedding_dim)  # type: ignore[arg-type]
        no_scale = bool(getattr(self.cfg, "no_scale_embedding", False))
        self.embed_scale: float = 1.0 if no_scale else math.sqrt(embed_dim)

        # Positional embeddings (keep as-is; no FQ)
        self.embed_positions = getattr(fp_encoder, "embed_positions", None)
        # Optional embedding LayerNorm
        self.layernorm_embedding = getattr(fp_encoder, "layernorm_embedding", None)

        # Final encoder LayerNorm (pre-norm stacks may set this to None)
        self.layer_norm = getattr(fp_encoder, "layer_norm", None)

        # Max positions (reuse for API parity)
        self.max_source_positions: int = int(fp_encoder.max_source_positions)  # type: ignore[arg-type]

        # --- wrap encoder layers with PTQWrapper ----------------------------
        assert hasattr(fp_encoder, "layers")
        fp_layers = list(fp_encoder.layers)  # type: ignore[arg-type]
        self.layers = nn.ModuleList()

        # Prepare child PTQConfig namespaces: layers/<idx>
        layers_qcfg = qcfg.child("layers") if qcfg else None
        for i, layer in enumerate(fp_layers):
            child_cfg = layers_qcfg.child(str(i)) if layers_qcfg else None
            self.layers.append(
                PTQWrapper(layer, qcfg=child_cfg, fp_name=f"{fp_name}.layers.{i}")
            )

        # Version buffer (keep for state_dict parity)
        version = getattr(fp_encoder, "version", None)
        if isinstance(version, torch.Tensor):
            self.register_buffer("version", version.clone(), persistent=False)
        else:
            self.register_buffer("version", torch.tensor([3.0]), persistent=False)

    # ----------------------------------------------------------------------
    def forward_embedding(
        self, src_tokens: Tensor, token_embedding: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Embed tokens and add positional embeddings. Dropout/quant_noise are removed.
        Returns:
            x (B, T, C), embed (B, T, C)  # embed is the token-only embedding
        """
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        embed = token_embedding  # token-only

        x = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        # No dropout, no quant_noise here (inference-only)
        return x, embed

    # ----------------------------------------------------------------------
    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None,
        *,
        # External-inputs branch (used for export)
        encoder_padding_mask: Optional[Tensor] = None,  # B x T (bool)
    ) -> Tensor | Dict[str, List[Optional[Tensor]]]:
        """
        If `self.use_external_inputs` is True:
          - Use only x_external and encoder_padding_mask.
          - Return a single Tensor (T, B, C) for export friendliness.

        Otherwise (False):
          - Behave like the original Fairseq encoder forward and return dict-of-lists.
        """
        if self.use_external_inputs:
            # ----- External-input mode: completely skip embedding/positional/LN/mask creation -----
            x_external = src_tokens  # T x B x C (already embedded + transposed)

            encoder_states: List[Tensor] = []
            if return_all_hiddens:
                encoder_states.append(x_external)

            for layer in self.layers:
                out = layer(x_external, encoder_padding_mask=encoder_padding_mask)
                x_external = (
                    out[0] if (isinstance(out, tuple) and len(out) == 2) else out
                )
                if return_all_hiddens:
                    encoder_states.append(x_external)

            if self.layer_norm is not None:
                x_external = self.layer_norm(x_external)

            if self.return_type == "dict":
                return {
                    "encoder_out": [x_external],
                    "encoder_padding_mask": [encoder_padding_mask],
                    "encoder_states": encoder_states,  # type: ignore[dict-item]
                }
            else:
                # For export, returning a single Tensor is simpler and more portable.
                return x_external

        # ----- Original path (training/eval compatibility) ------------------

        # Compute padding mask [B, T] (bool). We keep the original "has_pads" logic.
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads: Tensor = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        # Embedding path (B,T,C). No dropout/quant_noise.
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # Zero out padded timesteps prior to the stack (same as original)
        x = x * (
            1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states: List[Tensor] = []  # type: ignore[no-redef]
        fc_results: List[Optional[Tensor]] = []

        if return_all_hiddens:
            encoder_states.append(x)

        # Encoder layers (each item is PTQ-wrapped and uses static additive masks internally)
        for layer in self.layers:
            out = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if isinstance(out, tuple) and len(out) == 2:
                x, fc_res = out
            else:
                x = out
                fc_res = None

            if return_all_hiddens and not torch.jit.is_scripting():
                encoder_states.append(x)
                fc_results.append(fc_res)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # src_lengths (B, 1) int32, identical to original
        src_lengths_out = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # type: ignore[dict-item]  # List[T x B x C]
            "fc_results": fc_results,  # type: ignore[dict-item]  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths_out],
        }

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if "encoder_padding_mask" in net_input:
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
                encoder_padding_mask=net_input["encoder_padding_mask"],
            )
        else:
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
            )

    # ----------------------------------------------------------------------
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Match original API: reorder the batched dimension (B) according to new_order.
        """
        reordered = dict()  # type: ignore[var-annotated]
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        reordered["encoder_out"] = new_encoder_out
        keys = [
            "encoder_padding_mask",
            "encoder_embedding",
            "src_tokens",
            "src_lengths",
        ]
        for k in keys:
            if k not in encoder_out:
                continue
            if len(encoder_out[k]) == 0:
                reordered[k] = []
            else:
                reordered[k] = [encoder_out[k][0].index_select(0, new_order)]

        if "encoder_states" in encoder_out:
            encoder_states = encoder_out["encoder_states"]
            if len(encoder_states) > 0:
                for idx, state in enumerate(encoder_states):
                    encoder_states[idx] = state.index_select(1, new_order)
            reordered["encoder_states"] = encoder_states

        return reordered

    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order for beamable enc-dec attention (API parity)."""
        return encoder_out

    def max_positions(self) -> int:
        """Maximum input length supported by the encoder (same policy as the original)."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Forward-compat mapping for older checkpoints (mirror original behavior for LNs).
        The actual remapping of per-layer norms is delegated to the wrapped layers.
        """
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "upgrade_state_dict_named"):
                layer.upgrade_state_dict_named(state_dict, f"{name}.layers.{i}")

        version_key = f"{name}.version"
        v = state_dict.get(version_key, torch.Tensor([1]))
        if float(v[0].item()) < 2:
            self.layer_norm = None
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def _all_observers(self):
        for m in self.layers:
            if isinstance(m, QuantModuleBase):
                yield from m._all_observers()
