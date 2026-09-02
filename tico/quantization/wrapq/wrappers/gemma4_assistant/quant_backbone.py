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

"""Assistant-only container over the shared-KV Gemma4 text backbone."""

from typing import Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
    SUPPORTED_ASSISTANT_LAYER_TYPES,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


LayerKV = Tuple[torch.Tensor, torch.Tensor]


class QuantGemma4AssistantBackbone(QuantModuleBase):
    """PTQ container for the Gemma4 assistant decoder stack.

    The assistant's inner Hugging Face module is a ``Gemma4TextModel``, but
    the assistant never uses its token embedding, PLE path, mask factories, or
    model-level cache semantics. This container therefore wraps only
    ``layers`` and ``norm`` with the existing Gemma4 wrappers and consumes
    fully explicit per-layer-type masks, RoPE tables, and target shared K/V.

    It is intentionally not registered for ``Gemma4TextModel`` in the wrapper
    registry; ``QuantGemma4AssistantForCausalLM`` constructs it directly.
    """

    def __init__(
        self,
        fp_text_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.config = fp_text_model.config
        self.unique_layer_types = tuple(sorted(set(self.config.layer_types)))

        unsupported = set(self.unique_layer_types) - SUPPORTED_ASSISTANT_LAYER_TYPES
        if unsupported:
            raise ValueError(
                f"Unsupported Gemma4 assistant layer types: {sorted(unsupported)}."
            )
        for idx, layer in enumerate(fp_text_model.layers):
            if not bool(getattr(layer.self_attn, "is_kv_shared_layer", False)):
                raise ValueError(
                    "Gemma4 assistant backbone requires shared-KV consumer "
                    f"layers only, but layer {idx} owns K/V projections."
                )

        layers_cfg = qcfg.child("layers") if qcfg else None
        self.layers = nn.ModuleList(
            [
                PTQWrapper(
                    layer,
                    qcfg=layers_cfg.child(str(i)) if layers_cfg else None,
                    fp_name=join_name(fp_name, f"layers.{i}"),
                )
                for i, layer in enumerate(fp_text_model.layers)
            ]
        )
        self.norm = PTQWrapper(
            fp_text_model.norm,
            qcfg=qcfg.child("norm") if qcfg else None,
            fp_name=join_name(fp_name, "norm"),
        )
        # Eager RoPE construction reuses the floating-point rotary table
        # module; static export receives cos/sin as explicit inputs instead.
        self.rotary_emb = fp_text_model.rotary_emb

    def build_position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Build per-layer-type ``(cos, sin)`` tables for the eager path."""
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device,
            ).unsqueeze(0)
        position_ids = position_ids.to(device=hidden_states.device)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        with torch.no_grad():
            return {
                layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
                for layer_type in self.unique_layer_types
            }

    def _require_layer_type_entries(
        self,
        mapping: Mapping[str, object],
        *,
        what: str,
    ) -> None:
        """Validate that a per-layer-type mapping covers every layer type."""
        missing = [
            layer_type
            for layer_type in self.unique_layer_types
            if mapping.get(layer_type) is None
        ]
        if missing:
            raise ValueError(
                f"Gemma4 assistant backbone {what} is missing layer types: "
                f"{missing}."
            )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        attention_masks: Mapping[str, torch.Tensor],
        shared_kv_states: Mapping[str, LayerKV],
        position_embeddings: Optional[
            Mapping[str, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the assistant decoder stack and final norm.

        Args:
            inputs_embeds: Projected assistant input, shape ``(B, Q, hidden)``.
            attention_masks: Additive masks keyed by layer type, each
                broadcastable to ``(B, 1, Q, K_type)``.
            shared_kv_states: Target shared ``(key, value)`` tuples keyed by
                layer type. The assistant never projects or caches K/V.
            position_embeddings: Optional explicit ``(cos, sin)`` per layer
                type. Required by the static export path.
            position_ids: Positions used to build RoPE tables when
                ``position_embeddings`` is omitted (eager path).

        Returns:
            Final-normed hidden states shaped like ``inputs_embeds``.
        """
        self._require_layer_type_entries(attention_masks, what="attention_masks")
        self._require_layer_type_entries(shared_kv_states, what="shared_kv_states")

        if position_embeddings is None:
            position_embeddings = self.build_position_embeddings(
                inputs_embeds, position_ids
            )
        else:
            self._require_layer_type_entries(
                position_embeddings, what="position_embeddings"
            )

        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=attention_masks[layer_type],
                shared_key_value=shared_kv_states[layer_type],
                use_cache=False,
            )
        return self.norm(hidden_states)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return ()
