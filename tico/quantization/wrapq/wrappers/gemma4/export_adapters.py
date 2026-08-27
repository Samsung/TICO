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

"""Export adapters for Gemma4 E2B static-shape runtime.

The adapters define the contracts that should be exported to NPU-friendly static
graphs. CPU runtime code owns dynamic orchestration, cache writes, sampling, and
processor/tokenizer logic.
"""

from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.wrappers.gemma4.utils import fixed_slot_fuse
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


LayerKV = Tuple[torch.Tensor, torch.Tensor]


def _flatten_hidden_and_kv(output: Any, *, return_kv: bool) -> Any:
    """Return ``hidden`` or ``(hidden, key, value)`` from a layer-wrapper output."""
    if isinstance(output, tuple):
        if not output:
            raise RuntimeError("Gemma4 decoder export adapter received an empty tuple.")
        hidden_states = output[0]
        key_value = output[1] if len(output) > 1 else None
    else:
        hidden_states = output
        key_value = None

    if not return_kv:
        return hidden_states
    if key_value is None:
        return hidden_states
    if not isinstance(key_value, tuple) or len(key_value) != 2:
        raise RuntimeError(
            "Gemma4 decoder export adapter expected cache output to be a "
            "``(key, value)`` tuple."
        )
    key, value = key_value
    return hidden_states, key, value


def _flatten_attention_and_kv(output: Any, *, return_kv: bool) -> Any:
    """Return attention hidden states and optional K/V delta tensors."""
    if not isinstance(output, tuple) or not output:
        raise RuntimeError(
            "Gemma4 attention export adapter expected a non-empty tuple output."
        )

    hidden_states = output[0]
    if not return_kv:
        return hidden_states

    key_value = output[2] if len(output) > 2 else None
    if key_value is None:
        return hidden_states
    if not isinstance(key_value, tuple) or len(key_value) != 2:
        raise RuntimeError(
            "Gemma4 attention export adapter expected cache output to be a "
            "``(key, value)`` tuple."
        )
    key, value = key_value
    return hidden_states, key, value


class Gemma4TokenEmbeddingExportAdapter(nn.Module):
    """Export adapter for Gemma4 token embeddings.

    Input contract:
        ``input_ids`` has shape ``(1, S)`` for prefill or ``(1, 1)`` for decode.

    Output contract:
        ``hidden_states`` has shape ``(1, S, hidden_size)``.
    """

    def __init__(self, wrapped_text_model: nn.Module):
        super().__init__()
        self.embed_tokens = wrapped_text_model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token embeddings for static runtime execution."""
        return self.embed_tokens(input_ids)


class Gemma4VisionPatchEmbedderPrefillExportAdapter(nn.Module):
    """Export a patch embedder specialized for one fixed position profile.

    The construction-time position IDs and padding mask are folded into a
    positional-embedding template before export. The resulting runtime ABI
    accepts only pixel values.
    """

    def __init__(
        self,
        wrapped: nn.Module,
        *,
        position_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.register_buffer(
            "position_embeddings_template",
            position_embeddings.detach().clone(),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Project pixels and add the baked positional-embedding template."""
        return self.wrapped.forward_export(
            pixel_values,
            position_embeddings=self.position_embeddings_template,
        )


class Gemma4VisionPrefillExportAdapter(nn.Module):
    """Export the complete static Gemma4 vision-prefill stage.

    ``vision_model`` is specialized for one fixed ``pixel_position_ids`` profile
    through ``QuantGemma4VisionModel.as_export_module``. All profile-dependent
    tensors are owned by nested export adapters, and this stage then applies the
    multimodal projection that maps vision hidden states to text width.

    Input contract:
        ``pixel_values`` has the fixed patch layout selected at construction.

    Output contract:
        Returns visual soft tokens with shape ``(V, text_hidden_size)``. The
        fixed batch dimension is flattened by the static vision-model export.
    """

    def __init__(
        self,
        vision_model: nn.Module,
        vision_projection: nn.Module,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.vision_projection = vision_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the static vision model and project its soft tokens."""
        vision_outputs = self.vision_model(pixel_values)
        hidden_states = (
            vision_outputs
            if isinstance(vision_outputs, torch.Tensor)
            else vision_outputs.last_hidden_state
        )
        return self.vision_projection(hidden_states)


def build_gemma4_vision_prefill_export_module(
    wrapped_model: nn.Module,
    *,
    pixel_position_ids: torch.Tensor,
    mode: str = "prefill",
) -> Gemma4VisionPrefillExportAdapter:
    """Build a pixel-values-only Gemma4 vision-prefill export module.

    ``pixel_position_ids`` is construction-time specialization data. Both the
    Circle exporter and static runtime pass the canonical profile to this helper,
    but the returned module exposes only ``pixel_values`` as a runtime input.
    """
    if pixel_position_ids.dim() != 3 or pixel_position_ids.shape[0] != 1:
        raise ValueError(
            "Gemma4 vision export requires pixel_position_ids with shape "
            "(1, num_patches, 2), got "
            f"{tuple(pixel_position_ids.shape)}."
        )
    if pixel_position_ids.shape[-1] != 2:
        raise ValueError(
            "Gemma4 vision export requires two-dimensional patch coordinates, "
            f"got trailing dimension {pixel_position_ids.shape[-1]}."
        )

    vision_tower = getattr(wrapped_model, "vision_tower", None)
    if vision_tower is None:
        raise ValueError("Gemma4 vision prefill requires a vision tower.")
    quant_vision_model = getattr(vision_tower, "wrapped", vision_tower)
    as_export_module = getattr(quant_vision_model, "as_export_module", None)
    if not callable(as_export_module):
        raise TypeError("Gemma4 vision tower does not expose as_export_module().")

    vision_projection = getattr(wrapped_model, "embed_vision", None)
    if vision_projection is None:
        raise ValueError("Gemma4 vision prefill requires embed_vision.")

    vision_model = as_export_module(
        mode=mode,
        pixel_position_ids=pixel_position_ids,
    )
    return Gemma4VisionPrefillExportAdapter(
        vision_model=vision_model,
        vision_projection=vision_projection,
    )


class Gemma4MMFusionExportAdapter(nn.Module):
    """Export adapter for fixed-slot multimodal fusion."""

    def __init__(self, *, visual_start_idx: int, num_visual_tokens: int):
        super().__init__()
        self.visual_start_idx = int(visual_start_idx)
        self.num_visual_tokens = int(num_visual_tokens)

    def forward(
        self, text_embeds: torch.Tensor, visual_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Insert visual embeddings into a fixed contiguous slot range."""
        return fixed_slot_fuse(
            text_embeds,
            visual_embeds,
            visual_start_idx=self.visual_start_idx,
            num_visual_tokens=self.num_visual_tokens,
        )


class Gemma4VisionEncoderLayerPrefillExportAdapter(nn.Module):
    """Export adapter for one Gemma4 vision encoder layer.

    Input contract:
        ``hidden_states`` has shape ``(1, S, vision_hidden_size)``.
        ``attention_mask`` is a static additive or keep mask broadcastable to
        ``(1, heads, S, S)``. ``position_embeddings`` is the ``(cos, sin)`` tuple
        for the fixed patch layout.

    Output contract:
        Returns output patch states with shape ``(1, S, vision_hidden_size)``.
    """

    def __init__(self, wrapped_layer: nn.Module):
        super().__init__()
        self.wrapped = wrapped_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run a static vision encoder-layer prefill graph."""
        return self.wrapped(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )


class Gemma4VisionEncoderPrefillExportAdapter(nn.Module):
    """Export adapter for the full Gemma4 vision encoder in prefill mode.

    The adapter calls ``forward_export`` on the wrapped encoder, which reads
    pre-computed ``position_embeddings`` and ``attention_mask`` from registered
    buffers.  These buffers are materialised by ``as_export_module`` before the
    adapter is created.

    Input contract:
        ``inputs_embeds`` has shape ``(1, S, vision_hidden_size)``.

    Output contract:
        Returns output hidden states with shape ``(1, S, vision_hidden_size)``.
    """

    def __init__(self, wrapped_encoder: nn.Module):
        super().__init__()
        self.wrapped = wrapped_encoder

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run a static vision encoder prefill graph."""
        return self.wrapped.forward_export(inputs_embeds)


class Gemma4TextAttentionPrefillExportAdapter(nn.Module):
    """Export adapter for Gemma4 text attention in prefill mode.

    Input contract:
        ``hidden_states`` has shape ``(1, S, hidden_size)``.
        ``attention_mask`` is broadcastable to ``(1, heads, S, S)``.
        ``position_embeddings`` is a ``(cos, sin)`` tuple for ``S`` tokens.
        ``shared_key_value`` is optional and is used only by shared-KV consumers.

    Output contract:
        Non-shared layers return ``(hidden_states, new_key, new_value)`` when
        ``return_kv=True`` and only ``hidden_states`` otherwise. Shared-KV
        consumer layers always return only ``hidden_states`` because they do not
        own K/V projection weights.
    """

    def __init__(self, wrapped: nn.Module, *, return_kv: bool = True):
        """Store the wrapped attention module and cache-output policy."""
        super().__init__()
        self.wrapped = wrapped
        self.return_kv = bool(return_kv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        shared_key_value: Optional[LayerKV] = None,
    ) -> Any:
        """Run the fixed-shape prefill graph and return an optional K/V delta."""
        output = self.wrapped(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            shared_key_value=shared_key_value,
            past_key_value=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
        )
        return _flatten_attention_and_kv(output, return_kv=self.return_kv)


class Gemma4TextAttentionDecodeExportAdapter(nn.Module):
    """Export adapter for Gemma4 text attention in single-token decode mode.

    Input contract:
        ``hidden_states`` has shape ``(1, 1, hidden_size)``.
        ``attention_mask`` is broadcastable to ``(1, heads, 1, K)``.
        ``position_embeddings`` is a ``(cos, sin)`` tuple for one token.
        ``past_key_value`` contains ``K - 1`` cached tokens for non-shared
        layers. ``shared_key_value`` contains the full ``K`` tokens for a
        shared-KV consumer layer.

    Output contract:
        Non-shared layers return ``(hidden_states, new_key, new_value)`` when
        ``return_kv=True``. The K/V tensors contain only the current-token delta.
        Shared-KV consumer layers return only ``hidden_states``.
    """

    def __init__(self, wrapped: nn.Module, *, return_kv: bool = True):
        """Store the wrapped attention module and cache-output policy."""
        super().__init__()
        self.wrapped = wrapped
        self.return_kv = bool(return_kv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[LayerKV] = None,
        shared_key_value: Optional[LayerKV] = None,
    ) -> Any:
        """Run the fixed-shape decode graph and return an optional K/V delta."""
        output = self.wrapped(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            shared_key_value=shared_key_value,
            use_cache=self.return_kv,
            cache_output_mode="delta",
        )
        return _flatten_attention_and_kv(output, return_kv=self.return_kv)


class Gemma4TextDecoderLayerPrefillExportAdapter(nn.Module):
    """Export adapter for a Gemma4 text decoder layer in prefill mode.

    Input contract:
        ``hidden_states`` has shape ``(1, S, hidden_size)``. ``attention_mask``
        is a static additive or keep mask. ``position_embeddings`` is ``(cos,
        sin)`` for the current layer type.

    Output contract:
        If ``return_kv=True`` and the wrapped layer owns K/V projection weights,
        returns ``(hidden_states, new_key, new_value)``. Shared-KV consumer layers
        return only ``hidden_states`` because they do not produce new K/V states.
    """

    def __init__(self, wrapped_layer: nn.Module, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped_layer
        self.return_kv = bool(return_kv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        per_layer_input: Optional[torch.Tensor] = None,
        shared_key_value: Optional[LayerKV] = None,
    ):
        """Run a static prefill layer graph."""
        output = self.wrapped(
            hidden_states,
            per_layer_input=per_layer_input,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            shared_key_value=shared_key_value,
            past_key_value=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
        )
        return _flatten_hidden_and_kv(output, return_kv=self.return_kv)


class Gemma4TextDecoderLayerDecodeExportAdapter(nn.Module):
    """Export adapter for a Gemma4 text decoder layer in single-token decode mode.

    Input contract:
        ``hidden_states`` has shape ``(1, 1, hidden_size)``. ``past_key_value``
        is a fixed-size cache tuple for non-shared layers, and
        ``shared_key_value`` is a fixed-size full K/V tuple for shared-KV layers.

    Output contract:
        If ``return_kv=True`` and the wrapped layer owns K/V projection weights,
        returns ``(hidden_states, new_key, new_value)`` where ``new_key`` and
        ``new_value`` contain only the single-token delta. Shared-KV consumer
        layers return only ``hidden_states``.
    """

    def __init__(self, wrapped_layer: nn.Module, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped_layer
        self.return_kv = bool(return_kv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[LayerKV] = None,
        per_layer_input: Optional[torch.Tensor] = None,
        shared_key_value: Optional[LayerKV] = None,
    ):
        """Run a static decode layer graph and optionally return the K/V delta."""
        output = self.wrapped(
            hidden_states,
            per_layer_input=per_layer_input,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            shared_key_value=shared_key_value,
            use_cache=self.return_kv,
            cache_output_mode="delta",
        )
        return _flatten_hidden_and_kv(output, return_kv=self.return_kv)


class Gemma4VisionPoolerPrefillExportAdapter(nn.Module):
    """Export adapter for Gemma4 vision pooling with a static input contract.

    ``QuantGemma4VisionPooler.as_export_module`` precomputes the pooling weight
    matrix from ``pixel_position_ids`` and ``output_length``. Padding columns are
    zeroed before tracing, so the wrapped export path consumes no Boolean mask
    at runtime.

    Input contract:
        ``hidden_states`` has shape ``(1, S, D)`` where ``S`` is the fixed
        vision encoder sequence length.

    Output contract:
        Returns ``pooled_features`` with shape ``(1, V, D)`` in float32, where
        ``V`` is the fixed ``output_length``. Any invalid suffix rows are zero.
        ``num_valid_outputs`` records the static valid-prefix length for the
        enclosing vision-model export.
    """

    def __init__(
        self,
        wrapped_pooler: nn.Module,
    ):
        super().__init__()
        self.wrapped = wrapped_pooler
        self.num_valid_outputs = int(wrapped_pooler.num_valid_outputs)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.wrapped.forward_export(hidden_states=hidden_states)


class Gemma4LMHeadExportAdapter(nn.Module):
    """Export adapter for final normalization and LM head."""

    def __init__(self, wrapped_conditional_generation_model: nn.Module):
        super().__init__()
        wrapped_model = wrapped_conditional_generation_model.model.wrapped
        self.norm = wrapped_model.language_model.wrapped.norm
        self.lm_head = wrapped_conditional_generation_model.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return vocabulary logits for the final hidden state."""
        return self.lm_head(self.norm(hidden_states))


class Gemma4ModelPrefillExportAdapter(nn.Module):
    """Export adapter for the Gemma4Model (image-text) with static-shape contract.

    This adapter wraps a ``QuantGemma4Model`` that has been prepared for export
    via ``as_export_module()``.  Calling ``forward()`` delegates to the wrapped
    model's ``forward_export()`` method, which runs only the text decoder layers
    and final norm on pre-fused ``inputs_embeds``.

    The CPU runtime is responsible for:
    - Token embedding (with placeholder replacement)
    - Vision tower + projection
    - Multimodal fusion (fixed-slot)
    - PLE computation (if enabled)
    - Mask and RoPE generation per layer type
    - KV cache management

    Input contract:
        ``inputs_embeds`` has shape ``(1, S, hidden_size)`` — pre-fused.
        ``per_layer_inputs`` has shape ``(1, S, num_layers, ple_dim)`` or None.
        ``attention_masks`` is a dict mapping layer type to additive masks.
        ``position_embeddings`` is a dict mapping layer type to (cos, sin).

    Output contract:
        Returns the final hidden states with shape ``(1, S, hidden_size)``.
    """

    def __init__(self, wrapped_model: nn.Module):
        super().__init__()
        self.wrapped_model = wrapped_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_masks: Optional[dict] = None,
        position_embeddings: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the model export path via the wrapped model's forward_export."""
        return self.wrapped_model.forward_export(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_masks=attention_masks,
            position_embeddings=position_embeddings,
        )


class Gemma4VisionModelPrefillExportAdapter(nn.Module):
    """Export adapter for the Gemma4 vision model with static-shape contract.

    This adapter wraps a ``QuantGemma4VisionModel`` prepared through
    ``as_export_module()``. Position IDs, padding, RoPE, attention masks, and
    pooler geometry are construction-time data owned by nested adapters.

    Input contract:
        ``pixel_values`` has shape ``(1, num_patches, 3*patch_size^2)``.

    Output contract:
        Returns ``BaseModelOutputWithPast`` with ``last_hidden_state``
        containing visual soft tokens.
    """

    def __init__(
        self,
        wrapped_model: nn.Module,
    ):
        super().__init__()
        self.wrapped_model = wrapped_model

    def forward(self, pixel_values: torch.FloatTensor):
        """Run the pixel-values-only static vision model export path."""
        return self.wrapped_model.forward_export(pixel_values=pixel_values)
