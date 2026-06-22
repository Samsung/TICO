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


class Gemma4VisionPrefillExportAdapter(nn.Module):
    """Export adapter for Gemma4 vision tower and multimodal projection.

    Input contract:
        ``pixel_values`` and ``image_position_ids`` must use the static shape
        selected by the runtime profile.

    Output contract:
        Returns visual soft tokens with shape ``(1, V, text_hidden_size)``.
    """

    def __init__(self, wrapped_model: nn.Module):
        super().__init__()
        self.vision_tower = wrapped_model.vision_tower
        self.embed_vision = wrapped_model.embed_vision

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the vision tower and project features into text hidden space."""
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            return_dict=True,
        )
        return self.embed_vision(vision_outputs.last_hidden_state)


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
        for the fixed patch layout, and ``position_ids`` is the optional static
        2-D pixel coordinate tensor shaped ``(1, S, 2)``.

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
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a static vision encoder-layer prefill graph."""
        return self.wrapped(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )


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
    """Export adapter for the Gemma4 vision pooler with static-shape contract.

    This adapter is a **self-contained quantization wrapper** that inherits from
    ``QuantModuleBase`` and owns its own observers.  It replaces the original
    pooler's dynamic operations (``F.one_hot``, ``torch.div``, data-dependent
    conditionals) with a decomposed, ``torch.export``-friendly implementation
    that uses a **precomputed weight matrix** and **precomputed output mask**
    stored as buffers.

    The relationship between ``QuantGemma4VisionPooler`` and this adapter is:

    - ``QuantGemma4VisionPooler`` is more flexible — it supports conditional
      branches and dynamic tensor shapes by delegating to the original module,
      but it **cannot** be exported and converted to Circle.
    - ``Gemma4VisionPoolerPrefillExportAdapter`` allows only static computations and
      tensor shapes, but it **can** be exported and converted to Circle.

    The weight matrix and mask are deterministic given the fixed image profile
    (``seq_len``, ``output_length``, ``pixel_position_ids``), so they are
    computed once at construction time and never change at runtime.

    The adapter bakes ``output_length`` (the number of visual soft tokens) into
    the graph at construction time so that it is not a runtime argument.  This
    satisfies the static-shape contract required by ``torch.export`` and the
    NPU runtime.

    The CPU runtime is responsible for:
    - Pre-computing ``pixel_position_ids`` as a fixed-shape tensor.
    - Pre-computing ``padding_positions`` as a fixed-shape boolean mask.
    - Ensuring that the input sequence length and ``output_length`` are
      compatible with the static profile.

    Input contract:
        ``hidden_states`` has shape ``(1, S, D)`` where ``S`` is the fixed
        vision encoder sequence length.
        ``pixel_position_ids`` has shape ``(1, S, 2)`` — pre-computed on CPU.
        ``padding_positions`` has shape ``(1, S)`` — pre-computed on CPU.

    Output contract:
        Returns a tuple ``(pooled_features, updated_padding)`` where
        ``pooled_features`` has shape ``(1, V, D)`` in float32 with ``V``
        equal to the fixed ``output_length``, and ``updated_padding`` has
        shape ``(1, V)``.
    """

    def __init__(
        self,
        wrapped_pooler: nn.Module,
    ):
        super().__init__()
        self.wrapped = wrapped_pooler

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.wrapped.forward_export(
            hidden_states=hidden_states, padding_positions=padding_positions
        )


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
