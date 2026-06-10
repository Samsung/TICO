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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.wrapq.wrappers.gemma4.utils import fixed_slot_fuse


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


class Gemma4TextDecoderLayerPrefillExportAdapter(nn.Module):
    """Export adapter for a Gemma4 text decoder layer in prefill mode."""

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
        shared_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Run a static prefill layer graph.

        TODO: Implement shared-KV handling in the wrapped attention module. The
        expected return shape should be one of:

        - ``hidden_states`` when the layer is KV-shared and ``return_kv=False``.
        - ``(hidden_states, new_k, new_v)`` for non-shared layers.
        - ``(hidden_states, shared_k, shared_v)`` for store-full-length-KV layers.
        """
        return self.wrapped(
            hidden_states,
            per_layer_input=per_layer_input,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            shared_key_value=shared_key_value,
            past_key_value=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
        )


class Gemma4TextDecoderLayerDecodeExportAdapter(nn.Module):
    """Export adapter for a Gemma4 text decoder layer in single-token decode mode."""

    def __init__(self, wrapped_layer: nn.Module, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped_layer
        self.return_kv = bool(return_kv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        per_layer_input: Optional[torch.Tensor] = None,
        shared_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Run a static decode layer graph and return only the new KV delta."""
        return self.wrapped(
            hidden_states,
            per_layer_input=per_layer_input,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            shared_key_value=shared_key_value,
            use_cache=self.return_kv,
            cache_output_mode="delta",
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
