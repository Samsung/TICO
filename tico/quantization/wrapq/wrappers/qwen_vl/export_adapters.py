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

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Qwen3VLTextAttentionPrefillExportAdapter(nn.Module):
    """
    Export adapter for the Qwen3-VL text attention prefill path.

    Input contract:
        hidden_states:
            Tensor with shape `(B, S, hidden_size)`.
        position_embeddings:
            Tuple `(cos, sin)` where each tensor has shape `(B, S, head_dim)`.
        attention_mask:
            Optional additive mask with shape broadcastable to `(B, 1, S, S)`.

    Return contract when `return_kv=True`:
        `(hidden_states, new_key, new_value)`, where:
            hidden_states has shape `(B, S, hidden_size)`;
            new_key and new_value have shape `(B, num_kv_heads, S, head_dim)`.

    Return contract when `return_kv=False`:
        `hidden_states`.
    """

    def __init__(self, wrapped: nn.Module, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run prefill attention and optionally return the newly produced KV tensors."""
        outputs = self.wrapped(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[2]
        return hidden, new_k, new_v


class Qwen3VLTextAttentionDecodeExportAdapter(nn.Module):
    """
    Export adapter for the Qwen3-VL text attention decode path.

    Input contract:
        hidden_states:
            Tensor with shape `(B, 1, hidden_size)`.
        position_embeddings:
            Tuple `(cos, sin)` where each tensor has shape `(B, 1, head_dim)`.
        attention_mask:
            Optional additive mask with shape broadcastable to `(B, 1, 1, K)`.
        past_key_values:
            Tuple `(past_key, past_value)` where each tensor has shape
            `(B, num_kv_heads, K - 1, head_dim)`.

    Return contract when `return_kv=True`:
        `(hidden_states, new_key, new_value)`, where new_key and new_value are
        the KV delta for the current token with shape
        `(B, num_kv_heads, 1, head_dim)`.

    Return contract when `return_kv=False`:
        `hidden_states`.
    """

    def __init__(self, wrapped: nn.Module, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """Run decode attention and optionally return the current-token KV delta."""
        outputs = self.wrapped(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[2]
        return hidden, new_k, new_v


class Qwen3VLTextDecoderLayerPrefillExportAdapter(nn.Module):
    """
    Export adapter for the Qwen3-VL text decoder-layer prefill path.

    Input contract:
        hidden_states:
            Tensor with shape `(B, S, hidden_size)`.
        attention_mask:
            Additive mask with shape broadcastable to `(B, 1, S, S)`.
        position_embeddings:
            Tuple `(cos, sin)` where each tensor has shape `(B, S, head_dim)`.

    Return contract when `return_kv=True`:
        `(hidden_states, new_key, new_value)`, where:
            hidden_states has shape `(B, S, hidden_size)`;
            new_key and new_value have shape `(B, num_kv_heads, S, head_dim)`.

    Return contract when `return_kv=False`:
        `hidden_states`.
    """

    def __init__(self, wrapped: nn.Module, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ):
        """Run prefill and optionally return the newly produced KV tensors."""
        outputs = self.wrapped(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[1]
        return hidden, new_k, new_v


class Qwen3VLTextDecoderLayerDecodeExportAdapter(nn.Module):
    """
    Export adapter for the Qwen3-VL text decoder-layer decode path.

    Input contract:
        hidden_states:
            Tensor with shape `(B, 1, hidden_size)`.
        attention_mask:
            Additive mask with shape broadcastable to `(B, 1, 1, K)`.
        position_embeddings:
            Tuple `(cos, sin)` where each tensor has shape `(B, 1, head_dim)`.
        past_key_values:
            Tuple `(past_key, past_value)` where each tensor has shape
            `(B, num_kv_heads, K - 1, head_dim)`.

    Return contract when `return_kv=True`:
        `(hidden_states, new_key, new_value)`, where new_key and new_value are
        the KV delta for the current token with shape
        `(B, num_kv_heads, 1, head_dim)`.

    Return contract when `return_kv=False`:
        `hidden_states`.
    """

    def __init__(self, wrapped: nn.Module, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ):
        """Run decode and optionally return the current-token KV delta."""
        outputs = self.wrapped(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[1]
        return hidden, new_k, new_v


class Qwen3VLVisionPrefillExportAdapter(nn.Module):
    """
    Export adapter for the fixed-grid Qwen3-VL vision prefill path.

    Input contract:
        pixel_values:
            Flattened image patch tensor produced by the Qwen3-VL processor.
        image_grid_thw:
            Static image grid tensor with shape `(1, 3)`.

    Return contract:
        `(image_embeds, deepstack_features)`, where `image_embeds` is the merged
        visual token tensor used to replace image placeholder tokens and
        `deepstack_features` is a tuple of merged DeepStack tensors. Each tensor
        is statically sized by the fixed `image_grid_thw` used during export.
    """

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped

    @staticmethod
    def _unwrap_vision_output(vision_output):
        """Normalize Qwen3-VL vision outputs into image embeds and DeepStack features."""
        if hasattr(vision_output, "pooler_output"):
            image_embeds = vision_output.pooler_output
            deepstack_features = getattr(vision_output, "deepstack_features", None)
        elif isinstance(vision_output, (tuple, list)) and len(vision_output) >= 2:
            image_embeds, deepstack_features = vision_output[0], vision_output[1]
        else:
            image_embeds = vision_output
            deepstack_features = None

        if deepstack_features is None:
            deepstack_features = ()
        elif isinstance(deepstack_features, list):
            deepstack_features = tuple(deepstack_features)

        return image_embeds, deepstack_features

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ):
        """Run fixed-grid vision prefill and return merged visual features."""
        vision_output = self.wrapped(pixel_values, grid_thw=image_grid_thw, **kwargs)
        return self._unwrap_vision_output(vision_output)


class Qwen3VLVisualEmbeddingFusionAdapter(nn.Module):
    """
    Static adapter that fuses single-image embeddings into text embeddings.

    This adapter intentionally assumes one contiguous visual-token span. The
    visual span start is fixed at construction time so the exported graph avoids
    dynamic `nonzero`, boolean indexing, or scatter operators.
    """

    def __init__(self, visual_start_idx: int):
        super().__init__()
        if visual_start_idx < 0:
            raise ValueError("visual_start_idx must be non-negative.")
        self.visual_start_idx = int(visual_start_idx)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Replace the fixed visual-token slice with image embeddings."""
        if inputs_embeds.dim() != 3:
            raise RuntimeError(
                "inputs_embeds must have shape `(B, S, H)`, "
                f"got {tuple(inputs_embeds.shape)}."
            )
        if image_embeds.dim() != 2:
            raise RuntimeError(
                "image_embeds must have shape `(V, H)`, "
                f"got {tuple(image_embeds.shape)}."
            )

        visual_len = image_embeds.size(0)
        visual_end = self.visual_start_idx + visual_len
        if visual_end > inputs_embeds.size(1):
            raise RuntimeError(
                "The visual embedding span exceeds the input sequence length: "
                f"start={self.visual_start_idx}, len={visual_len}, "
                f"seq_len={inputs_embeds.size(1)}."
            )

        fused = inputs_embeds.clone()
        fused[:, self.visual_start_idx : visual_end, :] = image_embeds.unsqueeze(0).to(
            device=fused.device,
            dtype=fused.dtype,
        )
        return fused
