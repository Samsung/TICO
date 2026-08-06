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


def _find_vision_grid_thw(module: nn.Module) -> torch.Tensor:
    """Find the fixed vision grid owned by a wrapped Qwen3-VL vision model."""
    current: Optional[nn.Module] = module
    visited: set[int] = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        grid_thw = getattr(current, "vision_grid_thw", None)
        if isinstance(grid_thw, torch.Tensor):
            return grid_thw

        wrapped = getattr(current, "wrapped", None)
        current = wrapped if isinstance(wrapped, nn.Module) else None

    raise ValueError(
        "Qwen3VLVisionPrefillExportAdapter requires a wrapped vision model "
        "with fixed vision_grid_thw metadata."
    )


def _make_attention_split_sizes(grid_thw: torch.Tensor) -> tuple[int, ...]:
    """Build static per-frame attention split sizes from a fixed THW grid."""
    if grid_thw.dim() != 2 or grid_thw.size(1) != 3:
        raise ValueError(
            "vision_grid_thw must have shape `(N, 3)`, " f"got {tuple(grid_thw.shape)}."
        )

    split_sizes: list[int] = []
    for temporal, height, width in grid_thw.detach().cpu().tolist():
        temporal = int(temporal)
        height = int(height)
        width = int(width)
        if temporal <= 0 or height <= 0 or width <= 0:
            raise ValueError(
                "vision_grid_thw values must be positive, "
                f"got {(temporal, height, width)}."
            )
        split_sizes.extend([height * width] * temporal)

    return tuple(split_sizes)


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
            Flattened image patches. The processor-native shape is
            `(num_patches, patch_dim)`; static NPU export uses
            `(1, num_patches, patch_dim)`.
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
        self.attention_split_sizes = _make_attention_split_sizes(
            _find_vision_grid_thw(wrapped)
        )

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
        vision_output = self.wrapped(
            pixel_values,
            grid_thw=image_grid_thw,
            attention_split_sizes=self.attention_split_sizes,
            **kwargs,
        )
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


def _unwrap_qwen3_vl_text_model(wrapped: nn.Module) -> nn.Module:
    """Return the wrapped Qwen3-VL text model from the top-level wrapper."""
    qwen_model = wrapped.model.wrapped
    return qwen_model.language_model.wrapped


class Qwen3VLTextEmbeddingExportAdapter(nn.Module):
    """Export dynamic token embedding and optional SpinQuant rotation.

    The embedding wrapper already quantizes its output. When an embedding
    rotation exists, its wrapped linear module owns the next input/output
    quantization boundary. No model-level observer is replayed here.

    Input contract:
        input_ids: Token IDs with shape ``(1, S)``.

    Return contract:
        Text hidden states with shape ``(1, S, hidden_size)`` ready for a
        separately exported decoder layer.
    """

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        text_model = _unwrap_qwen3_vl_text_model(wrapped)
        self.embed_tokens = text_model.embed_tokens
        self.rotate_embedding = getattr(text_model, "rotate_embedding", None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to decoder-ready hidden states."""
        hidden_states = self.embed_tokens(input_ids)
        if self.rotate_embedding is not None:
            hidden_states = self.rotate_embedding(hidden_states)
        return hidden_states


class Qwen3VLMultimodalEmbeddingExportAdapter(nn.Module):
    """Export fixed-span visual embedding insertion and optional rotation.

    The fusion itself does not replay ``obs_mm_fusion`` or
    ``obs_inputs_embeds``. Quantization is already provided by the token
    embedding output, the optional rotation wrapper, or the next decoder
    layer's input observer.

    Input contract:
        input_ids: Token IDs with shape ``(1, S)``.
        image_embeds: Merged visual embeddings with shape ``(V, hidden_size)``.

    Return contract:
        Multimodal hidden states with shape ``(1, S, hidden_size)``.
    """

    def __init__(self, wrapped: nn.Module, *, visual_start_idx: int):
        super().__init__()
        if visual_start_idx < 0:
            raise ValueError("visual_start_idx must be non-negative.")

        text_model = _unwrap_qwen3_vl_text_model(wrapped)
        self.visual_start_idx = int(visual_start_idx)
        self.embed_tokens = text_model.embed_tokens
        self.rotate_embedding = getattr(text_model, "rotate_embedding", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Insert fixed-grid image embeddings into the token embedding sequence."""
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise RuntimeError(
                "input_ids must have shape `(1, S)`, " f"got {tuple(input_ids.shape)}."
            )
        if image_embeds.dim() != 2:
            raise RuntimeError(
                "image_embeds must have shape `(V, H)`, "
                f"got {tuple(image_embeds.shape)}."
            )

        hidden_states = self.embed_tokens(input_ids)
        if image_embeds.size(-1) != hidden_states.size(-1):
            raise RuntimeError(
                "image_embeds hidden size does not match token embeddings: "
                f"image_hidden={image_embeds.size(-1)}, "
                f"text_hidden={hidden_states.size(-1)}."
            )

        visual_end = self.visual_start_idx + image_embeds.size(0)
        if visual_end > hidden_states.size(1):
            raise RuntimeError(
                "The visual embedding span exceeds the input sequence length: "
                f"start={self.visual_start_idx}, visual_end={visual_end}, "
                f"seq_len={hidden_states.size(1)}."
            )

        hidden_states = hidden_states.clone()
        hidden_states[
            :, self.visual_start_idx : visual_end, :
        ] = image_embeds.unsqueeze(0).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if self.rotate_embedding is not None:
            hidden_states = self.rotate_embedding(hidden_states)
        return hidden_states


class Qwen3VLDeepstackFusionExportAdapter(nn.Module):
    """Export one fixed-span DeepStack residual insertion after a text layer.

    This adapter performs only the residual addition. The next decoder layer's
    input observer, or the final norm input observer after the last layer,
    provides the following quantization boundary.

    Input contract:
        hidden_states: Decoder hidden states with shape ``(1, S, hidden_size)``.
        visual_embeds: One DeepStack feature tensor with shape
            ``(V, hidden_size)``.

    Return contract:
        Updated hidden states with shape ``(1, S, hidden_size)``.
    """

    def __init__(self, *, visual_start_idx: int):
        super().__init__()
        if visual_start_idx < 0:
            raise ValueError("visual_start_idx must be non-negative.")
        self.visual_start_idx = int(visual_start_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Add a DeepStack feature tensor to its fixed visual-token span."""
        if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
            raise RuntimeError(
                "hidden_states must have shape `(1, S, H)`, "
                f"got {tuple(hidden_states.shape)}."
            )
        if visual_embeds.dim() != 2:
            raise RuntimeError(
                "visual_embeds must have shape `(V, H)`, "
                f"got {tuple(visual_embeds.shape)}."
            )
        if visual_embeds.size(-1) != hidden_states.size(-1):
            raise RuntimeError(
                "visual_embeds hidden size does not match hidden_states: "
                f"visual_hidden={visual_embeds.size(-1)}, "
                f"text_hidden={hidden_states.size(-1)}."
            )

        visual_end = self.visual_start_idx + visual_embeds.size(0)
        if visual_end > hidden_states.size(1):
            raise RuntimeError(
                "The DeepStack span exceeds the input sequence length: "
                f"start={self.visual_start_idx}, visual_end={visual_end}, "
                f"seq_len={hidden_states.size(1)}."
            )

        output = hidden_states.clone()
        output[:, self.visual_start_idx : visual_end, :] = output[
            :, self.visual_start_idx : visual_end, :
        ] + visual_embeds.unsqueeze(0).to(
            device=output.device,
            dtype=output.dtype,
        )
        return output


class Qwen3VLLMHeadExportAdapter(nn.Module):
    """Export Qwen3-VL final normalization, optional rotation, and LM head.

    Input contract:
        hidden_states: Last-layer hidden states with shape ``(1, Q, hidden_size)``.

    Return contract:
        Vocabulary logits with shape ``(1, Q, vocab_size)``.
    """

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        text_model = _unwrap_qwen3_vl_text_model(wrapped)
        self.norm = text_model.norm
        self.rotate_lm_head: Optional[nn.Module] = getattr(
            wrapped,
            "rotate_lm_head",
            None,
        )
        self.lm_head = wrapped.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return vocabulary logits for the supplied decoder hidden states."""
        hidden_states = self.norm(hidden_states)
        if self.rotate_lm_head is not None:
            hidden_states = self.rotate_lm_head(hidden_states)
        return self.lm_head(hidden_states)
