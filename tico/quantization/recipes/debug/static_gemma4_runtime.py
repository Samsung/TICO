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

"""Static-shape runtime skeleton for Gemma4 E2B.

This module mirrors the Llama static runtime design while adding a fixed image
prefill stage. CPU code owns processor/tokenizer logic, static layout checks,
RoPE and mask generation, KV cache writes, shared-KV bookkeeping, and sampling.
NPU-exportable subgraphs own quantized tensor compute.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoProcessor

from tico.quantization import prepare
from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4LMHeadExportAdapter,
    Gemma4MMFusionExportAdapter,
    Gemma4TokenEmbeddingExportAdapter,
    Gemma4VisionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    assert_gemma4_e2b_no_moe,
    build_decode_attention_mask,
    StaticGemma4Layout,
)

# =============================================================================
# Phase 1: CPU Helper Functions (pure Python, no model needed)
# =============================================================================


def _build_gemma4_rope_templates(
    config,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Build per-layer-type RoPE cos/sin tables for Gemma4.

    Gemma4 uses different RoPE configurations per layer type:
    - "full_attention" layers: use "proportional" RoPE with partial_rotary_factor
    - "sliding_attention" layers: use "default" RoPE with rope_theta

    Args:
        config: Gemma4 text config with layer_types and rope_parameters.
        max_seq: Maximum sequence length for the tables.
        device: Target device for the tensors.
        dtype: Target dtype for the tensors.

    Returns:
        Dict mapping layer_type -> (cos, sin) tensors of shape (1, max_seq, head_dim).
    """
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    global_head_dim = getattr(config, "global_head_dim", None) or head_dim
    layer_types = getattr(config, "layer_types", ["full_attention"])

    # Get rope parameters
    rope_params = getattr(config, "rope_parameters", {}) or {}

    # Build RoPE for each unique layer type
    result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for layer_type in set(layer_types):
        # Determine head_dim for this layer type:
        # full_attention uses global_head_dim, sliding_attention uses head_dim
        if layer_type == "full_attention" and global_head_dim:
            dim = int(global_head_dim)
        else:
            dim = int(head_dim)

        # Determine RoPE config for this layer type
        layer_rope_params = rope_params.get(layer_type, {})
        if isinstance(layer_rope_params, dict):
            theta = float(layer_rope_params.get("rope_theta", 10000.0))
            factor = float(layer_rope_params.get("partial_rotary_factor", 1.0))
        else:
            theta = 10000.0
            factor = 1.0

        # Compute rotary frequency.
        # For proportional RoPE (partial_rotary_factor < 1.0), HF computes
        # inv_freq with head_dim in the denominator (NOT rotary_dim), then pads
        # inv_freq with zeros to head_dim//2 elements. This ensures non-rotary
        # dimensions get cos=1, sin=0 (identity rotation) via cos(0)=1, sin(0)=0.
        rotary_dim = int(dim * factor)
        rope_angles = rotary_dim // 2  # number of rotary frequency components

        # inv_freq for rotary dimensions: divide by dim (head_dim), NOT rotary_dim
        inv_freq_rotated = 1.0 / (
            theta
            ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device)
                / dim
            )
        )

        # Pad inv_freq with zeros for non-rotary dimensions
        nope_angles = dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                (
                    inv_freq_rotated,
                    torch.zeros(nope_angles, dtype=torch.float32, device=device),
                ),
                dim=0,
            )
        else:
            inv_freq = inv_freq_rotated

        # Compute freqs and emb: cat(freqs, freqs) gives head_dim elements
        pos = torch.arange(max_seq, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)  # (max_seq, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq, head_dim)

        cos = emb.cos().unsqueeze(0).to(dtype=dtype)
        sin = emb.sin().unsqueeze(0).to(dtype=dtype)

        result[layer_type] = (cos, sin)

    return result


def _build_gemma4_prefill_masks(
    valid_token_mask: torch.Tensor,
    layer_types: List[str],
    sliding_window: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -1e9,
) -> Dict[str, torch.Tensor]:
    """Build per-layer-type attention masks for prefill.

    Args:
        valid_token_mask: Boolean mask of shape (B, S) indicating valid tokens.
        layer_types: List of layer types for each decoder layer.
        sliding_window: Sliding window size for "sliding_attention" layers.
        device: Target device.
        dtype: Target dtype.
        mask_value: Value to use for masked positions (negative large number).

    Returns:
        Dict mapping layer_type -> additive attention mask of shape (B, 1, S, S).
    """
    batch_size, seq_len = valid_token_mask.shape
    result: Dict[str, torch.Tensor] = {}

    for layer_type in set(layer_types):
        if layer_type == "sliding_attention" and sliding_window is not None:
            # Sliding window causal mask
            # Query at position q attends to keys in [max(0, q - sliding_window + 1), q]
            q_pos = torch.arange(seq_len, device=device).view(1, -1, 1)
            k_pos = torch.arange(seq_len, device=device).view(1, 1, -1)
            causal_mask = q_pos >= k_pos
            window_mask = (q_pos - k_pos) < sliding_window
            mask = ~(causal_mask & window_mask)
        else:
            # Standard causal mask for "full_attention"
            q_pos = torch.arange(seq_len, device=device).view(1, -1, 1)
            k_pos = torch.arange(seq_len, device=device).view(1, 1, -1)
            mask = q_pos < k_pos

        # Apply valid token mask (both query and key sides)
        valid_2d = valid_token_mask.unsqueeze(1)  # (B, 1, S)
        invalid_q = ~valid_token_mask  # (B, S)
        invalid_k = ~valid_token_mask  # (B, S)
        mask = mask | invalid_q.unsqueeze(-1) | invalid_k.unsqueeze(-2)

        # Convert to additive mask
        additive_mask = torch.zeros(
            batch_size, 1, seq_len, seq_len, device=device, dtype=dtype
        )
        additive_mask = additive_mask.masked_fill(mask.unsqueeze(1), mask_value)

        result[layer_type] = additive_mask

    return result


def _build_gemma4_decode_masks(
    batch_size: int,
    past_len: int,
    max_seq: int,
    layer_types: List[str],
    sliding_window: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -1e9,
) -> Dict[str, torch.Tensor]:
    """Build per-layer-type attention masks for one decode step.

    Args:
        batch_size: Batch size.
        past_len: Number of past tokens (KV cache length).
        max_seq: Maximum sequence length.
        layer_types: List of layer types.
        sliding_window: Sliding window size for "sliding_attention" layers.
        device: Target device.
        dtype: Target dtype.
        mask_value: Value for masked positions.

    Returns:
        Dict mapping layer_type -> additive attention mask of shape (B, 1, past_len+1).
    """
    result: Dict[str, torch.Tensor] = {}
    k_len = past_len + 1  # past keys + new key

    for layer_type in set(layer_types):
        if layer_type == "sliding_attention" and sliding_window is not None:
            # Sliding window: only last sliding_window tokens are visible
            start_idx = max(0, past_len - sliding_window + 1)
        else:
            # Full attention: all past tokens are visible
            start_idx = 0

        # Build mask sized to actual key length (past_len + 1)
        mask = torch.ones(batch_size, 1, k_len, device=device, dtype=dtype)
        mask[:, :, start_idx:] = 0.0
        mask = mask * mask_value

        result[layer_type] = mask

    return result


def _apply_logit_softcapping(
    logits: torch.Tensor,
    final_logit_softcapping: Optional[float],
) -> torch.Tensor:
    """Apply logit softcapping: tanh(logits / softcap) * softcap.

    Args:
        logits: Input logits tensor.
        final_logit_softcapping: Softcapping threshold (None to skip).

    Returns:
        Softcapped logits.
    """
    if final_logit_softcapping is None:
        return logits
    return torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping


def _normalize_valid_token_mask(
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    *,
    pad_token_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Normalize attention mask to a boolean valid-token mask.

    Mirrors the Llama runtime helper.
    """
    if attention_mask is None:
        if pad_token_id is None:
            valid = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        else:
            valid = input_ids.to(device).ne(int(pad_token_id))
    else:
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} != input_ids shape {tuple(input_ids.shape)}"
            )
        valid = attention_mask.to(device).bool()
    return valid


def _validate_padding_layout(
    input_ids: torch.LongTensor,
    valid_token_mask: torch.Tensor,
    *,
    padding_side: str,
) -> None:
    """Validate that padding is on the expected side.

    Mirrors the Llama runtime helper.
    """
    if padding_side == "right":
        # All valid tokens should be on the left, padding on the right
        for i in range(valid_token_mask.size(0)):
            row = valid_token_mask[i]
            # Find first False
            false_indices = torch.where(~row)[0]
            if len(false_indices) > 0:
                first_false = int(false_indices[0].item())
                # All tokens after first_false should be False
                if not torch.all(~row[first_false:]):
                    raise ValueError("Right padding expected but not found")
    elif padding_side == "left":
        # All valid tokens should be on the right, padding on the left
        for i in range(valid_token_mask.size(0)):
            row = valid_token_mask[i]
            true_indices = torch.where(row)[0]
            if len(true_indices) > 0:
                first_true = int(true_indices[0].item())
                # All tokens before first_true should be False
                if first_true > 0 and torch.any(row[:first_true]):
                    raise ValueError("Left padding expected but not found")


def _build_position_ids_from_valid_token_mask(
    valid_token_mask: torch.Tensor,
) -> torch.LongTensor:
    """Build position IDs from a valid token mask.

    Mirrors the Llama runtime helper.
    """
    batch_size, seq_len = valid_token_mask.shape
    position_ids = (
        torch.arange(seq_len, device=valid_token_mask.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    return position_ids


def _gather_last_token_logits(
    logits: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    padding_side: str,
) -> torch.Tensor:
    """Gather logits for the last valid token in each sequence.

    Mirrors the Llama runtime helper.
    """
    batch_size, seq_len, vocab_size = logits.shape

    if padding_side == "right":
        # Last valid token is at position valid_length - 1
        valid_lengths = valid_token_mask.sum(dim=1)  # (B,)
        gather_indices = (
            (valid_lengths - 1)
            .clamp(min=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, 1, vocab_size)
        )
        return logits.gather(1, gather_indices).squeeze(1)
    else:
        # Left padding: last valid token is at the end of valid region
        valid_lengths = valid_token_mask.sum(dim=1)
        gather_indices = (
            (valid_lengths - 1)
            .clamp(min=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, 1, vocab_size)
        )
        return logits.gather(1, gather_indices).squeeze(1)


def _gather_rope_by_position_ids(
    rope_tables: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    position_ids: torch.LongTensor,
    layer_types: List[str],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Gather RoPE cos/sin at specific positions for each layer type.

    Args:
        rope_tables: Dict of layer_type -> (cos, sin) with shape (1, max_seq, head_dim).
        position_ids: Position IDs of shape (B, S).
        layer_types: List of layer types for each layer.

    Returns:
        Dict of layer_type -> (cos, sin) gathered at position_ids.
    """
    batch_size, seq_len = position_ids.shape
    result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for layer_type in set(layer_types):
        if layer_type not in rope_tables:
            continue
        cos_full, sin_full = rope_tables[layer_type]
        # cos_full: (1, max_seq, head_dim)
        # Gather at position_ids: (B, S) -> (B, S, head_dim)
        cos = cos_full[0, position_ids]  # (B, S, head_dim)
        sin = sin_full[0, position_ids]  # (B, S, head_dim)
        result[layer_type] = (cos, sin)

    return result


# =============================================================================
# Phase 2: Data Classes
# =============================================================================


@dataclass
class LayerCache:
    """Static per-layer KV cache."""

    past_k: torch.Tensor
    past_v: torch.Tensor


@dataclass
class StaticGemma4RuntimeConfig:
    """Configuration for the Gemma4 E2B static runtime smoke flow."""

    model: str = "google/gemma-4-e2b-it"
    max_seq: int = 2048
    image_height: int = 896
    image_width: int = 896
    visual_start_idx: int = 0
    num_visual_tokens: int = 256
    padding_side: str = "right"
    device: str = "cpu"
    prompt: str = "<|image|>Describe the image."
    verify_steps: int = 4
    gen_steps: int = 16


class StaticGemma4Runtime:
    """CPU-orchestrated static runtime for Gemma4 E2B."""

    def __init__(
        self,
        model: nn.Module,
        processor: AutoProcessor,
        *,
        layout: StaticGemma4Layout,
        device: str = "cpu",
    ):
        """Create a runtime around a Gemma4 E2B model."""
        layout.validate()
        assert_gemma4_e2b_no_moe(model)

        self.model = model.eval().to(device)
        self.processor = processor
        self.layout = layout
        self.device = torch.device(device)
        self.config = model.config
        self.text_config = model.config.get_text_config()

        qcfg = build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_config.num_hidden_layers),
            num_vision_layers=int(model.config.vision_config.num_hidden_layers),
            model_args={
                "vision": {
                    "visual_start_idx": layout.visual_start_idx,
                    "num_visual_tokens": layout.num_visual_tokens,
                }
            },
        )
        self.qmodel = prepare(model, qcfg).to(self.device).eval()

        wrapped_top = (
            self.qmodel.wrapped if hasattr(self.qmodel, "wrapped") else self.qmodel
        )
        wrapped_model = wrapped_top.model.wrapped

        self.token_embedding = Gemma4TokenEmbeddingExportAdapter(
            wrapped_model.language_model.wrapped
        ).to(self.device)
        self.vision_prefill = Gemma4VisionPrefillExportAdapter(wrapped_model).to(
            self.device
        )
        self.mm_fusion = Gemma4MMFusionExportAdapter(
            visual_start_idx=layout.visual_start_idx,
            num_visual_tokens=layout.num_visual_tokens,
        ).to(self.device)
        self.lm_head = Gemma4LMHeadExportAdapter(wrapped_top).to(self.device)

        self.prefill_layers = nn.ModuleList(
            [
                layer.wrapped.as_export_module("prefill", return_kv=True)
                for layer in wrapped_model.language_model.wrapped.layers
            ]
        ).to(self.device)
        self.decode_layers = nn.ModuleList(
            [
                layer.wrapped.as_export_module("decode", return_kv=True)
                for layer in wrapped_model.language_model.wrapped.layers
            ]
        ).to(self.device)

        # Pre-build RoPE templates for all layer types
        self.rope_tables = _build_gemma4_rope_templates(
            self.text_config,
            self.layout.max_seq,
            self.device,
            torch.float32,
        )

        # Get sliding window config
        self.sliding_window = getattr(self.text_config, "sliding_window", None)

        # Check if PLE (Per-Layer Embeddings) is enabled
        self.hidden_size_per_layer_input = int(
            getattr(self.text_config, "hidden_size_per_layer_input", 0) or 0
        )

        # Store reference to wrapped text model for PLE computation
        self._wrapped_text_model = wrapped_model.language_model.wrapped

        self.layer_caches: list[LayerCache] = []
        self.past_len = 0

    def reset_cache(self) -> None:
        """Reset all runtime-managed KV caches."""
        self.layer_caches = []
        self.past_len = 0

    def _allocate_empty_cache(
        self, batch_size: int, dtype: torch.dtype
    ) -> list[LayerCache]:
        """Allocate fixed-size empty KV cache tensors.

        Each layer may have a different head_dim: full_attention layers use
        global_head_dim while sliding_attention layers use head_dim.
        """
        num_kv_heads = int(self.text_config.num_key_value_heads)
        head_dim = int(self.text_config.head_dim)
        global_head_dim = int(
            getattr(self.text_config, "global_head_dim", None) or head_dim
        )
        layer_types = getattr(self.text_config, "layer_types", ["full_attention"])
        caches = []
        for i in range(int(self.text_config.num_hidden_layers)):
            layer_type = layer_types[i] if i < len(layer_types) else "full_attention"
            if layer_type == "full_attention" and global_head_dim:
                layer_head_dim = global_head_dim
            else:
                layer_head_dim = head_dim
            past_k = torch.zeros(
                batch_size,
                num_kv_heads,
                self.layout.max_seq,
                layer_head_dim,
                device=self.device,
                dtype=dtype,
            )
            caches.append(LayerCache(past_k=past_k, past_v=torch.zeros_like(past_k)))
        return caches

    def build_static_inputs(
        self,
        prompt: str,
        image,
        max_seq: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Build static padded processor inputs for Gemma4 E2B.

        Args:
            prompt: Text prompt string.
            image: PIL Image or numpy array.
            max_seq: Maximum sequence length (defaults to self.layout.max_seq).

        Returns:
            Dict with keys:
                - llm_input_ids: (B, max_seq) padded input IDs
                - pixel_values: (B, C, H, W) image tensor
                - image_position_ids: (num_image_tokens,) or None
                - attention_mask: (B, max_seq) boolean valid token mask
                - valid_length: (1,) number of valid tokens
        """
        if max_seq is None:
            max_seq = self.layout.max_seq

        # Get pad token ID
        pad_token_id = getattr(self.text_config, "pad_token_id", 0)

        # Process prompt and image through HF processor
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=False,  # We'll pad manually to max_seq
        )

        input_ids = inputs["input_ids"].squeeze(0)  # (seq_len,)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)

        # Validate padding layout
        valid_token_mask = _normalize_valid_token_mask(
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0) if attention_mask is not None else None,
            pad_token_id=pad_token_id,
            device=self.device,
        ).squeeze(0)

        _validate_padding_layout(
            input_ids.unsqueeze(0),
            valid_token_mask.unsqueeze(0),
            padding_side=(
                self.layout.padding_side
                if hasattr(self.layout, "padding_side")
                else "right"
            ),
        )

        # Pad to max_seq
        seq_len = input_ids.shape[0]
        if seq_len > max_seq:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq {max_seq}"
            )

        # Pad input_ids
        padded_input_ids = torch.full(
            (max_seq,), pad_token_id, dtype=input_ids.dtype, device=self.device
        )
        padded_input_ids[:seq_len] = input_ids

        # Replace image placeholder token IDs with pad_token_id to create
        # llm_input_ids.  The visual embeddings are fused separately via
        # mm_fusion, so the text embedding path should not see image placeholder
        # tokens.  This follows the spec in .clinerules/Gemma.md:
        #   "Create llm_input_ids: replace image/video/audio placeholder token
        #    IDs with pad_token_id."
        # Note: image_token_id lives on the top-level model config, not on
        # text_config.
        image_token_id = getattr(self.config, "image_token_id", None)
        if image_token_id is not None:
            padded_input_ids[padded_input_ids == image_token_id] = pad_token_id

        # Pad attention mask
        padded_attention_mask = torch.zeros(
            max_seq, dtype=torch.bool, device=self.device
        )
        padded_attention_mask[:seq_len] = True

        # Get pixel_values
        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is None:
            raise ValueError("Processor did not return pixel_values")
        pixel_values = pixel_values.to(self.device)

        # Image position IDs (if available)
        image_position_ids = inputs.get("image_position_ids", None)
        if image_position_ids is not None:
            image_position_ids = image_position_ids.to(self.device)

        valid_length = torch.tensor([seq_len], dtype=torch.long, device=self.device)

        return {
            "llm_input_ids": padded_input_ids.unsqueeze(0),  # (1, max_seq)
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
            "attention_mask": padded_attention_mask.unsqueeze(0),  # (1, max_seq)
            "valid_length": valid_length,
        }

    def build_prefill_masks_and_rope(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Build CPU-owned static masks and RoPE tensors for prefill.

        Args:
            input_ids: (B, max_seq) input IDs.
            attention_mask: (B, max_seq) boolean valid token mask.

        Returns:
            Tuple of:
                - attention_masks: Dict[layer_type, (B, 1, S, S) additive mask]
                - position_embeddings: Dict[layer_type, (cos, sin)] with shape (B, S, head_dim)
        """
        batch_size, seq_len = input_ids.shape
        runtime_dtype = torch.float32
        if attention_mask.is_floating_point():
            runtime_dtype = attention_mask.dtype
            attention_mask = attention_mask.bool()

        # Build valid token mask
        valid_token_mask = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=getattr(self.text_config, "pad_token_id", 0),
            device=self.device,
        )

        # Get layer types
        layer_types = getattr(self.text_config, "layer_types", ["full_attention"])

        # Build per-layer-type attention masks
        attention_masks = _build_gemma4_prefill_masks(
            valid_token_mask,
            layer_types,
            self.sliding_window,
            self.device,
            runtime_dtype,
        )

        # Build position IDs and gather RoPE
        position_ids = _build_position_ids_from_valid_token_mask(valid_token_mask)
        position_embeddings = _gather_rope_by_position_ids(
            self.rope_tables,
            position_ids,
            layer_types,
        )

        return attention_masks, position_embeddings

    @torch.no_grad()
    def prefill(
        self,
        batch: dict[str, torch.Tensor],
        collect_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run static prefill and return last-token logits.

        If collect_hidden_states is True, also returns a list of per-layer
        hidden_states (last valid token) for divergence analysis.
        """
        llm_input_ids = batch["llm_input_ids"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        image_position_ids = batch.get("image_position_ids")
        if image_position_ids is not None:
            image_position_ids = image_position_ids.to(self.device)

        text_embeds = self.token_embedding(llm_input_ids)
        image_embeds = self.vision_prefill(pixel_values, image_position_ids)
        hidden_states = self.mm_fusion(text_embeds, image_embeds)
        self.layer_caches = self._allocate_empty_cache(
            hidden_states.shape[0],
            hidden_states.dtype,
        )

        attention_masks, position_embeddings = self.build_prefill_masks_and_rope(
            llm_input_ids,
            batch["attention_mask"].to(self.device),
        )

        # Compute PLE (Per-Layer Embeddings) if enabled
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            text_model = self._wrapped_text_model
            ple = text_model.get_per_layer_inputs(
                input_ids=llm_input_ids, inputs_embeds=hidden_states
            )
            per_layer_inputs = text_model.project_per_layer_inputs(
                inputs_embeds=hidden_states, per_layer_inputs=ple
            )

        # Track shared KV state for shared-KV layers
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        collected_hidden: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(self.prefill_layers):
            layer_type = self.text_config.layer_types[layer_idx]
            per_layer_input = (
                per_layer_inputs[:, :, layer_idx, :]
                if per_layer_inputs is not None
                else None
            )

            # Determine if this layer needs shared KV
            attn = layer.wrapped.self_attn.wrapped
            shared_key_value = None
            if getattr(attn, "is_kv_shared_layer", False):
                shared_key_value = shared_kv_states.get(layer_type)

            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
                per_layer_input=per_layer_input,
                shared_key_value=shared_key_value,
            )

            # Shared-KV layers return only hidden_states (no K/V);
            # non-shared layers return (hidden_states, key, value)
            if isinstance(out, tuple) and len(out) == 3:
                hidden_states, new_k, new_v = out
                valid_length = int(batch["valid_length"].item())
                # Store full-length KV for sharing with later shared-KV layers
                if getattr(attn, "store_full_length_kv", False):
                    shared_kv_states[layer_type] = (new_k[:, :, :valid_length, :], new_v[:, :, :valid_length, :])
                # Only cache valid (non-padding) tokens
                self.layer_caches[layer_idx].past_k[:, :, :valid_length, :] = new_k[:, :, :valid_length, :]
                self.layer_caches[layer_idx].past_v[:, :, :valid_length, :] = new_v[:, :, :valid_length, :]
            else:
                hidden_states = out

            if collect_hidden_states:
                vl = int(batch["valid_length"].item())
                collected_hidden.append(hidden_states[:, vl - 1, :].clone())

        self.past_len = int(batch["valid_length"].item())
        hidden_last = hidden_states[:, self.past_len - 1 : self.past_len, :]
        logits = self.lm_head(hidden_last)
        if collect_hidden_states:
            return logits[:, -1, :], collected_hidden
        return logits[:, -1, :]

    def build_decode_masks_and_rope(
        self,
        batch_size: int,
        dtype: torch.dtype,
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Build CPU-owned static masks and RoPE tensors for one decode step.

        Args:
            batch_size: Batch size.
            dtype: Target dtype.

        Returns:
            Tuple of:
                - attention_masks: Dict[layer_type, (B, 1, max_seq) additive mask]
                - position_embeddings: Dict[layer_type, (cos, sin)] with shape (B, 1, head_dim)
        """
        layer_types = getattr(self.text_config, "layer_types", ["full_attention"])

        # Build per-layer-type decode masks
        attention_masks = _build_gemma4_decode_masks(
            batch_size,
            self.past_len,
            self.layout.max_seq,
            layer_types,
            self.sliding_window,
            self.device,
            dtype,
        )

        # Gather RoPE at current position (past_len)
        position_ids = torch.full(
            (batch_size, 1),
            self.past_len,
            dtype=torch.long,
            device=self.device,
        )
        position_embeddings = _gather_rope_by_position_ids(
            self.rope_tables,
            position_ids,
            layer_types,
        )

        return attention_masks, position_embeddings

    @torch.no_grad()
    def decode_one(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run one static decode step and return next-token logits."""
        hidden_states = self.token_embedding(input_ids.to(self.device))
        attention_masks, position_embeddings = self.build_decode_masks_and_rope(
            batch_size=hidden_states.shape[0],
            dtype=hidden_states.dtype,
        )

        # Compute PLE (Per-Layer Embeddings) for the single decode token if enabled
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            text_model = self._wrapped_text_model
            ple = text_model.get_per_layer_inputs(
                input_ids=input_ids.to(self.device), inputs_embeds=hidden_states
            )
            per_layer_inputs = text_model.project_per_layer_inputs(
                inputs_embeds=hidden_states, per_layer_inputs=ple
            )

        # Track shared KV state for shared-KV layers
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for layer_idx, layer in enumerate(self.decode_layers):
            cache = self.layer_caches[layer_idx]
            layer_type = self.text_config.layer_types[layer_idx]
            per_layer_input = (
                per_layer_inputs[:, :, layer_idx, :]
                if per_layer_inputs is not None
                else None
            )

            # Determine if this layer needs shared KV
            attn = layer.wrapped.self_attn.wrapped
            shared_key_value = None
            if getattr(attn, "is_kv_shared_layer", False):
                shared_key_value = shared_kv_states.get(layer_type)

            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
                per_layer_input=per_layer_input,
                shared_key_value=shared_key_value,
                past_key_value=(
                    cache.past_k[:, :, : self.past_len, :],
                    cache.past_v[:, :, : self.past_len, :],
                ),
            )

            # Shared-KV layers return only hidden_states (no K/V);
            # non-shared layers return (hidden_states, key, value)
            if isinstance(out, tuple) and len(out) == 3:
                hidden_states, new_k, new_v = out
                # Store full-length KV for sharing with later shared-KV layers
                if getattr(attn, "store_full_length_kv", False):
                    shared_kv_states[layer_type] = (new_k, new_v)
                cache.past_k[:, :, self.past_len : self.past_len + 1, :] = new_k
                cache.past_v[:, :, self.past_len : self.past_len + 1, :] = new_v
            else:
                hidden_states = out

        self.past_len += 1
        logits = self.lm_head(hidden_states)
        return logits[:, -1, :]

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: str,
        image,
        max_new_tokens: int = 16,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Generate tokens using greedy sampling.

        Args:
            prompt: Text prompt string.
            image: PIL Image or numpy array.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: EOS token ID to stop generation.

        Returns:
            Tuple of:
                - generated_ids: (1, num_generated) tensor of generated token IDs
                - generated_text: List of generated token IDs
        """
        # Build static inputs
        batch = self.build_static_inputs(prompt, image)

        # Run prefill
        logits = self.prefill(batch)

        # Greedy decode
        generated_ids: List[int] = []
        all_logits = [logits]

        for _ in range(max_new_tokens):
            # Argmax
            next_token_id = int(logits.argmax(dim=-1).item())
            generated_ids.append(next_token_id)

            # Check EOS
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            # Decode one step
            input_ids = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=self.device
            )
            logits = self.decode_one(input_ids)
            all_logits.append(logits)

        generated_tensor = torch.tensor(
            [generated_ids], dtype=torch.long, device=self.device
        )
        return generated_tensor, generated_ids


def verify_against_reference(
    runtime: StaticGemma4Runtime,
    prompt: str,
    image,
    verify_steps: int = 4,
) -> None:
    """Verify runtime outputs against HF reference model.


    Args:
        runtime: StaticGemma4Runtime instance.
        prompt: Text prompt string.
        image: PIL Image or numpy array.
        verify_steps: Number of decode steps to verify.
    """
    from tico.quantization.evaluation.metric import compute_peir

    # Build inputs
    batch = runtime.build_static_inputs(prompt, image)
    llm_input_ids = batch["llm_input_ids"]
    valid_length = int(batch["valid_length"].item())

    # Get raw input_ids (with image placeholder tokens intact) for HF reference.
    # The runtime's llm_input_ids has image tokens replaced with pad_token_id,
    # but HF's Gemma4Model.forward() expects the original input_ids with image
    # placeholder tokens — it does the replacement internally via
    # get_placeholder_mask() + torch.where().
    raw_inputs = runtime.processor(
        text=prompt, images=image, return_tensors="pt", padding=False
    )
    raw_input_ids = raw_inputs["input_ids"]

    # Run reference model (full HF forward) — pass raw_input_ids
    with torch.no_grad():
        ref_outputs = runtime.model(
            input_ids=raw_input_ids,
            pixel_values=batch["pixel_values"],
            image_position_ids=batch.get("image_position_ids"),
            return_dict=True,
        )
        ref_logits = ref_outputs.logits[:, -1, :]  # Last token logits

    # Run runtime prefill
    runtime.reset_cache()
    runtime_logits = runtime.prefill(batch)

    # Compare prefill logits
    prefill_diff = (runtime_logits - ref_logits).abs()
    prefill_peir = compute_peir(runtime_logits, ref_logits)
    print(f"\n=== Prefill Verification ===")
    print(f"  Mean|diff|: {prefill_diff.mean().item():.6f}")
    print(f"  Max|diff|: {prefill_diff.max().item():.6f}")
    print(f"  PEIR: {prefill_peir:.6f}")

    # Verify decode steps
    print(f"\n=== Decode Verification ({verify_steps} steps) ===")
    for step in range(verify_steps):
        # Get next token from runtime
        next_token_id = int(runtime_logits.argmax(dim=-1).item())
        input_ids = torch.tensor(
            [[next_token_id]], dtype=torch.long, device=runtime.device
        )

        # Run runtime decode
        runtime_logits = runtime.decode_one(input_ids)

        # Run reference decode (append token to input)
        # Use raw_input_ids (with image tokens intact) — HF does the
        # replacement internally.
        extended_input_ids = torch.cat(
            [
                raw_input_ids,
                torch.tensor([[next_token_id]], device=runtime.device),
            ],
            dim=1,
        )

        with torch.no_grad():
            ref_outputs = runtime.model(
                input_ids=extended_input_ids,
                pixel_values=batch["pixel_values"],
                image_position_ids=batch.get("image_position_ids"),
                return_dict=True,
            )
            ref_logits = ref_outputs.logits[:, -1, :]

        # Compare
        decode_diff = (runtime_logits - ref_logits).abs()
        decode_peir = compute_peir(runtime_logits, ref_logits)
        print(
            f"  Step {step + 1}: Mean|diff|={decode_diff.mean().item():.6f}, "
            f"Max|diff|={decode_diff.max().item():.6f}, PEIR={decode_peir:.6f}"
        )


def verify_step_build_static_inputs(
    runtime: StaticGemma4Runtime,
    prompt: str,
    image,
    *,
    verbose: bool = True,
) -> dict:
    """Side-by-side validation of ``build_static_inputs()`` against HF reference.

    This function validates the first runtime workflow step in isolation:

    1. **llm_input_ids**: Compares the runtime's padded + image-placeholder-replaced
       input IDs against the HF reference computed via
       ``Gemma4Model.get_placeholder_mask()`` + ``torch.where(multimodal_mask, pad_token_id, input_ids)``.
    2. **valid_token_mask**: Compares the runtime's attention mask against the HF
       processor's attention mask (extended to max_seq with padding).
    3. **pixel_values**: Compares the pixel values from the processor (should be identical).
    4. **image_position_ids**: Compares image position IDs from the processor (should be identical).
    5. **valid_length**: Compares the valid token count.

    The runtime pads to ``max_seq`` and replaces image placeholder tokens with
    ``pad_token_id``. The HF reference does not pad, so we compare only the
    valid (non-padding) prefix.

    Args:
        runtime: ``StaticGemma4Runtime`` instance (wraps the HF model).
        prompt: Text prompt string.
        image: PIL Image or numpy array.
        verbose: If True, print per-step comparison details.

    Returns:
        Dict with keys:
            - ``"passed"``: bool, True if all sub-steps passed.
            - ``"batch"``: The runtime's ``build_static_inputs()`` output dict.
            - ``"ref_llm_input_ids"``: HF reference llm_input_ids (unpadded).
            - ``"results"``: Dict of per-sub-step pass/fail booleans.
    """
    # ------------------------------------------------------------------
    # Step 1a: Run the runtime's build_static_inputs
    # ------------------------------------------------------------------
    batch = runtime.build_static_inputs(prompt, image)
    rt_llm_input_ids = batch["llm_input_ids"]  # (1, max_seq)
    rt_attention_mask = batch["attention_mask"]  # (1, max_seq)
    rt_pixel_values = batch["pixel_values"]
    rt_image_position_ids = batch.get("image_position_ids")
    valid_length = int(batch["valid_length"].item())

    # ------------------------------------------------------------------
    # Step 1b: Compute HF reference
    # ------------------------------------------------------------------
    # Re-run the processor to get raw input_ids (before padding/replacement)
    inputs = runtime.processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=False,
    )
    raw_input_ids = inputs["input_ids"]  # (1, seq_len)

    # Access the HF Gemma4Model (runtime.model is Gemma4ForConditionalGeneration)
    hf_model = runtime.model  # Gemma4ForConditionalGeneration
    hf_gemma4_model = hf_model.model  # Gemma4Model

    # HF get_placeholder_mask: returns (image_mask, video_mask, audio_mask)
    image_mask, video_mask, audio_mask = hf_gemma4_model.get_placeholder_mask(
        input_ids=raw_input_ids
    )
    multimodal_mask = image_mask | video_mask | audio_mask

    # HF llm_input_ids: replace multimodal placeholder tokens with pad_token_id
    pad_token_id = getattr(runtime.text_config, "pad_token_id", 0)
    ref_llm_input_ids = torch.where(
        multimodal_mask, torch.tensor(pad_token_id, dtype=raw_input_ids.dtype), raw_input_ids
    )

    # HF attention mask from processor (if available)
    raw_attention_mask = inputs.get("attention_mask", None)

    # ------------------------------------------------------------------
    # Step 1c: Compare
    # ------------------------------------------------------------------
    results: dict[str, bool] = {}

    # 1. llm_input_ids: compare valid prefix only (runtime is padded to max_seq)
    rt_valid_ids = rt_llm_input_ids[0, :valid_length]
    ref_valid_ids = ref_llm_input_ids[0]
    # Both should be long tensors with identical values
    ids_match = torch.equal(rt_valid_ids, ref_valid_ids)
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"  Step 1.1: llm_input_ids (valid prefix, len={valid_length})")
        print(f"  Runtime shape:  {tuple(rt_valid_ids.shape)}")
        print(f"  HF shape:       {tuple(ref_valid_ids.shape)}")
        if ids_match:
            print(f"  Exact match: True")
            print(f"  [PASS]")
        else:
            diff_count = (rt_valid_ids != ref_valid_ids).sum().item()
            print(f"  Exact match: False ({diff_count} mismatches)")
            # Show first few mismatches
            mismatch_idx = (rt_valid_ids != ref_valid_ids).nonzero(as_tuple=True)[0]
            for idx in mismatch_idx[:5]:
                i = idx.item()
                print(f"    pos {i}: runtime={rt_valid_ids[i].item()} hf={ref_valid_ids[i].item()}")
            print(f"  [FAIL]")
    results["llm_input_ids"] = ids_match

    # 2. valid_token_mask: compare runtime's attention mask against HF's
    # Runtime pads to max_seq; HF's raw attention mask is unpadded.
    # Compare the valid prefix.
    if raw_attention_mask is not None:
        rt_valid_mask = rt_attention_mask[0, :valid_length]
        ref_valid_mask = raw_attention_mask[0].bool()
        mask_match = torch.equal(rt_valid_mask, ref_valid_mask)
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  Step 1.2: valid_token_mask (valid prefix, len={valid_length})")
            print(f"  Runtime shape:  {tuple(rt_valid_mask.shape)}")
            print(f"  HF shape:       {tuple(ref_valid_mask.shape)}")
            if mask_match:
                print(f"  Exact match: True")
                print(f"  [PASS]")
            else:
                print(f"  Exact match: False")
                print(f"  [FAIL]")
        results["valid_token_mask"] = mask_match
    else:
        if verbose:
            print(f"\n  Step 1.2: valid_token_mask — SKIPPED (no HF attention_mask)")
        results["valid_token_mask"] = True

    # 3. pixel_values: should be identical (both from same processor call)
    rt_pv = rt_pixel_values
    ref_pv = inputs.get("pixel_values", None)
    if ref_pv is not None:
        pv_match = torch.equal(rt_pv, ref_pv)
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  Step 1.3: pixel_values")
            print(f"  Runtime shape:  {tuple(rt_pv.shape)}")
            print(f"  HF shape:       {tuple(ref_pv.shape)}")
            if pv_match:
                print(f"  Exact match: True")
                print(f"  [PASS]")
            else:
                diff = (rt_pv - ref_pv).abs()
                print(f"  Exact match: False")
                print(f"  Max|diff|: {diff.max().item():.8f}")
                print(f"  [FAIL]")
        results["pixel_values"] = pv_match
    else:
        if verbose:
            print(f"\n  Step 1.3: pixel_values — SKIPPED (no HF pixel_values)")
        results["pixel_values"] = True

    # 4. image_position_ids: should be identical (both from same processor call)
    ref_ipids = inputs.get("image_position_ids", None)
    if rt_image_position_ids is not None and ref_ipids is not None:
        ipids_match = torch.equal(rt_image_position_ids, ref_ipids)
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  Step 1.4: image_position_ids")
            print(f"  Runtime shape:  {tuple(rt_image_position_ids.shape)}")
            print(f"  HF shape:       {tuple(ref_ipids.shape)}")
            if ipids_match:
                print(f"  Exact match: True")
                print(f"  [PASS]")
            else:
                diff = (rt_image_position_ids - ref_ipids).abs()
                print(f"  Exact match: False")
                print(f"  Max|diff|: {diff.max().item():.8f}")
                print(f"  [FAIL]")
        results["image_position_ids"] = ipids_match
    else:
        if verbose:
            print(f"\n  Step 1.4: image_position_ids — SKIPPED (not available)")
        results["image_position_ids"] = True

    # 5. valid_length: sanity check
    if raw_attention_mask is not None:
        ref_valid_length = int(raw_attention_mask[0].sum().item())
    else:
        ref_valid_length = raw_input_ids.shape[1]
    vl_match = valid_length == ref_valid_length
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"  Step 1.5: valid_length")
        print(f"  Runtime:  {valid_length}")
        print(f"  HF:       {ref_valid_length}")
        if vl_match:
            print(f"  [PASS]")
        else:
            print(f"  [FAIL]")
    results["valid_length"] = vl_match

    # 6. Padding check: verify runtime padded the rest with pad_token_id
    if valid_length < rt_llm_input_ids.shape[1]:
        padding_part = rt_llm_input_ids[0, valid_length:]
        all_pad = torch.all(padding_part == pad_token_id).item()
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  Step 1.6: padding (positions {valid_length}:{rt_llm_input_ids.shape[1]})")
            print(f"  All pad_token_id ({pad_token_id}): {all_pad}")
            if all_pad:
                print(f"  [PASS]")
            else:
                print(f"  [FAIL]")
        results["padding"] = all_pad
    else:
        results["padding"] = True

    # Summary
    all_passed = all(results.values())
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"  Step 1 Summary: build_static_inputs")
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"    {name:25s} [{status}]")
        overall = "PASS" if all_passed else "FAIL"
        print(f"  Overall: [{overall}]")

    return {
        "passed": all_passed,
        "batch": batch,
        "ref_llm_input_ids": ref_llm_input_ids,
        "results": results,
    }


def verify_per_layer_divergence(
    runtime: StaticGemma4Runtime,
    prompt: str,
    image,
) -> None:
    """Compare per-layer hidden_states between static runtime and HF reference.

    Registers forward hooks on the HF model's decoder layers to capture
    reference hidden_states, then compares against the static runtime's
    per-layer hidden_states (last valid token only).
    """
    batch = runtime.build_static_inputs(prompt, image)
    llm_input_ids = batch["llm_input_ids"]
    valid_length = int(batch["valid_length"].item())

    # Get raw input_ids (with image placeholder tokens intact) for HF reference.
    # The runtime's llm_input_ids has image tokens replaced with pad_token_id,
    # but HF's Gemma4Model.forward() expects the original input_ids with image
    # placeholder tokens — it does the replacement internally via
    # get_placeholder_mask() + torch.where().
    raw_inputs = runtime.processor(
        text=prompt, images=image, return_tensors="pt", padding=False
    )
    raw_input_ids = raw_inputs["input_ids"]

    # --- Capture reference per-layer hidden_states via hooks ---
    ref_hidden_states: list[torch.Tensor] = []

    # Access the HF model's text decoder layers
    text_model = runtime.model.model.language_model
    num_layers = len(text_model.layers)

    hooks = []
    for i, layer in enumerate(text_model.layers):
        def make_hook(idx):
            def hook_fn(module, input, output):
                # output is typically a tuple; hidden_states is output[0]
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                # Capture last valid token
                ref_hidden_states.append(hs[:, valid_length - 1, :].detach().clone())
            return hook_fn
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Run reference model — pass raw_input_ids (with image tokens intact)
    with torch.no_grad():
        _ = runtime.model(
            input_ids=raw_input_ids,
            pixel_values=batch["pixel_values"],
            image_position_ids=batch.get("image_position_ids"),
            return_dict=True,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # --- Run static runtime with hidden_states collection ---
    runtime.reset_cache()
    runtime_logits, runtime_hidden = runtime.prefill(
        batch, collect_hidden_states=True
    )

    # --- Compare per-layer ---
    print(f"\n=== Per-Layer Hidden States Divergence (last valid token) ===")
    print(f"  {'Layer':>5}  {'Type':>20}  {'Mean|diff|':>12}  {'Max|diff|':>12}  {'Ref|mean|':>12}  {'Rt|mean|':>12}")
    for i in range(min(num_layers, len(runtime_hidden), len(ref_hidden_states))):
        layer_type = runtime.text_config.layer_types[i]
        ref_hs = ref_hidden_states[i].float()
        rt_hs = runtime_hidden[i].float()
        diff = (rt_hs - ref_hs).abs()
        print(
            f"  {i:5d}  {layer_type:>20}  {diff.mean().item():12.6f}  "
            f"{diff.max().item():12.6f}  {ref_hs.abs().mean().item():12.6f}  "
            f"{rt_hs.abs().mean().item():12.6f}"
        )


def run_static_gemma4_runtime(cfg: StaticGemma4RuntimeConfig) -> None:
    """Run the Gemma4 E2B static runtime smoke flow.

    Args:
        cfg: StaticGemma4RuntimeConfig with model, prompt, and generation settings.
    """
    print("=== Loading model and processor ===")
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model,
        torch_dtype=torch.float32,
        device_map=cfg.device,
    )
    processor = AutoProcessor.from_pretrained(cfg.model)

    # Create layout
    layout = StaticGemma4Layout(
        max_seq=cfg.max_seq,
        visual_start_idx=cfg.visual_start_idx,
        num_visual_tokens=cfg.num_visual_tokens,
        batch_size=1,
    )

    # Create runtime
    runtime = StaticGemma4Runtime(
        model=model,
        processor=processor,
        layout=layout,
        device=cfg.device,
    )

    # Get EOS token ID from model config (Gemma4Processor doesn't expose eos_token_id)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None:
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            eos_token_id = getattr(text_config, "eos_token_id", None)

    # Create dummy image for testing (896x896 RGB)
    import numpy as np

    dummy_image = np.random.randint(
        0, 255, (cfg.image_height, cfg.image_width, 3), dtype=np.uint8
    )

    # Verify build_static_inputs
    result = verify_step_build_static_inputs(
        runtime, cfg.prompt, dummy_image, verbose=True
    )
    assert result["passed"], f"Step validation failed: {result['results']}"
    assert "batch" in result
    assert "ref_llm_input_ids" in result
    assert "results" in result

    # Per-layer divergence analysis
    print(f"\n=== Per-Layer Divergence Analysis ===")
    verify_per_layer_divergence(runtime, cfg.prompt, dummy_image)

    # Verify against reference
    if cfg.verify_steps > 0:
        print(f"\n=== Verification ({cfg.verify_steps} steps) ===")
        verify_against_reference(runtime, cfg.prompt, dummy_image, cfg.verify_steps)

    # Generate
    print(f"\n=== Generation ({cfg.gen_steps} steps) ===")
    runtime.reset_cache()
    generated_ids, _ = runtime.generate_greedy(
        prompt=cfg.prompt,
        image=dummy_image,
        max_new_tokens=cfg.gen_steps,
        eos_token_id=eos_token_id,
    )
    print(f"Generated tokens: {generated_ids.tolist()}")
    print("Done.")
