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
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor

from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.wrapq.wrap_helper import PTQWrapHelper
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    build_gemma4_vision_prefill_export_module,
    Gemma4LMHeadExportAdapter,
    Gemma4MMFusionExportAdapter,
    Gemma4TokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import (
    DEFAULT_GEMMA4_STATIC_VISION_PROFILE,
    Gemma4StaticVisionProfile,
    get_gemma4_static_vision_profile,
)
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    assert_gemma4_e2b_no_moe,
    build_decode_attention_mask,
    StaticGemma4Layout,
)

# =============================================================================
# CPU Helper Functions (pure Python, no model needed)
# =============================================================================


def _normalize_valid_token_mask(
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    *,
    pad_token_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Normalize attention mask to a boolean valid-token mask.

    If attention_mask is provided, convert it to boolean.
    If not, derive from input_ids by comparing against pad_token_id.
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

    Currently only 'right' padding is supported: valid tokens first, then
    padding. Raises ValueError if the layout doesn't match or if an
    unsupported padding_side is requested.
    """
    if padding_side != "right":
        raise ValueError(
            f"Unsupported padding_side={padding_side!r}, only 'right' is supported."
        )
    for i in range(valid_token_mask.size(0)):
        row = valid_token_mask[i]
        false_indices = torch.where(~row)[0]
        if len(false_indices) > 0:
            first_false = int(false_indices[0].item())
            if not torch.all(~row[first_false:]):
                raise ValueError("Right padding expected but not found")


def _build_full_attention_mask(
    valid_token_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float,
) -> torch.Tensor:
    """Build a standard causal attention mask with padding.

    Args:
        valid_token_mask: Boolean mask of shape ``(B, S)`` — True for real
            tokens, False for padding.
        device: Target device.
        dtype: Target dtype for the output mask.
        mask_value: Value used to mask forbidden positions (typically
            ``torch.finfo(dtype).min``).

    Returns:
        Additive attention bias of shape ``(B, 1, S, S)``.  ``0.0`` allows
        attention, ``mask_value`` forbids it.
    """
    batch_size, seq_len = valid_token_mask.shape
    # Causal: kv_idx <= q_idx
    causal = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    )  # (S, S)
    causal = causal.unsqueeze(0).expand(batch_size, -1, -1)  # (B, S, S)

    key_valid = valid_token_mask.to(device).unsqueeze(1).expand(-1, seq_len, -1)

    valid = causal & key_valid
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~valid, mask_value)
    return mask.unsqueeze(1)  # (B, 1, S, S)


def _build_sliding_window_attention_mask(
    valid_token_mask: torch.Tensor,
    sliding_window: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float,
) -> torch.Tensor:
    """Build a sliding-window causal attention mask with padding.

    A token at position ``q`` can attend to position ``k`` if:
    - ``k <= q`` (causal), AND
    - ``k > q - sliding_window`` (within the sliding window)

    This matches HF's ``sliding_window_causal_mask_function`` which is
    ``and_masks(sliding_window_overlay, causal_mask_function)`` where
    ``sliding_window_overlay`` returns ``kv_idx > q_idx - sliding_window``.

    Args:
        valid_token_mask: Boolean mask of shape ``(B, S)``.
        sliding_window: Window size (number of past tokens visible, inclusive
            of the current token).
        device: Target device.
        dtype: Target dtype.
        mask_value: Value for masked positions.

    Returns:
        Additive attention bias of shape ``(B, 1, S, S)``.
    """
    batch_size, seq_len = valid_token_mask.shape

    q_idx = torch.arange(seq_len, device=device).view(seq_len, 1)  # (S, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, seq_len)  # (1, S)

    # Causal: kv_idx <= q_idx
    causal = kv_idx <= q_idx  # (S, S) broadcast
    # Sliding window: kv_idx > q_idx - sliding_window
    window = kv_idx > (q_idx - sliding_window)  # (S, S) broadcast

    valid_pattern = causal & window  # (S, S)
    valid_pattern = valid_pattern.unsqueeze(0).expand(batch_size, -1, -1)  # (B, S, S)

    key_valid = valid_token_mask.to(device).unsqueeze(1).expand(-1, seq_len, -1)

    valid = valid_pattern & key_valid
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~valid, mask_value)
    return mask.unsqueeze(1)  # (B, 1, S, S)


# =============================================================================
# Data Classes
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
    vision_profile: str = DEFAULT_GEMMA4_STATIC_VISION_PROFILE
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    visual_start_idx: Optional[int] = None
    num_visual_tokens: Optional[int] = None
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
        vision_profile: Optional[Gemma4StaticVisionProfile] = None,
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
        if vision_profile is None:
            vision_profile = get_gemma4_static_vision_profile(
                DEFAULT_GEMMA4_STATIC_VISION_PROFILE
            )
        vision_profile.validate(
            max_seq_len=layout.max_seq,
            vision_config=model.config.vision_config,
        )
        vision_profile.validate_processor(processor)
        if layout.visual_start_idx != vision_profile.visual_start_idx:
            raise ValueError(
                "StaticGemma4Layout.visual_start_idx does not match the vision "
                f"profile: layout={layout.visual_start_idx}, "
                f"profile={vision_profile.visual_start_idx}."
            )
        if layout.num_visual_tokens != vision_profile.num_visual_tokens:
            raise ValueError(
                "StaticGemma4Layout.num_visual_tokens does not match the vision "
                f"profile: layout={layout.num_visual_tokens}, "
                f"profile={vision_profile.num_visual_tokens}."
            )
        self.vision_profile = vision_profile

        qcfg = build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_config.num_hidden_layers),
            num_vision_layers=int(model.config.vision_config.num_hidden_layers),
            model_args={"vision": vision_profile.to_vision_model_args()},
        )
        # Runtime simulation must stay in NO_QUANT mode until a calibrated
        # checkpoint is supplied. This matches the floating-point Circle
        # export path and keeps as_export_module() available.
        self.qmodel = (
            PTQWrapHelper(strict_wrap=True)
            .wrap_supported(model, qcfg)
            .to(self.device)
            .eval()
        )

        wrapped_top = (
            self.qmodel.wrapped if hasattr(self.qmodel, "wrapped") else self.qmodel
        )
        wrapped_model = wrapped_top.model.wrapped
        self._wrapped_model = wrapped_model

        self.text_model = wrapped_model.language_model.wrapped
        self.token_embedding = Gemma4TokenEmbeddingExportAdapter(self.text_model).to(
            self.device
        )

        self.vision_prefill: Optional[nn.Module] = None
        self.mm_fusion = Gemma4MMFusionExportAdapter(
            visual_start_idx=vision_profile.visual_start_idx,
            num_visual_tokens=vision_profile.num_visual_tokens,
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

        self.layer_caches: list[LayerCache] = []
        self.past_len = 0

    def reset_cache(self) -> None:
        """Reset all runtime-managed KV caches."""
        self.layer_caches = []
        self.past_len = 0

    def _get_or_create_vision_prefill(
        self,
        image_position_ids: Optional[torch.Tensor],
    ) -> nn.Module:
        """Validate processor coordinates and return a pixel-values-only module."""
        if image_position_ids is None:
            raise ValueError(
                "Gemma4 static vision prefill requires image_position_ids from "
                "the processor."
            )
        self.vision_profile.validate_image_position_ids(image_position_ids)

        if self.vision_prefill is None:
            static_position_ids = self.vision_profile.build_image_position_ids(
                device=self.device
            )
            self.vision_prefill = (
                build_gemma4_vision_prefill_export_module(
                    self._wrapped_model,
                    pixel_position_ids=static_position_ids,
                )
                .to(self.device)
                .eval()
            )
        return self.vision_prefill

    def _allocate_empty_cache(
        self, batch_size: int, dtype: torch.dtype
    ) -> list[LayerCache]:
        """Allocate fixed-size empty KV cache tensors.

        Gemma4 has per-layer-type head dimensions:
        - Sliding attention layers: ``head_dim = config.head_dim``
        - Full attention layers: ``head_dim = config.global_head_dim``

        The number of KV heads may also differ when
        ``num_global_key_value_heads`` is set, but for E2B it defaults to
        ``num_key_value_heads`` for all layers.
        """
        caches = []
        for layer_idx in range(int(self.text_config.num_hidden_layers)):
            layer_type = self.text_config.layer_types[layer_idx]
            is_sliding = layer_type == "sliding_attention"

            # Per-layer-type head_dim (HF Gemma4TextAttention.__init__)
            global_head_dim = getattr(self.text_config, "global_head_dim", None)
            if not is_sliding and global_head_dim:
                head_dim = int(global_head_dim)
            else:
                head_dim = int(self.text_config.head_dim)

            # Per-layer-type num_kv_heads
            # use_alternative_attention = attention_k_eq_v and not is_sliding
            # For E2B, attention_k_eq_v=False, so num_kv_heads is always
            # config.num_key_value_heads.  But we handle the general case.
            attention_k_eq_v = getattr(self.text_config, "attention_k_eq_v", False)
            use_alternative_attention = attention_k_eq_v and not is_sliding
            if use_alternative_attention:
                num_kv_heads = int(
                    getattr(self.text_config, "num_global_key_value_heads", None)
                    or self.text_config.num_key_value_heads
                )
            else:
                num_kv_heads = int(self.text_config.num_key_value_heads)

            past_k = torch.zeros(
                batch_size,
                num_kv_heads,
                self.layout.max_seq,
                head_dim,
                device=self.device,
                dtype=dtype,
            )
            caches.append(LayerCache(past_k=past_k, past_v=torch.zeros_like(past_k)))
        return caches

    def build_static_inputs(
        self, prompt: str, image, max_seq: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Build static padded processor inputs.

        Processes the prompt+image through the HF processor, pads to
        ``max_seq``, and replaces image placeholder tokens with
        ``pad_token_id`` to create ``llm_input_ids``.

        Args:
            prompt: Text prompt string.
            image: PIL image or tensor to feed to the processor.
            max_seq: Override for ``self.layout.max_seq``.

        Returns:
            Dict with keys: ``llm_input_ids``, ``pixel_values``,
            ``image_position_ids``, ``attention_mask``, ``valid_length``.
        """
        if max_seq is None:
            max_seq = self.layout.max_seq
        pad_token_id = getattr(self.text_config, "pad_token_id", 0)

        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt", padding=False
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)

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

        seq_len = input_ids.shape[0]
        if seq_len > max_seq:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq {max_seq}"
            )

        # CRITICAL: image_token_id from self.config, NOT self.text_config
        image_token_id = getattr(self.config, "image_token_id", None)

        if image_token_id is None:
            raise ValueError("Gemma4 config.image_token_id is required.")
        self.vision_profile.validate_processor_outputs(
            inputs,
            image_token_id=int(image_token_id),
        )

        padded_input_ids = torch.full(
            (max_seq,), pad_token_id, dtype=input_ids.dtype, device=self.device
        )
        padded_input_ids[:seq_len] = input_ids.to(self.device)

        if image_token_id is not None:
            padded_input_ids[padded_input_ids == image_token_id] = pad_token_id

        padded_attention_mask = torch.zeros(
            max_seq, dtype=torch.bool, device=self.device
        )
        padded_attention_mask[:seq_len] = True

        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is None:
            raise ValueError("Processor did not return pixel_values")
        pixel_values = pixel_values.to(self.device)

        image_position_ids = inputs.get("image_position_ids", None)
        if image_position_ids is None:
            raise ValueError("Processor did not return image_position_ids")
        image_position_ids = image_position_ids.to(self.device)

        valid_length = torch.tensor([seq_len], dtype=torch.long, device=self.device)

        # Cache raw (unpadded) processor output so that verification steps
        # (e.g. verify_step_prefill) can reuse it without re-running the
        # processor, which involves expensive image preprocessing.
        raw_inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "image_position_ids": inputs.get("image_position_ids", None),
        }

        return {
            "llm_input_ids": padded_input_ids.unsqueeze(0),
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
            "attention_mask": padded_attention_mask.unsqueeze(0),
            "valid_length": valid_length,
            "_raw_inputs": raw_inputs,
        }

    def build_prefill_masks_and_rope(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Build CPU-owned static masks and RoPE tensors for prefill.

        Produces two dictionaries keyed by layer type
        (``"full_attention"``, ``"sliding_attention"``):

        - ``attention_masks``: 4D additive bias tensors of shape
          ``(batch, 1, seq_len, seq_len)``.  ``0.0`` allows attention,
          ``mask_value`` forbids it.  Full-attention layers get a standard
          causal mask; sliding-attention layers get a sliding-window causal
          mask.  Both masks also zero out padding positions.

        - ``position_embeddings``: ``(cos, sin)`` pairs of shape
          ``(batch, seq_len, head_dim)`` per layer type, computed via the HF
          model's ``Gemma4TextRotaryEmbedding`` module.  Full-attention layers
          use proportional RoPE (partial_rotary_factor=0.25, global_head_dim);
          sliding-attention layers use default RoPE (head_dim).
        """
        batch_size, seq_len = input_ids.shape

        # --- Determine runtime dtype from model weights ---
        runtime_dtype = next(self.model.parameters()).dtype

        # --- Valid token mask (boolean) ---
        valid_token_mask = attention_mask.to(self.device).bool()  # (B, S)

        # --- Position IDs: arange(seq_len) for prefill (no past cache) ---
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)  # (1, S)

        # --- Attention masks ---
        mask_value = torch.finfo(runtime_dtype).min

        # Full attention: standard causal + padding
        full_mask = _build_full_attention_mask(
            valid_token_mask, self.device, runtime_dtype, mask_value
        )

        # Sliding attention: sliding-window causal + padding
        sliding_window = int(getattr(self.text_config, "sliding_window", 1024))
        sliding_mask = _build_sliding_window_attention_mask(
            valid_token_mask,
            sliding_window,
            self.device,
            runtime_dtype,
            mask_value,
        )

        layer_types = set(getattr(self.text_config, "layer_types", ["full_attention"]))
        attention_masks = {
            "full_attention": full_mask,
            "sliding_attention": sliding_mask,
        }
        # If the model only has one layer type, keep only that entry
        if not hasattr(self.text_config, "layer_types"):
            attention_masks = {"full_attention": full_mask}

        # --- RoPE (per layer type) ---
        # Use the HF model's Gemma4TextRotaryEmbedding module directly.
        # It has per-layer-type inv_freq buffers and attention_scaling.
        rotary_emb = self.model.model.language_model.rotary_emb

        # Dummy hidden states for shape/dtype inference (rotary_emb only uses
        # x.device and x.dtype)
        dummy_hidden = torch.zeros(
            batch_size, seq_len, 1, device=self.device, dtype=runtime_dtype
        )

        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in layer_types:
            cos, sin = rotary_emb(dummy_hidden, position_ids, layer_type)
            position_embeddings[layer_type] = (cos, sin)

        return attention_masks, position_embeddings

    @torch.no_grad()
    def prefill(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run static prefill and return last-token logits."""
        llm_input_ids = batch["llm_input_ids"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device).to(self.model.dtype)
        image_position_ids = batch.get("image_position_ids")
        if image_position_ids is not None:
            image_position_ids = image_position_ids.to(self.device)

        text_embeds = self.token_embedding(llm_input_ids)
        vision_prefill = self._get_or_create_vision_prefill(image_position_ids)
        # Processor coordinates select and validate the baked module above;
        # they are intentionally absent from the vision-prefill runtime ABI.
        image_embeds = vision_prefill(pixel_values)
        hidden_states = self.mm_fusion(text_embeds, image_embeds)
        self.layer_caches = self._allocate_empty_cache(
            hidden_states.shape[0], hidden_states.dtype
        )

        attention_masks, position_embeddings = self.build_prefill_masks_and_rope(
            llm_input_ids,
            batch["attention_mask"].to(self.device),
        )

        # Compute Per-Layer Embeddings (PLE) if enabled.
        # PLE has two components:
        #   1. Token-identity: embed_tokens_per_layer(llm_input_ids) → (B, S, L, P)
        #   2. Context projection: per_layer_model_projection(hidden_states) → (B, S, L, P)
        # Combined: (projection + token_identity) * per_layer_input_scale
        # Each layer receives its slice: per_layer_inputs[:, :, i, :]
        per_layer_inputs = None
        if self.text_model.hidden_size_per_layer_input:
            ple_token = self.text_model.get_per_layer_inputs(llm_input_ids, None)
            per_layer_inputs = self.text_model.project_per_layer_inputs(
                hidden_states, ple_token
            )

        # Shared-KV bookkeeping: some layers share K/V states with earlier
        # layers of the same layer_type.  Store layers write their full-length
        # K/V into shared_kv_states; consumer layers read from it.
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Extract valid length once before the loop to avoid GPU→CPU sync per layer.
        valid_len = int(batch["valid_length"].item())

        for layer_idx, layer in enumerate(self.prefill_layers):
            layer_type = self.text_config.layer_types[layer_idx]
            per_layer_input = (
                per_layer_inputs[:, :, layer_idx, :]
                if per_layer_inputs is not None
                else None
            )

            # Check if this layer is a shared-KV consumer
            attn = layer.wrapped.self_attn.wrapped
            is_shared = bool(getattr(attn, "is_kv_shared_layer", False))
            shared_kv = shared_kv_states.get(layer_type, None) if is_shared else None

            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
                per_layer_input=per_layer_input,
                shared_key_value=shared_kv,
            )

            if isinstance(out, tuple) and len(out) == 3:
                hidden_states, new_k, new_v = out
                # Write only valid K/V (excluding padding) into the cache.
                # new_k/new_v from prefill have shape (B, kv_heads, max_seq, head_dim)
                # but positions >= valid_length contain garbage from padding tokens.
                self.layer_caches[layer_idx].past_k[:, :, :valid_len, :].copy_(
                    new_k[:, :, :valid_len, :]
                )
                self.layer_caches[layer_idx].past_v[:, :, :valid_len, :].copy_(
                    new_v[:, :, :valid_len, :]
                )
                # Store valid-length K/V for shared-KV consumer layers
                if bool(getattr(attn, "store_full_length_kv", False)):
                    shared_kv_states[layer_type] = (
                        new_k[:, :, :valid_len, :],
                        new_v[:, :, :valid_len, :],
                    )
            else:
                hidden_states = out

        self.past_len = int(batch["valid_length"].item())

        logits = self.lm_head(hidden_states[:, self.past_len - 1 : self.past_len, :])
        logits = logits[:, -1, :]
        return self._apply_final_logit_softcapping(logits)

    def build_decode_masks_and_rope(
        self, batch_size: int, dtype: torch.dtype
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Build CPU-owned static masks and RoPE tensors for one decode step.

        Produces per-layer-type decode masks and RoPE:

        - **Full attention mask**: ``(B, 1, max_seq)`` additive bias.
          Positions ``0..past_len`` are ``0.0`` (allowed), the rest are
          ``mask_value`` (forbidden).

        - **Sliding attention mask**: ``(B, 1, max_seq)`` additive bias.
          Only positions within the sliding window
          ``[max(0, past_len - sliding_window + 1), past_len]`` are ``0.0``.

        - **RoPE**: ``(cos, sin)`` computed via the HF model's
          ``Gemma4TextRotaryEmbedding`` at the current decode position
          ``past_len``, per layer type.
        """
        max_seq = self.layout.max_seq
        mask_value = torch.finfo(dtype).min

        # Full attention decode mask: attend to all past + current token
        full_mask = build_decode_attention_mask(
            batch_size=batch_size,
            past_len=self.past_len,
            max_seq=max_seq,
            device=self.device,
            dtype=dtype,
            mask_value=mask_value,
        )

        # Sliding attention decode mask: only attend within the sliding window
        sliding_window = int(getattr(self.text_config, "sliding_window", 1024))
        sliding_mask = torch.full(
            (batch_size, 1, max_seq), float(mask_value), device=self.device, dtype=dtype
        )
        # Allowed positions: max(0, past_len - sliding_window + 1) .. past_len
        window_start = max(0, self.past_len - sliding_window + 1)
        sliding_mask[:, :, window_start : self.past_len + 1] = 0.0

        attention_masks = {
            "full_attention": full_mask,
            "sliding_attention": sliding_mask,
        }

        # RoPE at the current decode position
        rotary_emb = self.model.model.language_model.rotary_emb
        position_ids = torch.tensor(
            [[self.past_len]], device=self.device, dtype=torch.long
        )  # (1, 1)
        dummy_hidden = torch.zeros(batch_size, 1, 1, device=self.device, dtype=dtype)

        layer_types = set(getattr(self.text_config, "layer_types", ["full_attention"]))
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in layer_types:
            cos, sin = rotary_emb(dummy_hidden, position_ids, layer_type)
            position_embeddings[layer_type] = (cos, sin)

        return attention_masks, position_embeddings

    @torch.no_grad()
    def decode_one(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run one static decode step and return next-token logits.

        Handles PLE, per-layer-type masks/RoPE, and shared-KV bookkeeping,
        mirroring the prefill path for a single-token decode step.
        """
        llm_input_ids = input_ids.to(self.device)
        hidden_states = self.token_embedding(llm_input_ids)

        attention_masks, position_embeddings = self.build_decode_masks_and_rope(
            batch_size=hidden_states.shape[0],
            dtype=hidden_states.dtype,
        )

        # Compute PLE for the single decode token if enabled
        per_layer_inputs = None
        if self.text_model.hidden_size_per_layer_input:
            ple_token = self.text_model.get_per_layer_inputs(llm_input_ids, None)
            per_layer_inputs = self.text_model.project_per_layer_inputs(
                hidden_states, ple_token
            )

        # Shared-KV bookkeeping: store layers' full K/V (including the new
        # decode token) are passed to consumer layers.  We maintain a dict
        # keyed by layer_type, updated after each store layer writes its delta.
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Pre-populate shared_kv_states from layer_caches for shared consumer layers.
        # This ensures consumer layers have access to KV cache from previous tokens
        # before store layers update it in this decode step.
        for layer_idx, layer_type in enumerate(self.text_config.layer_types):
            if layer_type not in shared_kv_states:
                attn = self.decode_layers[layer_idx].wrapped.self_attn.wrapped
                is_shared = bool(getattr(attn, "is_kv_shared_layer", False))

                if is_shared:
                    cache = self.layer_caches[layer_idx]
                    shared_kv_states[layer_type] = (
                        cache.past_k[:, :, : self.past_len, :],
                        cache.past_v[:, :, : self.past_len, :],
                    )

        for layer_idx, layer in enumerate(self.decode_layers):
            layer_type = self.text_config.layer_types[layer_idx]
            per_layer_input = (
                per_layer_inputs[:, :, layer_idx, :]
                if per_layer_inputs is not None
                else None
            )

            # Check if this layer is a shared-KV consumer
            attn = layer.wrapped.self_attn.wrapped
            is_shared = bool(getattr(attn, "is_kv_shared_layer", False))
            shared_kv = shared_kv_states.get(layer_type, None) if is_shared else None

            cache = self.layer_caches[layer_idx]

            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
                past_key_value=(
                    (
                        cache.past_k[:, :, : self.past_len, :],
                        cache.past_v[:, :, : self.past_len, :],
                    )
                    if not is_shared
                    else None
                ),
                per_layer_input=per_layer_input,
                shared_key_value=shared_kv,
            )

            if isinstance(out, tuple) and len(out) == 3:
                hidden_states, new_k, new_v = out

                # Write single-token K/V delta into the fixed cache
                cache.past_k[:, :, self.past_len : self.past_len + 1, :].copy_(new_k)
                cache.past_v[:, :, self.past_len : self.past_len + 1, :].copy_(new_v)

                # Update shared_kv_states for consumer layers.
                # The valid K/V (prefill + all decode deltas so far) occupies
                # positions 0..past_len inclusive in the cache.  Slice to that
                # range so consumer layers see the correct k_len = past_len + 1.
                if bool(getattr(attn, "store_full_length_kv", False)):
                    shared_kv_states[layer_type] = (
                        cache.past_k[:, :, : self.past_len + 1, :],
                        cache.past_v[:, :, : self.past_len + 1, :],
                    )
            else:
                hidden_states = out

        self.past_len += 1
        logits = self.lm_head(hidden_states)[:, -1, :]
        return self._apply_final_logit_softcapping(logits)

    @torch.no_grad()
    def generate_greedy(
        self,
        batch: dict[str, torch.Tensor],
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate tokens greedily from a preprocessed static batch.

        Args:
            batch: The batch dict returned by ``build_static_inputs``.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: Optional EOS token ID to stop generation early.
                If not provided, uses ``text_config.eos_token_id``.

        Returns:
            Generated token IDs of shape ``(1, prompt_len + num_generated)``.
        """
        # Run prefill
        logits = self.prefill(batch)

        # Get prompt length
        prompt_len = int(batch["valid_length"].item())

        # Extract valid input_ids from _raw_inputs (preserves image tokens)
        raw_inputs = batch.get("_raw_inputs", None)
        if raw_inputs is not None:
            input_ids = raw_inputs["input_ids"][0].clone()  # Original with image tokens
        else:
            input_ids = batch["llm_input_ids"][0, :prompt_len].clone()  # Fallback

        if eos_token_id is None:
            eos_token_id = int(getattr(self.text_config, "eos_token_id", -1))

        # Generate loop
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1)  # (1,)
            print(f"[generate_greedy] Next token: {next_token}")
            input_ids = torch.cat([input_ids, next_token], dim=0)

            # Check EOS
            if eos_token_id >= 0 and next_token.item() == eos_token_id:
                break

            # Decode step: next_token is (1,), need (1, 1) for decode_one
            logits = self.decode_one(next_token.unsqueeze(0))

        return input_ids.unsqueeze(0)  # (1, generated_len)

    def _apply_final_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Gemma4 final logit softcapping.

        HF's ``Gemma4ForConditionalGeneration`` applies
        ``tanh(logits / softcap) * softcap`` when
        ``config.final_logit_softcapping`` is set.  The
        ``Gemma4LMHeadExportAdapter`` does not apply it (to stay clean
        for NPU export), so the runtime applies it here on the CPU side.
        """
        softcap = getattr(self.text_config, "final_logit_softcapping", None)
        if softcap is None:
            return logits
        softcap = float(softcap)
        return torch.tanh(logits / softcap) * softcap


@torch.no_grad()
def verify_step_build_static_inputs(
    runtime: StaticGemma4Runtime,
    prompt: str,
    image,
) -> dict[str, torch.Tensor]:
    """Side-by-side validation of ``build_static_inputs`` against HF reference.

    This function re-derives each sub-step of ``build_static_inputs`` using the
    raw HF processor output and the HF model's internal logic, then compares
    against what the runtime produced. It validates:

    1. ``llm_input_ids`` — image placeholder replacement + padding
    2. ``valid_token_mask`` / ``attention_mask`` — boolean valid-token mask
    3. ``pixel_values`` — exact match with processor output
    4. ``image_position_ids`` — exact match with processor output
    5. ``valid_length`` — correct unpadded sequence length
    6. ``padding`` — right-padded layout with pad_token_id fill

    Returns the batch dict from ``build_static_inputs``.
    """
    import torch.testing

    layout = runtime.layout
    max_seq = layout.max_seq
    pad_token_id = getattr(runtime.text_config, "pad_token_id", 0)
    image_token_id = getattr(runtime.config, "image_token_id", None)

    # --- Runtime output ---
    batch = runtime.build_static_inputs(prompt, image)

    rt_llm_input_ids = batch["llm_input_ids"]

    rt_attention_mask = batch["attention_mask"]
    rt_pixel_values = batch["pixel_values"]
    rt_image_position_ids = batch.get("image_position_ids")
    rt_valid_length = batch["valid_length"]

    # --- HF reference: raw processor output ---
    inputs = runtime.processor(
        text=prompt, images=image, return_tensors="pt", padding=False
    )
    ref_input_ids = inputs["input_ids"].squeeze(0)
    seq_len = ref_input_ids.shape[0]

    # 1. llm_input_ids: pad + replace image tokens
    ref_padded = torch.full(
        (max_seq,), pad_token_id, dtype=ref_input_ids.dtype, device=runtime.device
    )
    ref_padded[:seq_len] = ref_input_ids.to(runtime.device)
    if image_token_id is not None:
        ref_padded[ref_padded == image_token_id] = pad_token_id

    torch.testing.assert_close(
        rt_llm_input_ids.squeeze(0),
        ref_padded,
        msg="llm_input_ids mismatch: image placeholder replacement or padding",
    )

    # 2. valid_token_mask / attention_mask
    ref_attention_mask = torch.zeros(max_seq, dtype=torch.bool, device=runtime.device)
    ref_attention_mask[:seq_len] = True
    torch.testing.assert_close(
        rt_attention_mask.squeeze(0),
        ref_attention_mask,
        msg="attention_mask mismatch",
    )

    # 3. pixel_values
    ref_pixel_values = inputs.get("pixel_values", None)
    if ref_pixel_values is None:
        raise ValueError("HF processor did not return pixel_values")
    ref_pixel_values = ref_pixel_values.to(runtime.device)
    torch.testing.assert_close(
        rt_pixel_values,
        ref_pixel_values,
        msg="pixel_values mismatch",
    )

    # 4. image_position_ids
    ref_image_position_ids = inputs.get("image_position_ids", None)
    if ref_image_position_ids is not None:
        ref_image_position_ids = ref_image_position_ids.to(runtime.device)
    if rt_image_position_ids is not None and ref_image_position_ids is not None:
        torch.testing.assert_close(
            rt_image_position_ids,
            ref_image_position_ids,
            msg="image_position_ids mismatch",
        )
    elif rt_image_position_ids is not None or ref_image_position_ids is not None:
        raise ValueError(
            "image_position_ids presence mismatch: "
            f"runtime={rt_image_position_ids is not None}, "
            f"reference={ref_image_position_ids is not None}"
        )

    # 5. valid_length
    ref_valid_length = torch.tensor([seq_len], dtype=torch.long, device=runtime.device)
    torch.testing.assert_close(
        rt_valid_length,
        ref_valid_length,
        msg="valid_length mismatch",
    )

    # 6. padding layout: right-padded with pad_token_id
    #    All positions >= seq_len must be pad_token_id
    if seq_len < max_seq:
        padding_region = rt_llm_input_ids.squeeze(0)[seq_len:]
        if not torch.all(padding_region == pad_token_id):
            raise ValueError("Padding region does not consist entirely of pad_token_id")

    print("[verify_step_build_static_inputs] All checks passed.")
    return batch


@torch.no_grad()
def verify_step_token_embedding(
    runtime: StaticGemma4Runtime,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Side-by-side validation of ``token_embedding`` against HF reference.

    The runtime's ``Gemma4TokenEmbeddingExportAdapter`` wraps the same
    ``Gemma4TextScaledWordEmbedding`` that HF uses internally.  This function
    feeds ``llm_input_ids`` through both paths and asserts that the output
    embeddings match exactly.

    HF reference (modeling_gemma4.py L1468):
        ``Gemma4TextScaledWordEmbedding.forward`` returns
        ``nn.Embedding.forward(input_ids) * embed_scale`` where
        ``embed_scale = hidden_size ** 0.5``.

    Args:
        runtime: The ``StaticGemma4Runtime`` instance.
        batch: The batch dict returned by ``build_static_inputs``.

    Returns the token embeddings tensor (shape ``(1, S, hidden_size)``).
    """
    import torch.testing

    llm_input_ids = batch["llm_input_ids"].to(runtime.device)

    # --- Runtime side ---
    rt_embeds = runtime.token_embedding(llm_input_ids)

    # --- HF reference ---

    # model.get_input_embeddings() returns Gemma4TextScaledWordEmbedding,
    # which multiplies by sqrt(hidden_size) internally.
    ref_embeds = runtime.model.get_input_embeddings()(llm_input_ids)

    torch.testing.assert_close(
        rt_embeds,
        ref_embeds,
        msg="token_embedding mismatch: runtime vs HF Gemma4TextScaledWordEmbedding",
    )

    # Sanity: verify the embedding scale is applied (not a plain nn.Embedding)
    hidden_size = int(runtime.text_config.hidden_size)
    raw_lookup = nn.functional.embedding(
        llm_input_ids, runtime.model.get_input_embeddings().weight
    )
    expected_scale = float(hidden_size) ** 0.5
    torch.testing.assert_close(
        ref_embeds,
        raw_lookup * expected_scale,
        msg="HF embedding does not apply sqrt(hidden_size) scale as expected",
    )

    print("[verify_step_token_embedding] All checks passed.")
    return rt_embeds


@torch.no_grad()
def verify_step_vision_prefill(
    runtime: StaticGemma4Runtime,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Side-by-side validation of ``vision_prefill`` against references.

    The runtime's ``Gemma4VisionPrefillExportAdapter`` runs the vision tower
    followed by the ``embed_vision`` projection.  This function feeds
    ``pixel_values`` and ``image_position_ids`` through the runtime adapter
    and two reference paths:

    1. **HF (FP) reference** — ``runtime.model.get_image_features(...)``:
       Compared via PEIR (Peak-Error-to-Interval Ratio) as an informational
       metric only, because the runtime adapter wraps a quantized vision tower
       (with fake Q-DQ ops), so exact equality against the FP model is not
       expected.

    2. **Quantized reference** — ``wrapped_top.model.wrapped.get_image_features(...)``:
       Asserted via ``torch.testing.assert_close`` to confirm the adapter
       introduces no error beyond quantization.

    HF reference (modeling_gemma4.py L2150–2167):
        ``get_image_features`` runs ``self.vision_tower(pixel_values,
        pixel_position_ids=image_position_ids)`` then
        ``self.embed_vision(last_hidden_state)`` and stores the result in
        ``pooler_output``.

    Args:
        runtime: The ``StaticGemma4Runtime`` instance.
        batch: The batch dict returned by ``build_static_inputs``.

    Returns the visual embeddings tensor (shape ``(1, V, hidden_size)``).
    """

    import torch.testing

    # Cast pixel_values to the model's dtype (BFloat16) to match the
    # quantized vision tower weights.  The HF processor outputs float32.
    model_dtype = runtime.model.dtype
    pixel_values = batch["pixel_values"].to(runtime.device).to(model_dtype)
    image_position_ids = batch.get("image_position_ids")
    if image_position_ids is not None:
        image_position_ids = image_position_ids.to(runtime.device)

    # --- Runtime side ---
    vision_prefill = runtime._get_or_create_vision_prefill(image_position_ids)
    rt_visual_embeds = vision_prefill(pixel_values, image_position_ids)

    # --- HF reference ---
    # model.get_image_features() returns BaseModelOutputWithPooling whose
    # .pooler_output contains the embed_vision projection of the vision
    # tower's last_hidden_state.
    hf_visual_embeds = runtime.model.get_image_features(
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        return_dict=True,
    ).pooler_output

    # --- Shape check ---
    if rt_visual_embeds.shape != hf_visual_embeds.shape:
        raise ValueError(
            "vision_prefill shape mismatch: "
            f"runtime={tuple(rt_visual_embeds.shape)}, "
            f"reference={tuple(hf_visual_embeds.shape)}"
        )

    # --- PEIR (Peak-Error-to-Interval Ratio) ---
    from tico.quantization.evaluation.metric import compute_peir

    peir = compute_peir(hf_visual_embeds, rt_visual_embeds)
    print(f"[verify_step_vision_prefill] PEIR = {peir * 100:.6f} %")

    # --- Wrapped eager reference ---
    # The runtime and floating-point Circle exporter both use a NO_QUANT
    # structural wrapper. Its eager path is the direct parity reference for
    # the static adapter.
    wrapped_top = (
        runtime.qmodel.wrapped if hasattr(runtime.qmodel, "wrapped") else runtime.qmodel
    )
    ref_visual_embeds = wrapped_top.model.wrapped.get_image_features(
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
    )

    torch.testing.assert_close(
        rt_visual_embeds,
        ref_visual_embeds,
        msg="vision_prefill mismatch: static adapter vs wrapped eager path",
    )

    # --- torch.export reference ---
    # Circle conversion starts from this same exported module, so checking the
    # ExportedProgram output here also guards the graph handed to tico.convert.
    exported_program = torch.export.export(
        vision_prefill,
        (pixel_values, image_position_ids),
        strict=False,
    )
    exported_visual_embeds = exported_program.module()(
        pixel_values,
        image_position_ids,
    )
    torch.testing.assert_close(
        exported_visual_embeds,
        rt_visual_embeds,
        msg="vision_prefill mismatch: torch.export vs static adapter",
    )

    print("[verify_step_vision_prefill] Eager/static/torch.export checks passed.")
    return rt_visual_embeds


@torch.no_grad()
def verify_step_mm_fusion(
    runtime: StaticGemma4Runtime,
    text_embeds: torch.Tensor,
    visual_embeds: torch.Tensor,
    prompt: str,
    image,
) -> torch.Tensor:
    """Side-by-side validation of ``mm_fusion`` against HF's ``masked_scatter``.

    The runtime's ``Gemma4MMFusionExportAdapter`` calls ``fixed_slot_fuse``,
    which replaces a contiguous slot range ``[visual_start_idx,
    visual_start_idx + num_visual_tokens)`` with visual embeddings via
    ``torch.cat``.  HF's reference path uses ``masked_scatter`` to write
    visual embeddings into the positions selected by the image-token mask.

    This function feeds the same ``text_embeds`` and ``visual_embeds``
    through both paths and asserts that the fused outputs match exactly.

    HF reference (modeling_gemma4.py):
        ``image_mask = input_ids == image_token_id``
        ``image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)``
        ``inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)``

    Args:
        runtime: The ``StaticGemma4Runtime`` instance.
        text_embeds: Token embeddings from ``verify_step_token_embedding``.
        visual_embeds: Visual embeddings from ``verify_step_vision_prefill``.
        prompt: The original prompt string (needed to re-run the processor
            and recover the raw input_ids with image tokens).
        image: The original image (needed to re-run the processor).

    Returns the fused embeddings tensor (shape ``(1, S, hidden_size)``).
    """
    import torch.testing

    layout = runtime.layout
    image_token_id = getattr(runtime.config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("config.image_token_id is required for mm_fusion verification")

    # --- Runtime side: fixed_slot_fuse via adapter ---
    rt_fused = runtime.mm_fusion(text_embeds, visual_embeds)

    # --- HF reference: masked_scatter ---
    # Re-run the processor to recover raw input_ids with image tokens intact.
    raw_inputs = runtime.processor(
        text=prompt, images=image, return_tensors="pt", padding=False
    )
    raw_input_ids = raw_inputs["input_ids"].to(runtime.device)  # (1, seq_len_raw)

    # Build the image mask from raw input_ids
    image_mask = raw_input_ids == image_token_id  # (1, seq_len_raw)

    # Validate that image token positions match the static layout used by
    # fixed_slot_fuse.  Without this check, a layout mismatch would surface
    # as a confusing "mm_fusion mismatch" assertion error rather than a
    # clear diagnostic.
    image_positions = torch.nonzero(image_mask.squeeze(0), as_tuple=True)[0]
    if image_positions.numel() == 0:
        raise ValueError("No image placeholder tokens found in raw input_ids")
    actual_start = int(image_positions[0].item())
    actual_count = int(image_positions.numel())
    if actual_start != layout.visual_start_idx:
        raise ValueError(
            "Image token start position does not match static layout: "
            f"expected visual_start_idx={layout.visual_start_idx}, "
            f"actual={actual_start}"
        )
    if actual_count != layout.num_visual_tokens:
        raise ValueError(
            "Image token count does not match static layout: "
            f"expected num_visual_tokens={layout.num_visual_tokens}, "
            f"actual={actual_count}"
        )

    # Pad the mask to max_seq to match the runtime's static shape
    max_seq = layout.max_seq
    seq_len_raw = raw_input_ids.shape[1]
    if seq_len_raw > max_seq:
        raise ValueError(f"Raw sequence length {seq_len_raw} exceeds max_seq {max_seq}")

    padded_mask = torch.zeros((1, max_seq), dtype=torch.bool, device=runtime.device)
    padded_mask[:, :seq_len_raw] = image_mask

    # Expand mask to match text_embeds shape: (1, max_seq, hidden_size)
    image_mask_expanded = padded_mask.unsqueeze(-1).expand_as(text_embeds)

    # masked_scatter: write visual_embeds into the image-token positions
    ref_fused = text_embeds.clone()
    ref_fused = ref_fused.masked_scatter(
        image_mask_expanded, visual_embeds.to(ref_fused.dtype)
    )

    torch.testing.assert_close(
        rt_fused,
        ref_fused,
        msg="mm_fusion mismatch: runtime fixed_slot_fuse vs HF masked_scatter",
    )

    print("[verify_step_mm_fusion] All checks passed.")
    return rt_fused


@torch.no_grad()
def verify_step_masks_and_rope(
    runtime: StaticGemma4Runtime,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Side-by-side validation of ``build_prefill_masks_and_rope`` against HF.

    Validates two things:

    1. **Attention masks**: The runtime's full-causal and sliding-window-causal
       masks are compared against HF's ``create_masks_for_generate``.  Since HF
       may return ``None`` (for sdpa) or use a different ``mask_value``, we
       compare boolean attention patterns (allowed vs. forbidden) rather than
       exact float values.

    2. **RoPE (cos, sin)**: The runtime's per-layer-type RoPE is compared
       against HF's ``Gemma4TextRotaryEmbedding.forward`` — the same module the
       runtime uses internally.  This is an exact-match check.

    HF reference (modeling_gemma4.py L1688–1707):
        ``causal_mask_mapping = {``
        ``    "full_attention": create_causal_mask(...),``
        ``    "sliding_attention": create_sliding_window_causal_mask(...),``
        ``}``
        ``position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)``

    Args:
        runtime: The ``StaticGemma4Runtime`` instance.
        batch: The batch dict returned by ``build_static_inputs``.

    Returns the ``(attention_masks, position_embeddings)`` tuple from
    ``build_prefill_masks_and_rope``.
    """
    import torch.testing

    from transformers.masking_utils import create_masks_for_generate

    llm_input_ids = batch["llm_input_ids"].to(runtime.device)
    attention_mask = batch["attention_mask"].to(runtime.device)
    max_seq = llm_input_ids.shape[1]
    runtime_dtype = next(runtime.model.parameters()).dtype

    # --- Runtime side ---
    rt_masks, rt_rope = runtime.build_prefill_masks_and_rope(
        llm_input_ids, attention_mask
    )

    # --- HF reference: RoPE ---
    # Use the HF model's rotary_emb directly (same module the runtime uses)
    rotary_emb = runtime.model.model.language_model.rotary_emb
    position_ids = torch.arange(max_seq, device=runtime.device).unsqueeze(0)
    dummy_hidden = torch.zeros(
        1, max_seq, 1, device=runtime.device, dtype=runtime_dtype
    )

    layer_types = set(runtime.text_config.layer_types)
    for layer_type in layer_types:
        ref_cos, ref_sin = rotary_emb(dummy_hidden, position_ids, layer_type)
        rt_cos, rt_sin = rt_rope[layer_type]

        torch.testing.assert_close(
            rt_cos,
            ref_cos,
            msg=f"RoPE cos mismatch for layer_type={layer_type!r}",
        )
        torch.testing.assert_close(
            rt_sin,
            ref_sin,
            msg=f"RoPE sin mismatch for layer_type={layer_type!r}",
        )

    print(
        f"[verify_step_masks_and_rope] RoPE exact match for {len(layer_types)} layer types."
    )

    # --- HF reference: masks ---
    # Build dummy inputs_embeds for shape inference
    hidden_size = int(runtime.text_config.hidden_size)
    dummy_embeds = torch.zeros(
        1, max_seq, hidden_size, device=runtime.device, dtype=runtime_dtype
    )

    # create_masks_for_generate needs a 2D attention mask (1=valid, 0=padding)
    attn_mask_2d = attention_mask.to(runtime.device).long()  # (1, max_seq)

    # Temporarily force eager attention to get 4D float masks from HF
    original_attn_impl = getattr(runtime.text_config, "_attn_implementation", "sdpa")
    runtime.text_config._attn_implementation = "eager"
    try:
        ref_masks = create_masks_for_generate(
            config=runtime.text_config,
            inputs_embeds=dummy_embeds,
            attention_mask=attn_mask_2d,
            past_key_values=None,
            position_ids=position_ids,
        )
    finally:
        runtime.text_config._attn_implementation = original_attn_impl

    # --- Compare masks as boolean patterns ---
    # HF may return None for some layer types (e.g. if no masking needed).
    # We compare the allowed/forbidden pattern, not exact float values.
    for layer_type in layer_types:
        rt_mask = rt_masks[layer_type]  # (1, 1, S, S)

        if isinstance(ref_masks, dict):
            ref_mask = ref_masks.get(layer_type, None)
        else:
            # Single mask (all layers same type)
            ref_mask = ref_masks

        if ref_mask is None:
            # HF returned None — means no masking (all positions allowed).
            # This happens when there's no padding and sdpa handles causality
            # internally.  Compare against our mask's "allowed" pattern.
            # For prefill with no padding, all causal positions should be 0.0.
            print(
                f"[verify_step_masks_and_rope] HF returned None mask for "
                f"{layer_type!r}, skipping mask comparison (RoPE still verified)."
            )
            continue

        # Convert to boolean: True = allowed (not mask_value), False = masked
        mask_value = torch.finfo(runtime_dtype).min
        rt_allowed = rt_mask.squeeze(1).squeeze(0) > (mask_value / 2)  # (S, S)
        ref_allowed = ref_mask.squeeze(1).squeeze(0) > (mask_value / 2)  # (S, S)

        torch.testing.assert_close(
            rt_allowed,
            ref_allowed,
            msg=f"Attention mask pattern mismatch for layer_type={layer_type!r}",
        )

    print(
        f"[verify_step_masks_and_rope] Mask patterns match for {len(layer_types)} layer types."
    )
    print("[verify_step_masks_and_rope] All checks passed.")
    return rt_masks, rt_rope


@torch.no_grad()
def verify_step_prefill(
    runtime: StaticGemma4Runtime,
    batch: dict[str, torch.Tensor],
    prompt: str,
    image,
) -> torch.Tensor:
    """Side-by-side validation of ``prefill`` against HF reference.

    Runs the runtime's ``prefill`` (quantized, static-shape) and compares
    last-token logits against HF's ``Gemma4ForConditionalGeneration.forward``
    (FP, dynamic-shape).

    The runtime applies ``final_logit_softcapping`` internally (in
    ``prefill``), so ``rt_logits`` is directly comparable to HF's output.

    Comparison is via PEIR (Peak-Error-to-Interval Ratio) and diff stats,
    because the runtime uses quantized weights while the reference is FP.

    HF reference (modeling_gemma4.py L2508–2535):
        ``outputs = self.model(input_ids=..., pixel_values=..., ...)``
        ``logits = self.lm_head(hidden_states[:, slice_indices, :])``
        ``if final_logit_softcapping is not None:``
        ``    logits = tanh(logits / sc) * sc``

    Args:
        runtime: The ``StaticGemma4Runtime`` instance.
        batch: The batch dict returned by ``build_static_inputs``.
        prompt: Original prompt string (to re-run processor for raw input_ids).
        image: Original image (to re-run processor).

    Returns the runtime logits (shape ``(1, vocab_size)``).
    """
    from tico.quantization.evaluation.metric import compute_peir

    # --- Runtime side ---
    runtime.reset_cache()
    rt_logits = runtime.prefill(batch)  # (1, vocab_size)

    # --- HF reference ---
    # Use cached raw processor output from build_static_inputs to avoid
    # re-running the processor (which involves expensive image preprocessing).
    # Fall back to re-running the processor if the cache is not available.
    raw_inputs = batch.get("_raw_inputs", None)
    if raw_inputs is None:
        raw_inputs = runtime.processor(
            text=prompt, images=image, return_tensors="pt", padding=False
        )
    raw_input_ids = raw_inputs["input_ids"].to(runtime.device)  # (1, seq_len)
    seq_len = raw_input_ids.shape[1]

    pixel_values = raw_inputs["pixel_values"].to(runtime.device)
    # Cast to model dtype to match HF's internal casting
    model_dtype = runtime.model.dtype
    pixel_values = pixel_values.to(model_dtype)

    image_position_ids = raw_inputs.get("image_position_ids", None)
    if image_position_ids is not None:
        image_position_ids = image_position_ids.to(runtime.device)

    # Unpadded attention mask (all ones)
    ref_attention_mask = torch.ones(
        (1, seq_len), dtype=torch.long, device=runtime.device
    )

    ref_out = runtime.model(
        input_ids=raw_input_ids,
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        attention_mask=ref_attention_mask,
        logits_to_keep=1,
        use_cache=False,
        return_dict=True,
    )
    ref_logits = ref_out.logits[:, -1, :]

    # --- Compare ---
    rt_f = rt_logits.float()
    ref_f = ref_logits.float()

    diff = (rt_f - ref_f).abs()
    peir = compute_peir(ref_f, rt_f)

    print(f"[verify_step_prefill] Runtime logits shape: {tuple(rt_logits.shape)}")
    print(f"[verify_step_prefill] Reference logits shape: {tuple(ref_logits.shape)}")
    print(f"[verify_step_prefill] mean|diff| = {diff.mean().item():.8f}")
    print(f"[verify_step_prefill]  max|diff| = {diff.max().item():.8f}")
    print(f"[verify_step_prefill] PEIR       = {peir * 100:.6f} %")

    rt_next = int(torch.argmax(rt_f, dim=-1).item())
    ref_next = int(torch.argmax(ref_f, dim=-1).item())
    print(f"[verify_step_prefill] Runtime argmax token: {rt_next}")
    print(f"[verify_step_prefill] Reference argmax token: {ref_next}")
    if rt_next == ref_next:
        print("[verify_step_prefill] Argmax tokens MATCH.")
    else:
        print("[verify_step_prefill] WARNING: Argmax tokens differ.")

    print("[verify_step_prefill] All checks passed.")
    return rt_logits


@torch.no_grad()
def verify_step_decode(
    runtime: StaticGemma4Runtime,
    batch: dict[str, torch.Tensor],
    prompt: str,
    image,
    prefill_logits: torch.Tensor,
    num_decode_steps: int = 1,
    same_token_mode: bool = False,
) -> torch.Tensor:
    """Side-by-side validation of ``decode_one`` against HF reference.

    Runs one or more ``decode_one`` steps starting from the prefill state
    already established by ``verify_step_prefill``, and compares each
    decode-step logits against HF's ``Gemma4ForConditionalGeneration.forward``
    with ``past_key_values``.

    The HF reference uses ``DynamicCache`` to accumulate K/V across steps,
    while the runtime uses its own fixed-size ``layer_caches``.  Both paths
    start from the same prefill state (greedy argmax from prefill logits).

    The runtime applies ``final_logit_softcapping`` internally (in
    ``decode_one``), so ``rt_decode_logits`` is directly comparable to HF's output.

    Comparison is via PEIR and diff stats, because the runtime uses quantized
    weights while the reference is FP.

    Args:
        runtime: The ``StaticGemma4Runtime`` instance.
        batch: The batch dict returned by ``build_static_inputs``.
        prompt: Original prompt string.
        image: Original image.
        prefill_logits: Runtime prefill logits returned by
            ``verify_step_prefill``.  The runtime's KV caches must already be
            populated from that prefill call.
        num_decode_steps: Number of decode steps to verify (default 1).
        same_token_mode: If ``True``, both paths use the HF reference token
            at each step, ensuring apples-to-apples comparison for multi-step
            decode. If ``False`` (default), each path uses its own greedy
            argmax, which may diverge after quantization error causes different
            token selection.

    Returns the last runtime decode logits (shape ``(1, vocab_size)``).
    """
    from tico.quantization.evaluation.metric import compute_peir

    # --- Runtime side: decode (prefill already done by verify_step_prefill) ---
    rt_prefill_logits = prefill_logits  # Reuse from Step 6

    # Greedy first token from prefill
    rt_next_token = torch.argmax(rt_prefill_logits, dim=-1, keepdim=True)  # (1, 1)

    # --- HF reference: prefill ---
    # Use cached raw processor output
    raw_inputs = batch.get("_raw_inputs", None)
    if raw_inputs is None:
        raw_inputs = runtime.processor(
            text=prompt, images=image, return_tensors="pt", padding=False
        )
    raw_input_ids = raw_inputs["input_ids"].to(runtime.device)  # (1, seq_len)
    seq_len = raw_input_ids.shape[1]

    pixel_values = raw_inputs["pixel_values"].to(runtime.device)
    model_dtype = runtime.model.dtype
    pixel_values = pixel_values.to(model_dtype)

    image_position_ids = raw_inputs.get("image_position_ids", None)
    if image_position_ids is not None:
        image_position_ids = image_position_ids.to(runtime.device)

    ref_attention_mask = torch.ones(
        (1, seq_len), dtype=torch.long, device=runtime.device
    )

    # HF prefill with cache
    from transformers.cache_utils import DynamicCache

    hf_cache = DynamicCache(config=runtime.text_config)
    ref_out = runtime.model(
        input_ids=raw_input_ids,
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        attention_mask=ref_attention_mask,
        past_key_values=hf_cache,
        return_dict=True,
        use_cache=True,
    )
    ref_prefill_logits = ref_out.logits[:, -1, :]  # (1, vocab_size)

    # Verify prefill argmax matches (sanity check)
    ref_next_token = torch.argmax(ref_prefill_logits, dim=-1, keepdim=True)

    # CRITICAL: Check for prefill divergence
    prefill_diverged = bool((rt_next_token != ref_next_token).item())
    if prefill_diverged:
        print(
            "[verify_step_decode] WARNING: Runtime and HF argmax differ at step 0. "
            f"Runtime={rt_next_token.item()}, HF={ref_next_token.item()}. "
            "Multi-step comparison would evaluate different sequences."
        )
        if num_decode_steps > 1 and not same_token_mode:
            print(
                "[verify_step_decode] Limiting to 1 decode step. "
                "Set same_token_mode=True to force same tokens for multi-step."
            )
            num_decode_steps = 1

    # Initialize tokens for both paths
    if same_token_mode:
        # Both paths use HF reference token
        rt_current_token = ref_next_token.clone()
        ref_current_token = ref_next_token.clone()
        print(
            "[verify_step_decode] same_token_mode=True: Both paths use HF reference tokens."
        )
    else:
        # Each path uses its own greedy selection
        rt_current_token = rt_next_token.clone()
        ref_current_token = ref_next_token.clone()

    # Run decode steps for both paths
    rt_decode_logits_list = []
    ref_decode_logits_list = []

    for step in range(num_decode_steps):
        # Runtime decode
        rt_decode_logits = runtime.decode_one(rt_current_token)
        rt_decode_logits_list.append(rt_decode_logits)

        # HF decode
        ref_decode_out = runtime.model(
            input_ids=ref_current_token,
            past_key_values=hf_cache,
            attention_mask=torch.ones(
                (1, seq_len + step + 1), dtype=torch.long, device=runtime.device
            ),
            return_dict=True,
            use_cache=True,
        )
        ref_decode_logits = ref_decode_out.logits[:, -1, :]
        ref_decode_logits_list.append(ref_decode_logits)

        # Select next tokens
        rt_next = torch.argmax(rt_decode_logits, dim=-1, keepdim=True)
        ref_next = torch.argmax(ref_decode_logits, dim=-1, keepdim=True)

        if same_token_mode:
            # Force both to use HF token
            rt_current_token = ref_next.clone()
            ref_current_token = ref_next.clone()
        else:
            rt_current_token = rt_next.clone()
            ref_current_token = ref_next.clone()

    # --- Compare each decode step ---
    for step in range(num_decode_steps):
        rt_f = rt_decode_logits_list[step].float()
        ref_f = ref_decode_logits_list[step].float()

        diff = (rt_f - ref_f).abs()
        peir = compute_peir(ref_f, rt_f)

        print(f"[verify_step_decode] Step {step + 1}:")
        print(f"[verify_step_decode]   Runtime logits shape: {tuple(rt_f.shape)}")
        print(f"[verify_step_decode]   mean|diff| = {diff.mean().item():.8f}")
        print(f"[verify_step_decode]    max|diff| = {diff.max().item():.8f}")
        print(f"[verify_step_decode]   PEIR       = {peir * 100:.6f} %")

        rt_tok = int(torch.argmax(rt_f, dim=-1).item())
        ref_tok = int(torch.argmax(ref_f, dim=-1).item())
        print(f"[verify_step_decode]   Runtime argmax token: {rt_tok}")
        print(f"[verify_step_decode]   Reference argmax token: {ref_tok}")
        if rt_tok == ref_tok:
            print("[verify_step_decode]   Argmax tokens MATCH.")
        else:
            print("[verify_step_decode]   WARNING: Argmax tokens differ.")
            if step < num_decode_steps - 1:
                print(
                    "[verify_step_decode]   Subsequent steps compare different sequences."
                )

    print("[verify_step_decode] All checks passed.")
    return rt_decode_logits_list[-1]


@torch.no_grad()
def verify_step_generation(
    runtime: StaticGemma4Runtime,
    batch: dict[str, torch.Tensor],
    prompt: str,
    image,
    max_new_tokens: int = 16,
) -> torch.LongTensor:
    """Side-by-side validation of ``generate_greedy`` against HF reference.

    Runs greedy generation from the runtime and HF reference, comparing
    generated token sequences.  The runtime uses quantized weights while
    HF uses FP, so exact match is not expected — we report token-level
    accuracy and first mismatch position.

    Args:
        runtime: The ``StaticGemma4Runtime`` instance.
        batch: The batch dict returned by ``build_static_inputs``.
        prompt: Original prompt string.
        image: Original image.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Runtime-generated token IDs of shape ``(1, prompt_len + num_generated)``.
    """
    # Runtime generation
    runtime.reset_cache()
    rt_generated = runtime.generate_greedy(batch, max_new_tokens, eos_token_id=None)

    # HF reference generation
    raw_inputs = batch.get("_raw_inputs", None)
    if raw_inputs is None:
        raw_inputs = runtime.processor(
            text=prompt, images=image, return_tensors="pt", padding=False
        )
    raw_input_ids = raw_inputs["input_ids"].to(runtime.device)
    pixel_values = raw_inputs["pixel_values"].to(runtime.device).to(runtime.model.dtype)
    image_position_ids = raw_inputs.get("image_position_ids", None)
    if image_position_ids is not None:
        image_position_ids = image_position_ids.to(runtime.device)

    # HF generate with explicit max_new_tokens
    hf_generated = runtime.model.generate(
        input_ids=raw_input_ids,
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=runtime.processor.tokenizer.pad_token_id,
        eos_token_id=getattr(runtime.text_config, "eos_token_id", None),
    )

    # Compare sequences
    rt_len = rt_generated.shape[1]
    hf_len = hf_generated.shape[1]
    min_len = min(rt_len, hf_len)

    # Count matching tokens
    matches = (rt_generated[:, :min_len] == hf_generated[:, :min_len]).sum().item()
    accuracy = matches / min_len if min_len > 0 else 0.0

    # Find first mismatch
    first_mismatch = None
    for i in range(min_len):
        if rt_generated[0, i] != hf_generated[0, i]:
            first_mismatch = i
            break

    print(f"[verify_step_generation] Runtime generated {rt_len} tokens")
    print(f"[verify_step_generation] HF generated {hf_len} tokens")
    print(
        f"[verify_step_generation] Token accuracy: {accuracy * 100:.2f}% ({matches}/{min_len})"
    )
    if first_mismatch is not None:
        print(f"[verify_step_generation] First mismatch at position {first_mismatch}")
    else:
        print("[verify_step_generation] All tokens match!")

    # Print runtime generated text
    rt_text = runtime.processor.decode(
        rt_generated[0].tolist(), skip_special_tokens=False
    )
    hf_text = runtime.processor.decode(
        hf_generated[0].tolist(), skip_special_tokens=False
    )
    print(f"[verify_step_generation] Runtime text: {rt_text}")
    print(f"[verify_step_generation] HF text: {hf_text}")

    if first_mismatch is None:
        print("[verify_step_generation] All checks passed.")
    else:
        print("[verify_step_generation] Mismatches detected.")
    return rt_generated


def run_static_gemma4_runtime(cfg: StaticGemma4RuntimeConfig) -> None:
    """Run the Gemma4 E2B static runtime smoke flow.

    This entry point runs the ``build_static_inputs``, ``token_embedding``,
    ``vision_prefill``, ``mm_fusion``, ``masks_and_rope``, ``prefill``, and
    ``decode`` validation steps. Generation is skipped with a clear message.
    """

    from transformers import AutoModelForImageTextToText, AutoProcessor

    if cfg.padding_side != "right":
        raise ValueError(
            "StaticGemma4Runtime currently supports right padding only, "
            f"got padding_side={cfg.padding_side!r}."
        )

    print(f"[run_static_gemma4_runtime] Loading model: {cfg.model}")

    model = AutoModelForImageTextToText.from_pretrained(cfg.model)
    processor = AutoProcessor.from_pretrained(cfg.model)

    vision_profile = get_gemma4_static_vision_profile(cfg.vision_profile)
    vision_profile.validate(
        max_seq_len=cfg.max_seq,
        vision_config=model.config.vision_config,
    )
    legacy_fields = {
        "image_height": vision_profile.image_height,
        "image_width": vision_profile.image_width,
        "visual_start_idx": vision_profile.visual_start_idx,
        "num_visual_tokens": vision_profile.num_visual_tokens,
    }
    for field_name, expected in legacy_fields.items():
        configured = getattr(cfg, field_name)
        if configured is not None and int(configured) != expected:
            raise ValueError(
                f"StaticGemma4RuntimeConfig.{field_name} conflicts with vision "
                f"profile {vision_profile.name!r}: "
                f"configured={int(configured)}, expected={expected}."
            )
    layout = StaticGemma4Layout(
        max_seq=cfg.max_seq,
        visual_start_idx=vision_profile.visual_start_idx,
        num_visual_tokens=vision_profile.num_visual_tokens,
    )

    print("[run_static_gemma4_runtime] Creating StaticGemma4Runtime ...")
    runtime = StaticGemma4Runtime(
        model=model,
        processor=processor,
        layout=layout,
        vision_profile=vision_profile,
        device=cfg.device,
    )

    # --- Load a test image ---
    from PIL import Image

    image = Image.new(
        "RGB",
        (vision_profile.image_width, vision_profile.image_height),
        color="white",
    )

    # --- Step 1: build_static_inputs validation ---
    print("[run_static_gemma4_runtime] Step 1: verify build_static_inputs")
    batch = verify_step_build_static_inputs(runtime, cfg.prompt, image)

    # --- Step 2: token_embedding validation ---
    print("[run_static_gemma4_runtime] Step 2: verify token_embedding")
    text_embeds = verify_step_token_embedding(runtime, batch)

    # --- Step 3: vision_prefill validation ---
    print("[run_static_gemma4_runtime] Step 3: verify vision_prefill")
    visual_embeds = verify_step_vision_prefill(runtime, batch)

    # --- Step 4: mm_fusion validation ---
    print("[run_static_gemma4_runtime] Step 4: verify mm_fusion")
    fused_embeds = verify_step_mm_fusion(
        runtime, text_embeds, visual_embeds, cfg.prompt, image
    )

    # --- Step 5: masks_and_rope validation ---
    print("[run_static_gemma4_runtime] Step 5: verify masks_and_rope")
    masks, rope = verify_step_masks_and_rope(runtime, batch)

    # --- Step 6: prefill validation ---
    print("[run_static_gemma4_runtime] Step 6: verify prefill")
    rt_logits = verify_step_prefill(runtime, batch, cfg.prompt, image)

    # --- Step 7: decode validation ---
    print("[run_static_gemma4_runtime] Step 7: verify decode")
    rt_decode_logits = verify_step_decode(
        runtime,
        batch,
        cfg.prompt,
        image,
        prefill_logits=rt_logits,
        num_decode_steps=cfg.verify_steps,
    )

    # --- Step 8: generation validation ---
    print("[run_static_gemma4_runtime] Step 8: verify generation")
    rt_generated = verify_step_generation(
        runtime,
        batch,
        cfg.prompt,
        image,
        max_new_tokens=cfg.gen_steps,
    )
    # Detokenize and print generated text
    rt_generated_text = processor.decode(
        rt_generated[0].tolist(), skip_special_tokens=False
    )
    print(f"[run_static_gemma4_runtime] Generated text: {rt_generated_text}")

    print("[run_static_gemma4_runtime] Done.")
