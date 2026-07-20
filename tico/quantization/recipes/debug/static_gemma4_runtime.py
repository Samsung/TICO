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
from typing import Optional

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
    image_height: int = 896
    image_width: int = 896
    visual_start_idx: int = 1
    num_visual_tokens: int = 256
    padding_side: str = "right"
    device: str = "cpu"
    prompt: str = "Describe the image."
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

        self.layer_caches: list[LayerCache] = []
        self.past_len = 0

    def reset_cache(self) -> None:
        """Reset all runtime-managed KV caches."""
        self.layer_caches = []
        self.past_len = 0

    def _allocate_empty_cache(
        self, batch_size: int, dtype: torch.dtype
    ) -> list[LayerCache]:
        """Allocate fixed-size empty KV cache tensors."""
        num_kv_heads = int(self.text_config.num_key_value_heads)
        head_dim = int(self.text_config.head_dim)
        caches = []
        for _ in range(int(self.text_config.num_hidden_layers)):
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
    ) -> dict[str, torch.Tensor]:
        """Build static padded processor inputs.

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

        # Validate that image placeholder positions match the static export profile
        if image_token_id is not None:
            raw_input_ids = inputs["input_ids"]  # (1, seq_len)
            image_mask = raw_input_ids[0] == image_token_id
            image_positions = torch.nonzero(image_mask, as_tuple=True)[0]
            if image_positions.numel() == 0:
                raise ValueError(
                    "No image placeholder tokens found in processor output"
                )
            actual_start = int(image_positions[0].item())
            actual_count = int(image_positions.numel())
            expected_positions = torch.arange(
                actual_start,
                actual_start + actual_count,
                device=image_positions.device,
            )
            if not torch.equal(image_positions, expected_positions):
                raise ValueError(
                    "Image placeholder tokens must form one contiguous span. "
                    f"positions={image_positions.tolist()}"
                )
            if actual_start != self.layout.visual_start_idx:
                raise ValueError(
                    "Processor output does not match the static export profile: "
                    f"expected visual_start_idx={self.layout.visual_start_idx}, "
                    f"actual={actual_start}"
                )
            if actual_count != self.layout.num_visual_tokens:
                raise ValueError(
                    "Processor output does not match the static export profile: "
                    f"expected num_visual_tokens={self.layout.num_visual_tokens}, "
                    f"actual={actual_count}"
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
        if image_position_ids is not None:
            image_position_ids = image_position_ids.to(self.device)

        valid_length = torch.tensor([seq_len], dtype=torch.long, device=self.device)

        return {
            "llm_input_ids": padded_input_ids.unsqueeze(0),
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
            "attention_mask": padded_attention_mask.unsqueeze(0),
            "valid_length": valid_length,
        }

    def build_prefill_masks_and_rope(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Build CPU-owned static masks and RoPE tensors for prefill.

        This skeleton returns shape-compatible placeholder tensors so downstream
        runtime code and linters can type-check while the exact Gemma4 mask and
        RoPE implementation is developed. The final implementation should
        replace this method with full/sliding attention masks and layer-type
        specific RoPE generated from the Gemma4 text configuration.
        """
        batch_size, seq_len = input_ids.shape
        runtime_dtype = torch.float32
        if attention_mask.is_floating_point():
            runtime_dtype = attention_mask.dtype

        full_mask = torch.zeros(
            batch_size,
            1,
            seq_len,
            seq_len,
            device=self.device,
            dtype=runtime_dtype,
        )
        head_dim = int(
            getattr(
                self.text_config,
                "head_dim",
                self.text_config.hidden_size // self.text_config.num_attention_heads,
            )
        )
        cos = torch.ones(
            batch_size, seq_len, head_dim, device=self.device, dtype=runtime_dtype
        )
        sin = torch.zeros_like(cos)

        layer_types = set(getattr(self.text_config, "layer_types", ["full_attention"]))
        attention_masks = {layer_type: full_mask for layer_type in layer_types}
        position_embeddings = {layer_type: (cos, sin) for layer_type in layer_types}
        return attention_masks, position_embeddings

    @torch.no_grad()
    def prefill(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run static prefill and return last-token logits."""
        llm_input_ids = batch["llm_input_ids"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        image_position_ids = batch.get("image_position_ids")
        if image_position_ids is not None:
            image_position_ids = image_position_ids.to(self.device)

        text_embeds = self.token_embedding(llm_input_ids)
        image_embeds = self.vision_prefill(pixel_values, image_position_ids)
        hidden_states = self.mm_fusion(text_embeds, image_embeds)
        self.layer_caches = self._allocate_empty_cache(
            hidden_states.shape[0], hidden_states.dtype
        )

        attention_masks, position_embeddings = self.build_prefill_masks_and_rope(
            llm_input_ids,
            batch["attention_mask"].to(self.device),
        )

        for layer_idx, layer in enumerate(self.prefill_layers):
            layer_type = self.text_config.layer_types[layer_idx]
            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
            )
            hidden_states, new_k, new_v = out
            self.layer_caches[layer_idx].past_k[:, :, : self.layout.max_seq, :] = new_k
            self.layer_caches[layer_idx].past_v[:, :, : self.layout.max_seq, :] = new_v

        self.past_len = int(batch["valid_length"].item())
        logits = self.lm_head(hidden_states[:, self.past_len - 1 : self.past_len, :])
        return logits[:, -1, :]

    def build_decode_masks_and_rope(
        self, batch_size: int, dtype: torch.dtype
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Build CPU-owned static masks and RoPE tensors for one decode step.

        This skeleton returns shape-compatible placeholder tensors so the runtime
        orchestration can be linted and extended independently. The final
        implementation should create distinct full/sliding decode masks and
        position-specific RoPE slices for each Gemma4 layer type.
        """
        mask = build_decode_attention_mask(
            batch_size=batch_size,
            past_len=self.past_len,
            max_seq=self.layout.max_seq,
            device=self.device,
            dtype=dtype,
            mask_value=-120.0,
        )
        head_dim = int(
            getattr(
                self.text_config,
                "head_dim",
                self.text_config.hidden_size // self.text_config.num_attention_heads,
            )
        )
        cos = torch.ones(batch_size, 1, head_dim, device=self.device, dtype=dtype)
        sin = torch.zeros_like(cos)

        layer_types = set(getattr(self.text_config, "layer_types", ["full_attention"]))
        attention_masks = {layer_type: mask for layer_type in layer_types}
        position_embeddings = {layer_type: (cos, sin) for layer_type in layer_types}
        return attention_masks, position_embeddings

    @torch.no_grad()
    def decode_one(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run one static decode step and return next-token logits."""
        hidden_states = self.token_embedding(input_ids.to(self.device))
        attention_masks, position_embeddings = self.build_decode_masks_and_rope(
            batch_size=hidden_states.shape[0],
            dtype=hidden_states.dtype,
        )

        for layer_idx, layer in enumerate(self.decode_layers):
            cache = self.layer_caches[layer_idx]
            layer_type = self.text_config.layer_types[layer_idx]
            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
                past_key_value=(cache.past_k, cache.past_v),
            )
            hidden_states, new_k, new_v = out
            cache.past_k[:, :, self.past_len : self.past_len + 1, :] = new_k
            cache.past_v[:, :, self.past_len : self.past_len + 1, :] = new_v

        self.past_len += 1
        return self.lm_head(hidden_states)[:, -1, :]


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
    rt_visual_embeds = runtime.vision_prefill(pixel_values, image_position_ids)

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
    print(f"[verify_step_vision_prefill] PEIR = {peir:.6e}")

    # --- Quantized reference (quantized model) ---
    # Use the quantized model's get_image_features, not the original model's,
    # because the runtime's vision_prefill adapter wraps the quantized vision
    # tower (with fake Q-DQ ops).  The quantized QuantGemma4Model.get_image_features
    # returns the projected visual soft tokens directly (not wrapped in
    # BaseModelOutputWithPooling).
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
        msg="vision_prefill mismatch: runtime vs quantized get_image_features",
    )

    print("[verify_step_vision_prefill] All checks passed.")
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


def run_static_gemma4_runtime(cfg: StaticGemma4RuntimeConfig) -> None:
    """Run the Gemma4 E2B static runtime smoke flow.

    This entry point runs the ``build_static_inputs``, ``token_embedding``,
    and ``vision_prefill`` validation steps. Prefill, decode, and generation
    are skipped with clear messages.
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

    layout = StaticGemma4Layout(
        max_seq=cfg.max_seq,
        visual_start_idx=cfg.visual_start_idx,
        num_visual_tokens=cfg.num_visual_tokens,
    )

    print("[run_static_gemma4_runtime] Creating StaticGemma4Runtime ...")
    runtime = StaticGemma4Runtime(
        model=model,
        processor=processor,
        layout=layout,
        device=cfg.device,
    )

    # --- Load a test image ---
    from PIL import Image

    image = Image.new("RGB", (cfg.image_width, cfg.image_height), color="white")

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

    # --- Steps 5+: prefill / decode / generation (not yet implemented) ---
    print("[run_static_gemma4_runtime] SKIP: prefill (not implemented in this PR)")
    print("[run_static_gemma4_runtime] SKIP: decode (not implemented in this PR)")
    print("[run_static_gemma4_runtime] SKIP: generation (not implemented in this PR)")
    print("[run_static_gemma4_runtime] Done.")
