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

    def build_static_inputs(self, prompt: str, image) -> dict[str, torch.Tensor]:
        """Build static padded processor inputs.

        TODO: Implement exact Gemma4 processor calls and visual slot validation.
        """
        raise NotImplementedError(
            "Static Gemma4 processor input builder is not wired yet."
        )

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


def run_static_gemma4_runtime(cfg: StaticGemma4RuntimeConfig) -> None:
    """Run the Gemma4 E2B static runtime smoke flow.

    TODO: Load a real image input and wire reference parity checks.
    """
    raise NotImplementedError(
        "Gemma4 static runtime entry point is not fully implemented yet."
    )
