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

from dataclasses import dataclass

import torch

from tico.quantization.config.gptq import GPTQConfig


@dataclass
class Gemma4GPTQConfig(GPTQConfig):
    """
    Configuration for GPTQ on Gemma4 (E2B).

    This config extends the generic GPTQ configuration with Gemma4 specific
    switches so that the quantizer can process the model in stage order:

        1) vision patch embedder (Linear ``input_proj``)
        2) vision encoder blocks
        3) vision pooler  (currently a no-op — no trainable Linear weights)
        4) multimodal embedder (``embed_vision`` projection)
        5) text decoder layers
        6) lm_head (optional)

    The main purpose of this configuration is to support layerwise/stagewise
    GPTQ for Gemma4 multimodal models (Gemma4ForConditionalGeneration) and
    text-only models (Gemma4ForCausalLM).

    Gemma4 vs Qwen3-VL architectural differences that affect GPTQ:

    - **Patch embedding**: Gemma4 uses a ``Linear`` (``input_proj``) inside
      ``patch_embedder``; Qwen3-VL uses a ``Conv3d``.
    - **Vision merger**: Gemma4 has a ``pooler`` (spatial pooling, no Linear
      weights) and a separate ``embed_vision`` multimodal embedder (Linear
      ``embedding_projection``); Qwen3-VL has a ``merger`` and
      ``deepstack_merger_list``.
    - **Deepstack**: Qwen3-VL injects deepstack visual embeddings into early
      text decoder layers and replays ``_deepstack_process`` after each layer.
      Gemma4 has no deepstack; instead it uses Per-Layer Embeddings (PLE) which
      are handled internally by each text decoder layer.
    - **Text layer re-forward**: Gemma4 text decoder layers return plain
      ``hidden_states`` (when ``use_cache=False``) and need no special
      post-processing; Qwen3-VL requires ``_deepstack_process``.
    """

    # ------------------------------------------------------------------
    # Model identity
    # ------------------------------------------------------------------
    model_type: str = "gemma4"

    # ------------------------------------------------------------------
    # Stage-level enable/disable switches
    # ------------------------------------------------------------------
    quantize_vision: bool = True
    quantize_text: bool = True
    quantize_lm_head: bool = False

    # ------------------------------------------------------------------
    # Vision-side stage switches
    # ------------------------------------------------------------------
    quantize_vision_patch_embed: bool = True
    quantize_vision_blocks: bool = True
    quantize_vision_pooler: bool = True
    quantize_multimodal_embedder: bool = True

    # ------------------------------------------------------------------
    # Text-side stage switches
    # ------------------------------------------------------------------
    quantize_text_layers: bool = True

    # ------------------------------------------------------------------
    # Cache behavior
    # ------------------------------------------------------------------
    move_cache_to_cpu: bool = False
    cache_dtype: torch.dtype | None = None

    # ------------------------------------------------------------------
    # Optional attribute paths for architecture lookup
    # These defaults follow the Gemma4ForConditionalGeneration HF structure.
    #
    # For Gemma4ForCausalLM (text-only), override:
    #   language_model_attr = "model"
    #   text_layers_attr     = "model.layers"
    #   lm_head_attr          = "lm_head"
    #   and set quantize_vision = False
    # ------------------------------------------------------------------
    vision_tower_attr: str = "model.vision_tower"
    vision_patch_embed_attr: str = "model.vision_tower.patch_embedder"
    vision_encoder_attr: str = "model.vision_tower.encoder"
    vision_encoder_layers_attr: str = "model.vision_tower.encoder.layers"
    vision_pooler_attr: str = "model.vision_tower.pooler"
    multimodal_embedder_attr: str = "model.embed_vision"

    language_model_attr: str = "model.language_model"
    text_layers_attr: str = "model.language_model.layers"
    lm_head_attr: str = "lm_head"

    @property
    def name(self) -> str:
        return "gemma4_gptq"

    def validate(self) -> None:
        """
        Validate Gemma4 specific GPTQ settings.

        Raises:
            ValueError: If a numeric or logical option is invalid.
            TypeError: If a field has an unexpected type.
        """
        super().validate()

        if self.model_type != "gemma4":
            raise ValueError(f"model_type must be 'gemma4'. got {self.model_type!r}")

        if not isinstance(self.quantize_lm_head, bool):
            raise TypeError(
                f"quantize_lm_head must be bool. got {type(self.quantize_lm_head)}"
            )

        if not (self.quantize_vision or self.quantize_text or self.quantize_lm_head):
            raise ValueError(
                "At least one of quantize_vision, quantize_text, or "
                "quantize_lm_head must be True."
            )

        if not self.quantize_vision:
            if self.quantize_vision_patch_embed:
                raise ValueError(
                    "quantize_vision_patch_embed=True requires quantize_vision=True."
                )
            if self.quantize_vision_blocks:
                raise ValueError(
                    "quantize_vision_blocks=True requires quantize_vision=True."
                )
            if self.quantize_vision_pooler:
                raise ValueError(
                    "quantize_vision_pooler=True requires quantize_vision=True."
                )
            if self.quantize_multimodal_embedder:
                raise ValueError(
                    "quantize_multimodal_embedder=True requires "
                    "quantize_vision=True."
                )

        if not self.quantize_text and self.quantize_text_layers:
            raise ValueError("quantize_text_layers=True requires quantize_text=True.")

        if self.cache_dtype is not None and not isinstance(
            self.cache_dtype, torch.dtype
        ):
            raise TypeError(
                f"cache_dtype must be torch.dtype or None. got {type(self.cache_dtype)}"
            )

        attr_fields = {
            "vision_tower_attr": self.vision_tower_attr,
            "vision_patch_embed_attr": self.vision_patch_embed_attr,
            "vision_encoder_attr": self.vision_encoder_attr,
            "vision_encoder_layers_attr": self.vision_encoder_layers_attr,
            "vision_pooler_attr": self.vision_pooler_attr,
            "multimodal_embedder_attr": self.multimodal_embedder_attr,
            "language_model_attr": self.language_model_attr,
            "text_layers_attr": self.text_layers_attr,
            "lm_head_attr": self.lm_head_attr,
        }

        for field_name, field_value in attr_fields.items():
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(
                    f"{field_name} must be a non-empty string. got {field_value!r}"
                )
