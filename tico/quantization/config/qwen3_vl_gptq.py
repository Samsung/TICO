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
class Qwen3VLGPTQConfig(GPTQConfig):
    """
    Configuration for GPTQ on Qwen3-VL.

    This config extends the generic GPTQ configuration with Qwen3-VL specific
    switches so that the quantizer can process the model in stage order:

        1) vision patch embed
        2) vision blocks
        3) vision merger / deepstack mergers
        4) text decoder layers
        5) lm_head (optional)

    The main purpose of this configuration is to support layerwise/stagewise
    GPTQ for Qwen3-VL.
    """

    # ------------------------------------------------------------------
    # Model identity
    # ------------------------------------------------------------------
    model_type: str = "qwen3_vl"

    # ------------------------------------------------------------------
    # Stage-level enable/disable switches
    # ------------------------------------------------------------------
    quantize_vision: bool = True
    quantize_text: bool = True
    quantize_lm_head: bool = True

    # ------------------------------------------------------------------
    # Vision-side stage switches
    # ------------------------------------------------------------------
    quantize_vision_patch_embed: bool = True
    quantize_vision_blocks: bool = True
    quantize_vision_merger: bool = True
    quantize_vision_deepstack_mergers: bool = True

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
    # These defaults follow the current Qwen3-VL HF structure.
    # ------------------------------------------------------------------
    visual_attr: str = "model.visual"
    visual_blocks_attr: str = "model.visual.blocks"
    visual_patch_embed_attr: str = "model.visual.patch_embed.proj"
    visual_merger_attr: str = "model.visual.merger"
    visual_deepstack_mergers_attr: str = "model.visual.deepstack_merger_list"

    language_model_attr: str = "model.language_model"
    text_layers_attr: str = "model.language_model.layers"
    lm_head_attr: str = "lm_head"

    @property
    def name(self) -> str:
        return "qwen3_vl_gptq"

    def validate(self) -> None:
        """
        Validate Qwen3-VL specific GPTQ settings.

        Raises:
            ValueError: If a numeric or logical option is invalid.
            TypeError: If a field has an unexpected type.
        """
        super().validate()

        if self.model_type != "qwen3_vl":
            raise ValueError(f"model_type must be 'qwen3_vl'. got {self.model_type!r}")

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
            if self.quantize_vision_merger:
                raise ValueError(
                    "quantize_vision_merger=True requires quantize_vision=True."
                )
            if self.quantize_vision_deepstack_mergers:
                raise ValueError(
                    "quantize_vision_deepstack_mergers=True requires "
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
            "visual_attr": self.visual_attr,
            "visual_blocks_attr": self.visual_blocks_attr,
            "visual_patch_embed_attr": self.visual_patch_embed_attr,
            "visual_merger_attr": self.visual_merger_attr,
            "visual_deepstack_mergers_attr": self.visual_deepstack_mergers_attr,
            "language_model_attr": self.language_model_attr,
            "text_layers_attr": self.text_layers_attr,
            "lm_head_attr": self.lm_head_attr,
        }

        for field_name, field_value in attr_fields.items():
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(
                    f"{field_name} must be a non-empty string. got {field_value!r}"
                )
