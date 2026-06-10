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

from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    assert_gemma4_e2b_no_moe,
    fixed_slot_fuse,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4Model")
class QuantGemma4Model(QuantModuleBase):
    """PTQ wrapper skeleton for image-text Gemma4 E2B."""

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        assert_gemma4_e2b_no_moe(fp_model)
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config
        self.vision_tower = PTQWrapper(
            fp_model.vision_tower,
            qcfg=qcfg.child("vision_tower") if qcfg else None,
            fp_name=join_name(fp_name, "vision_tower"),
        )
        self.language_model = PTQWrapper(
            fp_model.language_model,
            qcfg=qcfg.child("language_model") if qcfg else None,
            fp_name=join_name(fp_name, "language_model"),
        )
        self.embed_vision = PTQWrapper(
            fp_model.embed_vision,
            qcfg=qcfg.child("embed_vision") if qcfg else None,
            fp_name=join_name(fp_name, "embed_vision"),
        )
        if getattr(fp_model, "audio_tower", None) is not None:
            raise NotImplementedError(
                "Gemma4 E2B skeleton does not implement audio static runtime."
            )

        self.visual_start_idx = int(
            self.qcfg.model_args.get("vision", {}).get("visual_start_idx", 0)
        )
        self.num_visual_tokens = int(
            self.qcfg.model_args.get("vision", {}).get("num_visual_tokens", 0)
        )
        self.obs_mm_fusion = self._make_obs("mm_fusion")

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: Optional[torch.Tensor] = None,
    ):
        """Return projected image soft tokens."""
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            return_dict=True,
        )
        return self.embed_vision(vision_outputs.last_hidden_state)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask=None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run Gemma4 image-text forward with fixed-slot fusion.

        TODO: Implement full HF-compatible output objects after the static layer
        wrappers are complete.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "input_ids must be provided when inputs_embeds is None."
                )
            llm_input_ids = input_ids.clone()
            # TODO: Replace image token positions with pad_token_id on CPU before calling this wrapper.
            inputs_embeds = self.language_model.wrapped.embed_tokens(llm_input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values, image_position_ids=image_position_ids
            )
            inputs_embeds = fixed_slot_fuse(
                inputs_embeds,
                image_embeds,
                visual_start_idx=self.visual_start_idx,
                num_visual_tokens=self.num_visual_tokens,
            )
            inputs_embeds = self._fq(inputs_embeds, self.obs_mm_fusion)

        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            per_layer_inputs=per_layer_inputs,
            **kwargs,
        )

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_mm_fusion,)
