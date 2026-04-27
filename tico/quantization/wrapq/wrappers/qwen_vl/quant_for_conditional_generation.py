# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register

from transformers.generation.utils import GenerationMixin


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration",
)
class QuantQwen3VLForConditionalGeneration(QuantModuleBase, GenerationMixin):
    """
    Quantization wrapper for Qwen3VLForConditionalGeneration module.

    This is the full multimodal generation model that includes:
    - Vision and language model (Qwen3VLModel): Processes inputs and generates hidden states
    - Language modeling head (lm_head): Projects hidden states to vocabulary logits for generation

    The forward pass simply delegates to the wrapped model and lm_head,
    with no additional quantization operations needed at this level.
    """

    main_input_name = "input_ids"
    _is_stateful = True
    _supports_cache_class = True

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config

        # Wrap the vision-language model
        self.model = PTQWrapper(
            fp_model.model,
            qcfg=qcfg.child("model") if qcfg else None,
            fp_name=f"{fp_name}.model",
        )

        # Wrap the language modeling head
        self.lm_head = PTQWrapper(
            fp_model.lm_head,
            qcfg=qcfg.child("lm_head") if qcfg else None,
            fp_name=f"{fp_name}.lm_head",
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with fake quantization.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            position_ids: Position IDs for RoPE
            past_key_values: Past key-value caches for autoregressive generation
            inputs_embeds: Input embeddings of shape (batch_size, sequence_length, hidden_size)
            labels: Labels for computing masked language modeling loss
            pixel_values: Image pixel values of shape (batch_size, C, H, W)
            pixel_values_videos: Video pixel values
            image_grid_thw: Grid dimensions for images of shape (num_images, 3)
            video_grid_thw: Grid dimensions for videos
            cache_position: Cache positions for generation
            logits_to_keep: Number of logits to keep from the end
            **kwargs: Additional keyword arguments

        Returns:
            Model output containing logits, past key values, etc.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        # Get hidden states from the vision-language model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Return output with proper structure
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLCausalLMOutputWithPast,
        )

        if return_dict:
            output = Qwen3VLCausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )
        else:
            # (logits, past_key_values)
            output = (logits,) if len(outputs) == 1 else (logits, outputs[1])
        return output

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module and wrapped submodules."""
        # No local observers - all observers are in wrapped submodules
        # This method is required by QuantModuleBase but yields nothing
        return iter([])

    @property
    def device(self):
        """Return the device for generation."""
        return self.module.device

    @property
    def generation_config(self):
        """Return the generation config."""
        return self.module.generation_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        """Prepare inputs for generation step."""
        return self.module.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
