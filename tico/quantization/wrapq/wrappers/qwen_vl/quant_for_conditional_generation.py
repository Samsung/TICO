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

from typing import cast, Iterable, Optional, TYPE_CHECKING, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers._typing import GenerativePreTrainedModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLCausalLMOutputWithPast,
        Qwen3VLForConditionalGeneration,
    )


from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register

from transformers.generation.utils import GenerationMixin


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration",
    "tico.quantization.algorithm.spinquant.spin_qwen3_vl.SpinQwen3VLForConditionalGeneration",
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
        fp_model = cast("Qwen3VLForConditionalGeneration", fp_model)
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config

        # Wrap the vision-language model
        self.model = PTQWrapper(
            fp_model.model,
            qcfg=qcfg.child("model") if qcfg else None,
            fp_name=join_name(fp_name, "model"),
        )

        # Wrap the language modeling head
        self.lm_head = PTQWrapper(
            fp_model.lm_head,
            qcfg=qcfg.child("lm_head") if qcfg else None,
            fp_name=join_name(fp_name, "lm_head"),
        )

        rotate_lm_head_cfg = qcfg.child("rotate_lm_head") if qcfg else None
        # `rotate_lm_head` exists only for SpinQuant-style custom models.
        # For a standard model, skip creating the wrapper and
        # bypass it during forward.
        self.rotate_lm_head = None
        if hasattr(fp_model, "rotate_lm_head") and isinstance(
            fp_model.rotate_lm_head, torch.nn.Module
        ):
            self.rotate_lm_head = PTQWrapper(
                fp_model.rotate_lm_head,
                rotate_lm_head_cfg,
                fp_name=join_name(fp_name, "rotate_lm_head"),
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
        mm_token_type_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Union["Qwen3VLCausalLMOutputWithPast", tuple]:
        """
        Forward pass with fake quantization.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            position_ids: Position IDs for RoPE
            past_key_values: Past key-value caches for autoregressive generation
            inputs_embeds: Input embeddings of shape (batch_size, sequence_length, hidden_size)
            labels: Labels for computing masked language modeling loss
            pixel_values: Flattened image patches with shape
                (num_patches, patch_dim) or (1, num_patches, patch_dim).
            pixel_values_videos: Flattened video patches with the same contract.
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
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        if self.rotate_lm_head is not None:
            hidden_states = self.rotate_lm_head(hidden_states)

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

        loss = None
        if labels is not None:
            loss = self.module.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=cast(
                    "Qwen3VLTextConfig", self.module.config.text_config
                ).vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        output = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

        return output

    def _all_observers(self) -> Iterable:
        """This wrapper owns no observers directly."""
        return ()

    @property
    def device(self):
        """Return the device for generation."""
        return self.module.device

    @property
    def generation_config(self):
        """Return the generation config."""
        return self.module.generation_config

    @generation_config.setter
    def generation_config(self, value):
        """Update the wrapped model's generation configuration."""
        self.module.generation_config = value

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length: int | None = None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration: bool | None = False,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        """Prepare inputs for generation step."""
        return cast(
            "GenerativePreTrainedModel", self.module
        ).prepare_inputs_for_generation(
            input_ids=input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """Delegate 3D mrope position-id preparation to the wrapped model.

        ``GenerationMixin.generate`` calls this hook on ``self``. The generic
        implementation returns 1D sequential ids which would bypass the 3D
        mrope computation in the vision-language wrapper; the wrapped
        Qwen3-VL model computes the proper [4, batch, seq] layout instead.
        """
        return self.module._prepare_position_ids_for_generation(  # type: ignore[operator]
            inputs_tensor, model_kwargs
        )

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids=None,
        **model_kwargs,
    ):
        """Use the expansion logic matching Qwen3-VL position preparation."""
        return cast(
            "GenerativePreTrainedModel", self.module
        )._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )

    def tie_weights(self):
        pass
