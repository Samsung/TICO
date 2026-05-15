# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLCausalLMOutputWithPast,
    )
except ImportError:
    Qwen3VLCausalLMOutputWithPast = CausalLMOutputWithPast


def _build_causal_lm_output(**kwargs: Any) -> Any:
    """
    Build a Qwen3-VL causal LM output while tolerating transformer version drift.

    Parameters:
        **kwargs: Output fields.

    Returns:
        A causal LM output object.
    """
    try:
        return Qwen3VLCausalLMOutputWithPast(**kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("rope_deltas", None)
        return CausalLMOutputWithPast(**fallback_kwargs)


class SpinQwen3VLTextModel(Qwen3VLTextModel):
    """
    Qwen3-VL text model extended with an input-side SpinQuant rotation.

    The added ``rotate_embedding`` layer rotates the complete input embedding
    tensor after text and main visual embeddings are already resolved. This
    preserves tied word embeddings because the shared embedding table itself is
    not modified.
    """

    def __init__(self, config):
        """
        Initialize the SpinQwen3VL text model.

        Parameters:
            config: Qwen3-VL text configuration.
        """
        super().__init__(config)
        self.rotate_embedding = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Run the text forward pass with an input-side SpinQuant rotation.

        Parameters:
            input_ids: Optional token IDs.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            past_key_values: Optional KV cache.
            inputs_embeds: Optional precomputed embeddings.
            use_cache: Whether to use the KV cache.
            visual_pos_masks: Optional masks for DeepStack visual positions.
            deepstack_visual_embeds: Optional DeepStack visual embeddings.
            **kwargs: Additional arguments forwarded to the base text model.

        Returns:
            The output of the base Qwen3-VL text model.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "Exactly one of `input_ids` or `inputs_embeds` must be provided."
                )
            inputs_embeds = self.embed_tokens(input_ids)
        elif input_ids is not None:
            raise ValueError(
                "Exactly one of `input_ids` or `inputs_embeds` must be provided."
            )

        inputs_embeds = self.rotate_embedding(inputs_embeds)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )


class SpinQwen3VLModel(Qwen3VLModel):
    """
    Qwen3-VL multimodal model that uses SpinQwen3VLTextModel.

    The vision tower is explicitly initialized from ``config.vision_config`` and
    the text tower is explicitly initialized from ``config.text_config``. This
    avoids accidentally constructing a vision wrapper with the top-level
    Qwen3VLConfig, which would break PTQ wrapping.
    """

    def __init__(self, config):
        """
        Initialize the SpinQwen3VL multimodal model.

        Parameters:
            config: Qwen3-VL configuration.
        """
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = SpinQwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class SpinQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    Qwen3-VL conditional generation model extended with SpinQuant boundaries.

    This model preserves tied word embeddings by leaving ``embed_tokens`` and
    ``lm_head`` weights unchanged. The input-side rotation is stored in
    ``model.language_model.rotate_embedding`` and the output-side correction is
    stored in ``rotate_lm_head``.
    """

    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config):
        """
        Initialize the SpinQwen3VL conditional generation model.

        Parameters:
            config: Qwen3-VL configuration.
        """
        super().__init__(config)
        self.model = SpinQwen3VLModel(config)

        hidden_size = int(config.text_config.hidden_size)
        vocab_size = int(config.text_config.vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.rotate_lm_head = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        """
        Run the conditional generation forward pass with an output-side rotation.

        Parameters:
            input_ids: Optional token IDs.
            pixel_values: Optional image pixel values.
            image_grid_thw: Optional image temporal-height-width grid.
            pixel_values_videos: Optional video pixel values.
            video_grid_thw: Optional video temporal-height-width grid.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            past_key_values: Optional KV cache.
            inputs_embeds: Optional precomputed embeddings.
            labels: Optional language modeling labels.
            use_cache: Whether to use the KV cache.
            cache_position: Optional cache positions.
            output_attentions: Whether to return attentions.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return a ModelOutput.
            logits_to_keep: Token positions to keep for logits.
            **kwargs: Additional arguments forwarded to the Qwen3-VL model.

        Returns:
            A causal LM output containing logits and optional loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        model_kwargs = dict(kwargs)
        optional_kwargs = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        for key, value in optional_kwargs.items():
            if value is not None:
                model_kwargs[key] = value

        outputs = self.model(input_ids=input_ids, **model_kwargs)

        hidden_states = outputs[0]
        hidden_states = self.rotate_lm_head(hidden_states)

        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep

        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=int(self.config.text_config.vocab_size),
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return _build_causal_lm_output(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )
