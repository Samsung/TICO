# type: ignore

# Copyright 2018- The Hugging Face team. All rights reserved.
# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import Optional, Tuple, Union

import torch
from transformers import AutoModelForImageTextToText
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, ModelOutput


def Idefics3VisionTransformer_forward(
    self,
    pixel_values,
    patch_attention_mask: Optional[torch.BoolTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutput]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    batch_size = pixel_values.size(0)
    if patch_attention_mask is None:
        patch_size = self.patch_size
        patch_attention_mask = torch.ones(
            (
                batch_size,
                pixel_values.size(2) // patch_size,
                pixel_values.size(3) // patch_size,
            )
        )
        patch_attention_mask = patch_attention_mask.to(
            dtype=torch.bool, device=pixel_values.device
        )

    hidden_states = self.embeddings(
        pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
    )

    patch_attention_mask = patch_attention_mask.view(batch_size, -1)

    # The call to `_upad_input` in `_flash_attention_forward` is expensive
    # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
    # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
    #
    # [ORIGINAL CODE]
    #
    # ```py
    # if not torch.any(~patch_attention_mask):
    #     patch_attention_mask = None
    # elif not self._use_flash_attention_2:
    #     patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
    # ```

    patch_attention_mask = None

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=patch_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.post_layernorm(last_hidden_state)

    if not return_dict:
        return (last_hidden_state,) + encoder_outputs[1:]

    return BaseModelOutput(
        last_hidden_state=last_hidden_state,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def Idefics3VisionEmbeddings_forward(
    self,
    pixel_values: torch.FloatTensor,
    patch_attention_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    batch_size, _, max_im_h, max_im_w = pixel_values.shape

    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)

    # The commented codes generate position_ids based on the size of pixel_values. The codes dynamically determine the shape of tensors, which is difficult to implement efficiently with NPUs. So, we modify the code to generate position_ids with static shapes. Note that this code works only for specific input sizes. If input size is changed, we would have to change this code too.
    #
    # [ORIGINAL CODE]
    #
    # ```py
    # max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
    # boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
    # position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

    # for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
    #     nb_patches_h = p_attn_mask[:, 0].sum()
    #     nb_patches_w = p_attn_mask[0].sum()

    #     fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
    #     fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

    #     bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    #     bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

    #     pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
    #     position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
    # ```

    position_ids = torch.arange(0, 1024, 1)
    # 26 must match with the batch size.
    position_ids = position_ids[None, :].repeat(26, 1)
    position_ids = position_ids.to(self.position_embedding.weight.device)
    embeddings = embeddings + self.position_embedding(position_ids)
    return embeddings


class SmolVLM_vision_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rtol = 1e-2
        self.atol = 1e-3
        self.model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct"
        ).model.vision_model.to("cpu")
        self.model.embeddings.forward = Idefics3VisionEmbeddings_forward.__get__(
            self.model.embeddings, type(self.model.embeddings).__class__
        )
        self.model.forward = Idefics3VisionTransformer_forward.__get__(
            self.model, type(self.model).__class__
        )

    def forward(self, *x):
        return self.model(*x)

    def get_example_inputs(self):
        pixel_values = torch.randn(26, 3, 512, 512)
        return (pixel_values,)
