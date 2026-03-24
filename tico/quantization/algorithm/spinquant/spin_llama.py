# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class SpinLlamaPreTrainedModel(PreTrainedModel):
    """
    Base pretrained model class for SpinLlama models.
    """

    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


class SpinLlamaModel(SpinLlamaPreTrainedModel):
    """
    LLaMA backbone extended with a learnable rotation layer before the decoder stack.
    """

    def __init__(self, config: LlamaConfig):
        """
        Initialize the SpinLlama backbone.

        Parameters:
            config: The model configuration.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.rotate_embedding = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        Run the forward pass of the SpinLlama backbone.

        Parameters:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            past_key_values: Cache object for autoregressive decoding.
            inputs_embeds: Optional embedded inputs.
            cache_position: Cache positions for decoding.
            use_cache: Whether to use KV cache.
            **kwargs: Extra decoder-layer arguments.

        Returns:
            A BaseModelOutputWithPast containing the final hidden states and cache.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Exactly one of `input_ids` or `inputs_embeds` must be provided."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        # Rotate
        hidden_states = self.rotate_embedding(hidden_states)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class SpinLlamaForCausalLM(SpinLlamaPreTrainedModel, GenerationMixin):
    """
    Causal language modeling head for SpinLlama.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: LlamaConfig):
        """
        Initialize the SpinLlama causal language model.

        Parameters:
            config: The model configuration.
        """
        super().__init__(config)
        self.model = SpinLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rotate_lm_head = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """
        Return the input embedding layer.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        """
        Set the input embedding layer.

        Parameters:
            value: The embedding module to set.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        """
        Return the output projection layer.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """
        Set the output projection layer.

        Parameters:
            new_embeddings: The output embedding module to set.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Module) -> None:
        """
        Set the decoder backbone.

        Parameters:
            decoder: The decoder module to set.
        """
        self.model = decoder

    def get_decoder(self) -> nn.Module:
        """
        Return the decoder backbone.
        """
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Run the forward pass of SpinLlama for causal language modeling.

        Parameters:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            past_key_values: Cache object for autoregressive decoding.
            inputs_embeds: Optional embedded inputs.
            labels: Optional labels for loss computation.
            use_cache: Whether to use KV cache.
            cache_position: Cache positions for decoding.
            logits_to_keep: Controls which token positions produce logits.
            **kwargs: Extra arguments forwarded to the backbone.

        Returns:
            A CausalLMOutputWithPast containing loss, logits, and cache.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Rotate
        hidden_states = self.rotate_lm_head(hidden_states)

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
