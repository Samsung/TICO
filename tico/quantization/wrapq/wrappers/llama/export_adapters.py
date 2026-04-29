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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from transformers.cache_utils import Cache


class LlamaAttentionPrefillExportAdapter(nn.Module):
    """
    Export adapter for prefill attention.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.
    """

    def __init__(self, wrapped, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Run prefill attention with export-friendly cache semantics.

        When `return_kv=True`, the wrapped attention module is asked to return
        only the newly produced K/V tensors instead of the full present cache.
        """
        outputs = self.wrapped(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[2]
        return hidden, new_k, new_v


class LlamaAttentionDecodeExportAdapter(nn.Module):
    """
    Export adapter for decode attention.

    Input contract
    --------------
    - hidden_states:        (B, 1, D)
    - position_embeddings:  (B, 1, head_dim)
    - attention_mask:       (B, 1, K)
    - past_key_value:       (B, num_kv_heads, K - 1, head_dim)


    Return contract
    ---------------
    - return_kv=True:
        (hidden_states, (new_key, new_value))
      where new_key/new_value are the KV delta for the current token.

    - return_kv=False:
        hidden_states
    """

    def __init__(self, wrapped, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Run decode attention with delta-only cache output.

        The wrapped attention module still builds the full present K/V for
        attention computation, but only the newly produced K/V tensors are
        returned to the caller.
        """
        outputs = self.wrapped(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[2]
        return hidden, new_k, new_v


class LlamaDecoderLayerPrefillExportAdapter(nn.Module):
    """
    Export adapter for prefill.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.
    """

    def __init__(self, wrapped, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ):
        """
        Run the decoder layer prefill path and return delta-only K/V tensors
        when caching is enabled.
        """
        outputs = self.wrapped(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[1]
        return hidden, new_k, new_v


class LlamaDecoderLayerDecodeExportAdapter(nn.Module):
    """
    Export adapter for decode.

    Input contract
    --------------
    - hidden_states:     (B, 1, D)
    - attention_mask:    additive mask, typically (B, 1, K)
    - past_key / past_value:
                        (B, num_kv_heads, K-1, head_dim)
    - cos / sin:         (B, 1, head_dim)

    Return contract
    ---------------
    - return_kv=True:
        (hidden_states, new_key, new_value)
      where new_key/new_value are the delta KV for the current token.

    - return_kv=False:
        hidden_states
    """

    def __init__(self, wrapped, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ):
        """
        Run the decoder layer decode path and return delta-only K/V tensors
        when caching is enabled.
        """
        outputs = self.wrapped(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[1]
        return hidden, new_k, new_v


class QuantLlamaForCausalLMPrefillExportAdapter(nn.Module):
    """
    Export adapter for prefill mode of QuantLLamaForCausalLM.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.

    Input contract
    --------------
    - input_ids: (B, S) input token ids

    Return contract
    ---------------
    - return_kv=True:
        (logits, past_key_values) where logits have shape (B, S) and
        past_key_values is a list of KV tensors for the next token generation.
    - return_kv=False:
        logits ~ (B, S)
    """

    def __init__(self, wrapped, *, return_kv: bool = False):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] | None = (
            None if self.return_kv is False else []
        )
        outputs = self.wrapped(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=self.return_kv,
            return_dict=True,
            **kwargs,
        )

        logits = outputs.logits
        if self.return_kv is False:
            return logits

        past_key_values = outputs.past_key_values
        return (logits, past_key_values)


class QuantLlamaForCausalLMDecodeExportAdapter(nn.Module):
    """
    Export adapter for decode mode of QuantLLamaForCausalLM.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.

    Input contract
    --------------
    - input_ids: (B, S) input token ids
    - past_key_values: optional list of past key/value tensors, each of shape
      (B, num_kv_heads, S-1, head_dim) when provided.

    Return contract
    ---------------
    - Returns a tuple (logits, new_key_values) where logits have (B, S)-shape and
      new_key_values is a list of KV tensors for the next token.
    """

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ):
        outputs = self.wrapped(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            cache_output_mode="delta",
            **kwargs,
        )

        logits = outputs.logits
        new_key_values = outputs.past_key_values

        return (logits, new_key_values)
