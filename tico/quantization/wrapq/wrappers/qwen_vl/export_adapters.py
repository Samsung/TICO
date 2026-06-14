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

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Qwen3VLTextDecoderLayerPrefillExportAdapter(nn.Module):
    """
    Export adapter for the Qwen3-VL text decoder-layer prefill path.

    Input contract:
        hidden_states:
            Tensor with shape `(B, S, hidden_size)`.
        attention_mask:
            Additive mask with shape broadcastable to `(B, 1, S, S)`.
        position_embeddings:
            Tuple `(cos, sin)` where each tensor has shape `(B, S, head_dim)`.

    Return contract when `return_kv=True`:
        `(hidden_states, new_key, new_value)`, where:
            hidden_states has shape `(B, S, hidden_size)`;
            new_key and new_value have shape `(B, num_kv_heads, S, head_dim)`.

    Return contract when `return_kv=False`:
        `hidden_states`.
    """

    def __init__(self, wrapped: nn.Module, *, return_kv: bool = True):
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
        """Run prefill and optionally return the newly produced KV tensors."""
        outputs = self.wrapped(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[1]
        return hidden, new_k, new_v


class Qwen3VLTextDecoderLayerDecodeExportAdapter(nn.Module):
    """
    Export adapter for the Qwen3-VL text decoder-layer decode path.

    Input contract:
        hidden_states:
            Tensor with shape `(B, 1, hidden_size)`.
        attention_mask:
            Additive mask with shape broadcastable to `(B, 1, 1, K)`.
        position_embeddings:
            Tuple `(cos, sin)` where each tensor has shape `(B, 1, head_dim)`.
        past_key_values:
            Tuple `(past_key, past_value)` where each tensor has shape
            `(B, num_kv_heads, K - 1, head_dim)`.

    Return contract when `return_kv=True`:
        `(hidden_states, new_key, new_value)`, where new_key and new_value are
        the KV delta for the current token with shape
        `(B, num_kv_heads, 1, head_dim)`.

    Return contract when `return_kv=False`:
        `hidden_states`.
    """

    def __init__(self, wrapped: nn.Module, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ):
        """Run decode and optionally return the current-token KV delta."""
        outputs = self.wrapped(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[1]
        return hidden, new_k, new_v
