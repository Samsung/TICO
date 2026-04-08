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


class _AttentionExportAdapterBase(nn.Module):
    """Base class for thin export adapters around a unified Llama attention wrapper."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def extra_repr(self) -> str:
        return f"attn={self.attn.__class__.__name__}"


class LlamaAttentionPrefillExportAdapter(_AttentionExportAdapterBase):
    """
    Export adapter for prefill attention.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.
    """

    def __init__(self, attn: nn.Module, *, return_kv: bool = True):
        super().__init__(attn)
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=None,
            output_attentions=False,
            use_cache=self.return_kv,
        )

        if self.return_kv:
            assert len(out) == 3
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "Expected attention output as (hidden_states, attn_weights, (new_k, new_v))."
                )
            return (out[0], out[2])

        assert len(out) == 2
        return out[0]


class LlamaAttentionDecodeExportAdapter(_AttentionExportAdapterBase):
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

    def __init__(self, attn: nn.Module, *, return_kv: bool = True):
        super().__init__(attn)
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
    ):
        out = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=self.return_kv,
        )

        if self.return_kv:
            assert len(out) == 3
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "Expected attention output as (hidden_states, attn_weights, (new_k, new_v))."
                )
            return (out[0], out[2])

        assert len(out) == 2
        return out[0]


class _DecoderLayerExportAdapterBase(nn.Module):
    """Base class for thin export adapters around a unified decoder layer."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.layer.return_type = "tuple"

    def extra_repr(self) -> str:
        return f"layer={self.layer.__class__.__name__}"


class LlamaDecoderLayerPrefillExportAdapter(_DecoderLayerExportAdapterBase):
    """
    Export adapter for prefill.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.
    """

    def __init__(self, layer: nn.Module, *, return_kv: bool = True):
        super().__init__(layer)
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        out = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=False,
            use_cache=self.return_kv,
        )

        if self.return_kv:
            assert len(out) == 2
            return out
        return out[0] if isinstance(out, tuple) else out


class LlamaDecoderLayerDecodeExportAdapter(_DecoderLayerExportAdapterBase):
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

    def __init__(self, layer: nn.Module, *, return_kv: bool = True):
        super().__init__(layer)
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        out = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings,
            output_attentions=False,
            use_cache=self.return_kv,
        )

        if self.return_kv:
            assert len(out) == 2
            return out
        return out[0] if isinstance(out, tuple) else out
