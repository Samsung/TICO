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

import copy
from typing import Any

import torch
from torch import nn


def build_causal_mask(
    *,
    q_len: int,
    k_len: int,
    past_len: int,
    device: torch.device,
    fill_value: float,
) -> torch.Tensor:
    """Build a [1, Q, K] mask for the current input, without a context-sized cache."""
    query_positions = torch.arange(q_len, device=device) + past_len
    key_positions = torch.arange(k_len, device=device)
    visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return (
        torch.zeros(q_len, k_len, device=device, dtype=torch.float32)
        .masked_fill(~visible, fill_value)
        .unsqueeze(0)
    )


def combine_binary_attention_mask(
    causal_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    fill_value: float,
) -> torch.Tensor:
    """Combine a causal mask with a binary token or query/key mask."""
    valid = attention_mask != 0
    if valid.dim() == 2:
        valid = valid.unsqueeze(1)
    elif valid.dim() == 4 and valid.size(1) == 1:
        valid = valid.squeeze(1)
    additive = torch.zeros_like(valid, dtype=torch.float32).masked_fill(
        ~valid, fill_value
    )
    return torch.clamp(causal_mask + additive, min=fill_value)


def make_rotary_embedding(config: Any, source: nn.Module | None = None) -> nn.Module:
    """Retain the source rotary implementation or reconstruct it from its config."""
    if source is not None:
        return copy.deepcopy(source)

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    return LlamaRotaryEmbedding(config=config)


def position_embeddings(
    owner: nn.Module,
    config: Any,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    pre_negated_sin: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate pretrained RoPE only at the requested logical token positions.

    Older pickled wrappers have no rotary module. Reconstructing it uses their
    recorded configuration, including its original context limit; it does not
    recover a larger native limit from an already-clamped checkpoint.
    """
    rotary = getattr(owner, "rotary_emb", None)
    if rotary is None:
        rotary = make_rotary_embedding(config).to(hidden_states.device)
        owner.add_module("rotary_emb", rotary)
    cos, sin = rotary(hidden_states, position_ids)
    if pre_negated_sin:
        half = sin.shape[-1] // 2
        sin = torch.cat((-sin[..., :half], sin[..., half:]), dim=-1)
    return cos, sin
