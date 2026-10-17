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

"""Shared padded-text token helpers for the LLaMA and Qwen3-VL debug runtimes."""

from typing import cast, Optional

import torch


def normalize_valid_token_mask(
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    *,
    pad_token_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean mask where True marks real prompt tokens."""
    if attention_mask is None:
        if pad_token_id is None:
            valid = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        else:
            valid = input_ids.to(device).ne(int(pad_token_id))
    else:
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "attention_mask must have the same shape as input_ids. "
                f"Got attention_mask={tuple(attention_mask.shape)}, "
                f"input_ids={tuple(input_ids.shape)}."
            )
        valid = attention_mask.to(device=device).bool()

    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("Each batch row must contain at least one real token.")
    return valid


def validate_padding_layout(valid_token_mask: torch.Tensor, padding_side: str) -> None:
    """Validate that each batch row uses contiguous left or right padding."""
    batch_size, seq_len = valid_token_mask.shape
    valid_lengths = valid_token_mask.sum(dim=1)
    positions = torch.arange(seq_len, device=valid_token_mask.device).unsqueeze(0)

    if padding_side == "right":
        expected = positions < valid_lengths.unsqueeze(1)
    elif padding_side == "left":
        expected = positions >= (seq_len - valid_lengths).unsqueeze(1)
    else:
        raise ValueError(
            f"padding_side must be 'left' or 'right'. got {padding_side!r}"
        )

    if not torch.equal(valid_token_mask, expected):
        raise ValueError(
            "Input padding layout does not match padding_side. "
            f"Expected contiguous {padding_side} padding for shape "
            f"(B={batch_size}, S={seq_len})."
        )


def build_position_ids_from_valid_token_mask(
    valid_token_mask: torch.Tensor,
) -> torch.LongTensor:
    """Build compact text-only position IDs for padded prefill inputs."""
    position_ids = valid_token_mask.to(torch.long).cumsum(dim=1) - 1
    position_ids = torch.clamp(position_ids, min=0)
    return cast(torch.LongTensor, position_ids.masked_fill(~valid_token_mask, 0))


def last_valid_token_indices(valid_token_mask: torch.Tensor) -> torch.LongTensor:
    """Return the input index of the last real token for each batch row."""
    batch_size, seq_len = valid_token_mask.shape
    positions = torch.arange(seq_len, device=valid_token_mask.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    return cast(
        torch.LongTensor, positions.masked_fill(~valid_token_mask, 0).max(dim=1).values
    )
