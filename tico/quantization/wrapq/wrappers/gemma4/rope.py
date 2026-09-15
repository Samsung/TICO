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

"""Host-side sine-table preparation for Gemma4 text and assistant attention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tico.quantization.config.gemma4_attention import RopeConvention


def prepare_gemma4_rope_sin(
    sin: torch.Tensor, rope: RopeConvention = "hf"
) -> torch.Tensor:
    """Convert a fresh HF sine table to the consuming attention convention.

    For ``pre_negated_sin``, negate the first half of the LAST dimension.
    This is equivalent to moving the negation out of ``[-x2, x1]`` and into
    the table. It works for both full and proportional/partial RoPE, whose
    non-rotating positions have zero sine. Cosine, frequencies, dtype, device,
    and shape are unchanged. The input is never modified in place.

    Apply this exactly once, before sine observation/fake quantization. The
    operation is NOT idempotent. Cached templates already prepared with this
    function must only be sliced, and shared or cached K/V must not be passed
    here. The HF convention returns the original tensor without allocating.
    """
    if rope not in ("hf", "pre_negated_sin"):
        raise ValueError(f"Unsupported RoPE convention: {rope!r}.")
    if sin.dim() == 0 or sin.size(-1) == 0 or sin.size(-1) % 2:
        raise ValueError(
            "Gemma4 sine tables require a positive, even last dimension, "
            f"got shape={tuple(sin.shape)}."
        )
    if not torch.is_floating_point(sin):
        raise TypeError(f"Gemma4 sine tables must be floating point, got {sin.dtype}.")
    if rope == "hf":
        return sin
    first, second = sin.chunk(2, dim=-1)
    return torch.cat((-first, second), dim=-1)
