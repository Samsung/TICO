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

from enum import IntEnum
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import torch

from tico.utils.errors import InvalidArgumentError


PaddingValue = Union[str, Sequence[int]]  # "same" | "valid" | [pad_h, pad_w]


class ConvPadding(IntEnum):
    SAME = 0  # auto-pad, HW out == HW in
    VALID = 1  # no implicit padding


class ConvPaddingInfo(NamedTuple):
    """
    Result of padding analysis.
    """

    conv_padding_type: ConvPadding
    explicit_pad_hw: Optional[Tuple[int, int]]  # None -> no extra Pad() op needed


def identify_padding(
    padding: PaddingValue,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    stride: Sequence[int],
) -> ConvPaddingInfo:
    """
    Normalizes all PyTorch `padding` variants to a single decision.

    Rules
    -----
    1. "valid" or [0, 0]                              → VALID, no Pad().
    2. "same" or the shapes already match (stride==1) → SAME, no Pad().
    3. Any other 2-element list                       → VALID + explicit Pad().

    TODO The following SAME padding check assumes stride == 1.
         For stride > 1, Conv2D and TransposeConv2D require different formulas
          to determine the SAME padding. Update this logic to handle general
         stride values correctly for both cases.
    """
    # ─── 1. String form ────────────────────────────────────────────────────
    if isinstance(padding, str):
        pad = padding.lower()
        if pad == "valid":
            return ConvPaddingInfo(ConvPadding.VALID, None)
        if pad == "same":
            return ConvPaddingInfo(ConvPadding.SAME, None)
        raise InvalidArgumentError(f"Unknown padding string: {padding}")

    # ─── 2. List / tuple form ─────────────────────────────────────────────
    if not (isinstance(padding, (list, tuple)) and len(padding) == 2):
        raise InvalidArgumentError(
            "Padding must be 'valid', 'same', or a [pad_h, pad_w] list"
        )

    pad_h, pad_w = padding
    # [0, 0]  → VALID
    if pad_h == 0 and pad_w == 0:
        return ConvPaddingInfo(ConvPadding.VALID, None)

    # SAME heuristic: output H/W already match input when stride is 1
    hw_in = tuple(input_shape[1:3])
    hw_out = tuple(output_shape[1:3])
    if hw_in == hw_out and stride == [1, 1]:
        return ConvPaddingInfo(ConvPadding.SAME, None)

    # Anything else = explicit symmetric padding
    return ConvPaddingInfo(ConvPadding.VALID, (pad_h, pad_w))
