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

"""Module facade for a quantizable concatenation boundary."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Concat(nn.Module):
    """Concatenate tensors along a fixed dimension."""

    def __init__(
        self,
        dim: int = 0,
        *,
        allow_distinct_input_qparams: bool = False,
    ) -> None:
        """Store the axis and the quantization calibration policy.

        ``allow_distinct_input_qparams`` is a WrapQ calibration policy, not an
        export contract. When it is false, ``QuantConcat`` fake-quantizes every
        input with the shared output observer, so all branches are observed and
        rounded in one domain. When it is true, each input keeps the domain of
        its own producer during calibration and fake-quantized evaluation. In
        both cases the exported Circle CONCATENATION uses one domain on all of
        its inputs and its output; any difference is converted by an explicit
        QUANTIZE inserted during conversion.
        """
        super().__init__()
        self.dim = int(dim)
        self.allow_distinct_input_qparams = bool(allow_distinct_input_qparams)

    def forward(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concatenate a non-empty tensor sequence."""
        values = tuple(tensors)
        if not values:
            raise ValueError("Concat requires at least one input tensor.")
        return torch.cat(values, dim=self.dim)

    def extra_repr(self) -> str:
        """Return the fixed dimension and nondefault calibration policy."""
        fields = [f"dim={self.dim}"]
        if self.allow_distinct_input_qparams:
            fields.append("allow_distinct_input_qparams=True")
        return ", ".join(fields)
