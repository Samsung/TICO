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

"""Explicit activation boundary for TICO quantization workflows."""

from __future__ import annotations

import torch
from torch import nn


class QuantStub(nn.Module):
    """Act as identity in FP32 and as a WrapQ boundary after preparation."""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged in the floating-point model."""
        return input_
