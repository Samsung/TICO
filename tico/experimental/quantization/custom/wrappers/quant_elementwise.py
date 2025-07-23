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

from typing import Callable

import torch
import torch.nn as nn

from tico.experimental.quantization.custom.quant_config import QuantConfig
from tico.experimental.quantization.custom.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.custom.wrappers.registry import register


class QuantElementwise(QuantModuleBase):
    """
    Generic wrapper for *any* 1-to-1 element-wise op `y = f(x)`.

    Sub-classes only need to implement:
        • `FUNC`: a Callable that maps tensor→tensor
    """

    # subclass must set this
    FUNC: Callable[[torch.Tensor], torch.Tensor] = staticmethod(lambda x: x)

    def __init__(self, fp_module: nn.Module, *, qcfg: QuantConfig | None = None):
        super().__init__(qcfg)
        self.module = fp_module  # keep original for export
        self.act_in_obs = self._make_obs("act_in")
        self.act_out_obs = self._make_obs("act_out")

    # ------------------------------------------------------------
    def forward(self, x):
        x_q = self._fq(x, self.act_in_obs)
        y = self.FUNC(x_q)  # element-wise op
        y_q = self._fq(y, self.act_out_obs)
        return y_q

    # ------------------------------------------------------------
    def _all_observers(self):
        return (self.act_in_obs, self.act_out_obs)


# Sigmoid
@register(nn.Sigmoid)
class QuantSigmoid(QuantElementwise):
    FUNC = staticmethod(torch.sigmoid)


# Tanh
@register(nn.Tanh)
class QuantTanh(QuantElementwise):
    FUNC = staticmethod(torch.tanh)


# ReLU
@register(nn.ReLU)
class QuantReLU(QuantElementwise):
    FUNC = staticmethod(torch.relu)


# GELU (approximate)
@register(nn.GELU)
class QuantGELU(QuantElementwise):
    FUNC = staticmethod(torch.nn.functional.gelu)
