# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.activations.GELUTanh")
class QuantGELUTanh(QuantModuleBase):
    """
    QuantGELUTanh — drop-in quantized implementation of the Tanh-based GELUTanh activation.

    This module quantizes both intermediate tensors:
        t  = tanh(sqrt(2/π) * (x + 0.044715 * x^3))  (tanh)
        y  = x * 0.5 * (1 + t)                       (mul)

    GELUTanh formula:
        GELUTanh(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """

    def __init__(
        self,
        fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.obs_act_in = self._make_obs("act_in")
        self.obs_tanh = self._make_obs("tanh")
        self.obs_mul = self._make_obs("mul")
        self.module = fp

    def forward(self, x: torch.Tensor):
        # Quantize input
        x_q = self._fq(x, self.obs_act_in)

        # GELUTanh computation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        x3 = x_q * x_q * x_q
        inner = x_q + 0.044715 * x3
        t = torch.tanh(math.sqrt(2.0 / math.pi) * inner)

        # Quantize tanh output
        t = self._fq(t, self.obs_tanh)

        y = x_q * 0.5 * (1 + t)

        # Quantize final output
        y = self._fq(y, self.obs_mul)

        return y

    def _all_observers(self):
        return (self.obs_act_in, self.obs_tanh, self.obs_mul)
