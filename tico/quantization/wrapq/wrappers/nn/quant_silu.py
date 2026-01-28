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

from typing import Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("torch.nn.SiLU", "transformers.activations.SiLUActivation")
class QuantSiLU(QuantModuleBase):
    """
    QuantSiLU — drop-in quantized implementation of the SiLU operation.

    This module quantizes both intermediate tensors:
        • s  = sigmoid(x)   (logistic)
        • y  = x * s        (mul)
    """

    def __init__(
        self,
        fp: nn.SiLU,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.obs_act_in = self._make_obs("act_in")
        self.obs_sig = self._make_obs("sigmoid")
        self.obs_mul = self._make_obs("mul")
        self.module = fp

    def forward(self, x: torch.Tensor):
        x_q = self._fq(x, self.obs_act_in)

        s = torch.sigmoid(x_q)
        s = self._fq(s, self.obs_sig)

        y = x * s
        y = self._fq(y, self.obs_mul)

        return y

    def _all_observers(self):
        return (self.obs_act_in, self.obs_sig, self.obs_mul)
