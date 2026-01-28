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

import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


@register(nn.Linear)
class QuantLinear(QuantModuleBase):
    """Per-channel weight fake-quant, eager-output activation fake-quant."""

    def __init__(
        self,
        fp: nn.Linear,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.obs_weight = self._make_obs(
            "weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")
        self.module = fp

    def enable_calibration(self) -> None:
        super().enable_calibration()
        # immediately capture the fixed weight range
        self.obs_weight.collect(self.module.weight)

    def forward(self, x):
        x_q = self._fq(x, self.obs_act_in)

        w = self.module.weight
        if self._mode is Mode.QUANT:
            w = self.obs_weight.fake_quant(w)
        b = self.module.bias

        out = F.linear(x_q, w, b)

        return self._fq(out, self.obs_act_out)

    def _all_observers(self):
        return (self.obs_weight, self.obs_act_in, self.obs_act_out)
