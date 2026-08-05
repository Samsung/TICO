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

"""WrapQ support for ``torch.nn.PReLU`` modules."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


def _per_channel_weight_qscheme(qcfg: PTQConfig) -> QScheme:
    """Return a per-channel weight qscheme with the configured symmetry."""
    role = qcfg.get_role_kwargs("weight")
    qscheme = role.get("qscheme")
    if qscheme is not None:
        return (
            QScheme.PER_CHANNEL_SYMM
            if qscheme.is_symmetric()
            else QScheme.PER_CHANNEL_ASYMM
        )

    dtype = role.get("dtype")
    if dtype is not None and dtype.signed:
        return QScheme.PER_CHANNEL_SYMM
    return QScheme.PER_CHANNEL_ASYMM


@register(nn.PReLU)
class QuantPReLU(QuantModuleBase):
    """Fake-quantize PReLU input, per-channel slope, and output tensors."""

    def __init__(
        self,
        fp: nn.PReLU,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ) -> None:
        """Create observers around one floating-point PReLU module."""
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.obs_weight = self._make_obs(
            "weight",
            qscheme=_per_channel_weight_qscheme(self.qcfg),
            channel_axis=0,
        )
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")

    def enable_calibration(self) -> None:
        """Enable activation calibration and collect every fixed slope value."""
        super().enable_calibration()
        self.obs_weight.collect(self.module.weight.detach())

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Execute PReLU under the current WrapQ mode."""
        input_q = self._fq(input_, self.obs_act_in)
        weight = self.module.weight
        if self._mode is Mode.QUANT:
            weight = self.obs_weight.fake_quant(weight)
        output = F.prelu(input_q, weight)
        return self._fq(output, self.obs_act_out)

    def _all_observers(self):
        """Return observers owned directly by this wrapper."""
        return self.obs_weight, self.obs_act_in, self.obs_act_out
