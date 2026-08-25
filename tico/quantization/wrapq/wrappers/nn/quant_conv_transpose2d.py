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

"""WrapQ support for ``torch.nn.ConvTranspose2d`` modules."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_conv2d import _per_channel_weight_qscheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


@register(nn.ConvTranspose2d)
class QuantConvTranspose2d(QuantModuleBase):
    """Fake-quantize ConvTranspose2d input, per-output-channel weight, and output.

    ``ConvTranspose2d`` stores its weight as ``[in_channels, out_channels //
    groups, kH, kW]``, so the per-output-channel axis is 1.
    """

    def __init__(
        self,
        fp: nn.ConvTranspose2d,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ) -> None:
        """Create observers around one floating-point ConvTranspose2d module."""
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.obs_weight = self._make_obs(
            "weight",
            qscheme=_per_channel_weight_qscheme(self.qcfg),
            channel_axis=1,
        )
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")

    def enable_calibration(self) -> None:
        """Enable activation calibration and collect the fixed weight range."""
        super().enable_calibration()
        self.obs_weight.collect(self.module.weight.detach())

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Execute ConvTranspose2d under the current WrapQ mode."""
        input_q = self._fq(input_, self.obs_act_in)
        weight = self.module.weight
        if self._mode is Mode.QUANT:
            weight = self.obs_weight.fake_quant(weight)

        output = F.conv_transpose2d(
            input_q,
            weight,
            self.module.bias,
            self.module.stride,
            self.module.padding,
            self.module.output_padding,
            self.module.groups,
            self.module.dilation,
        )
        return self._fq(output, self.obs_act_out)

    def _all_observers(self):
        """Return observers owned directly by this wrapper."""
        return self.obs_weight, self.obs_act_in, self.obs_act_out
