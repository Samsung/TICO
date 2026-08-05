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

"""WrapQ support for ``torch.nn.MaxPool2d`` modules."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


@register(nn.MaxPool2d)
class QuantMaxPool2d(QuantModuleBase):
    """Use one shared activation observer on both sides of MaxPool2d."""

    def __init__(
        self,
        fp: nn.MaxPool2d,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ) -> None:
        """Create one shared activation quantization domain for MaxPool2d."""
        super().__init__(qcfg, fp_name=fp_name)
        if fp.return_indices:
            raise ValueError("QuantMaxPool2d does not support return_indices=True.")
        self.module = fp
        self.obs_act_in = self._make_obs("act_in")

    @property
    def obs_act_out(self):
        """Return the shared output observer alias."""
        return self.obs_act_in

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Execute MaxPool2d with identical input and output qparams."""
        input_q = self._fq(input_, self.obs_act_in)
        output = self.module(input_q)
        return self._fq(output, self.obs_act_in)

    def _all_observers(self):
        """Return the single shared activation observer."""
        return (self.obs_act_in,)
