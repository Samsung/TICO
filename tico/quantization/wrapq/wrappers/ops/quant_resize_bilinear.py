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

"""WrapQ support for ``tico.ops.ResizeBilinear2d`` modules."""

from __future__ import annotations

from typing import Optional

import torch

from tico.ops import ResizeBilinear2d
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


@register(ResizeBilinear2d)
class QuantResizeBilinear2d(QuantModuleBase):
    """Fake-quantize both sides of one ResizeBilinear boundary."""

    def __init__(
        self,
        fp: ResizeBilinear2d,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ) -> None:
        """Create activation observers around one ResizeBilinear module."""
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Execute ResizeBilinear under the current WrapQ mode."""
        input_q = self._fq(input_, self.obs_act_in)
        output = self.module(input_q)
        return self._fq(output, self.obs_act_out)

    def _all_observers(self):
        """Return observers owned directly by this wrapper."""
        return self.obs_act_in, self.obs_act_out
