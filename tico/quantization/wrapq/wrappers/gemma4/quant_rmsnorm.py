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

from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4RMSNorm")
class QuantGemma4RMSNorm(QuantModuleBase):
    """PTQ wrapper for Gemma4 RMSNorm with optional scale weights."""

    def __init__(
        self,
        fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.eps = float(getattr(fp, "eps", getattr(fp, "variance_epsilon", 1e-6)))
        self.with_scale = hasattr(fp, "weight") and getattr(fp, "weight") is not None

        self.obs_weight = self._make_obs("weight")
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")

    def enable_calibration(self) -> None:
        """Enable calibration and capture the static scale weight when it exists."""
        super().enable_calibration()
        if self.with_scale:
            self.obs_weight.collect(self.module.weight)

    def _raw_weight_for_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_scale:
            return self.module.weight

        return torch.ones((x.shape[-1],), dtype=x.dtype, device=x.device)

    def _weight_for_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Return a calibrated or fake-quantized RMSNorm scale tensor."""
        weight = self._raw_weight_for_rms_norm(x)
        if self._mode is Mode.QUANT:
            weight = self.obs_weight.fake_quant(weight)
        return weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._fq(hidden_states, self.obs_act_in)
        weight = self._weight_for_rms_norm(x)

        out = torch.ops.circle_custom.rms_norm(
            x,
            weight=weight,
            eps=self.eps,
        )
        out = out.to(dtype=hidden_states.dtype)
        return self._fq(out, self.obs_act_out)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_weight, self.obs_act_in, self.obs_act_out)
