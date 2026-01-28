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

from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.llama.modeling_llama.LlamaRMSNorm",
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm",
)
class QuantRMSNorm(QuantModuleBase):
    def __init__(
        self,
        fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.eps = float(self.module.variance_epsilon)

        self.obs_weight = self._make_obs("weight")
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")

    def enable_calibration(self) -> None:
        super().enable_calibration()
        # immediately capture the fixed weight range
        self.obs_weight.collect(self.module.weight)

    def forward(self, x: torch.Tensor):
        # 1) quantize input once
        x_q = self._fq(x, self.obs_act_in)

        # 2) quantize weights
        w = self.module.weight
        if self._mode is Mode.QUANT:
            w = self.obs_weight.fake_quant(w)

        # 3) rms
        rms = torch.ops.circle_custom.rms_norm(
            x_q,
            weight=w,
            eps=self.eps,
        )
        rms_q = self._fq(rms, self.obs_act_out)

        return rms_q

    def _all_observers(self) -> Iterable:
        return (self.obs_weight, self.obs_act_in, self.obs_act_out)
