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
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4TextMLP")
class QuantGemma4TextMLP(QuantModuleBase):
    """PTQ wrapper for dense Gemma4 text MLP blocks."""

    def __init__(
        self,
        fp_mlp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_mlp
        self.act_fn = fp_mlp.act_fn

        self.gate_proj = PTQWrapper(
            fp_mlp.gate_proj,
            qcfg=qcfg.child("gate_proj") if qcfg else None,
            fp_name=join_name(fp_name, "gate_proj"),
        )
        self.up_proj = PTQWrapper(
            fp_mlp.up_proj,
            qcfg=qcfg.child("up_proj") if qcfg else None,
            fp_name=join_name(fp_name, "up_proj"),
        )
        self.down_proj = PTQWrapper(
            fp_mlp.down_proj,
            qcfg=qcfg.child("down_proj") if qcfg else None,
            fp_name=join_name(fp_name, "down_proj"),
        )

        self.obs_gate_act = self._make_obs("gate_act")
        self.obs_mul = self._make_obs("mul")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the dense MLP with activation observers around non-linear math."""
        gate = self._fq(self.act_fn(self.gate_proj(x)), self.obs_gate_act)
        up = self.up_proj(x)
        hidden = self._fq(gate * up, self.obs_mul)
        return self.down_proj(hidden)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_gate_act, self.obs_mul)
