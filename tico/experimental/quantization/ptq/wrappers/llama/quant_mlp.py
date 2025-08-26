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

from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.ptq.wrappers.registry import try_register


@try_register("transformers.models.llama.modeling_llama.LlamaMLP")
class QuantLlamaMLP(QuantModuleBase):
    def __init__(
        self,
        mlp_fp: nn.Module,
        *,
        qcfg: Optional[QuantConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # ----- child configs (hierarchical override) -------------------
        gate_cfg = qcfg.child("gate_proj") if qcfg else None
        up_cfg = qcfg.child("up_proj") if qcfg else None
        down_cfg = qcfg.child("down_proj") if qcfg else None
        act_cfg = qcfg.child("act_fn") if qcfg else None

        # ----- wrap three Linear layers -------------------------------
        assert hasattr(mlp_fp, "gate_proj") and isinstance(
            mlp_fp.gate_proj, torch.nn.Module
        )
        assert hasattr(mlp_fp, "up_proj") and isinstance(
            mlp_fp.up_proj, torch.nn.Module
        )
        assert hasattr(mlp_fp, "down_proj") and isinstance(
            mlp_fp.down_proj, torch.nn.Module
        )
        self.gate_proj = PTQWrapper(
            mlp_fp.gate_proj, qcfg=gate_cfg, fp_name=f"{fp_name}.gate_proj"
        )
        self.up_proj = PTQWrapper(
            mlp_fp.up_proj, qcfg=up_cfg, fp_name=f"{fp_name}.up_proj"
        )
        self.down_proj = PTQWrapper(
            mlp_fp.down_proj, qcfg=down_cfg, fp_name=f"{fp_name}.down_proj"
        )

        # ----- activation ---------------------------------------------
        assert hasattr(mlp_fp, "act_fn") and isinstance(mlp_fp.act_fn, torch.nn.Module)
        self.act_fn = PTQWrapper(
            mlp_fp.act_fn, qcfg=act_cfg, fp_name=f"{fp_name}.act_fn"
        )

        # ----- local observers ----------------------------------------
        self.act_in_obs = self._make_obs("act_in")
        self.mul_obs = self._make_obs("mul")

    def forward(self, x: torch.Tensor):
        # 1) quantize input once
        x_q = self._fq(x, self.act_in_obs)

        # 2) parallel projections
        g = self.gate_proj(x_q)
        u = self.up_proj(x_q)

        # 3) activation on gate
        a = self.act_fn(g)

        # 4) element-wise product
        h = self._fq(a * u, self.mul_obs)

        # 5) final projection
        return self.down_proj(h)

    def _all_observers(self):
        # local first
        yield self.act_in_obs
        yield self.mul_obs
        # recurse into children that are QuantModuleBase
        for m in (self.gate_proj, self.up_proj, self.down_proj, self.act_fn):
            yield from m._all_observers()
