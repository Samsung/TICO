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
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionMLP",
)
class QuantQwen3VLVisionMLP(QuantModuleBase):
    def __init__(
        self,
        mlp_fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        linear_fc1_cfg = qcfg.child("linear_fc1") if qcfg else None
        linear_fc2_cfg = qcfg.child("linear_fc2") if qcfg else None
        act_cfg = qcfg.child("act_fn") if qcfg else None

        # ----- wrap three Linear layers -------------------------------
        assert hasattr(mlp_fp, "linear_fc1") and isinstance(
            mlp_fp.linear_fc1, torch.nn.Module
        )
        assert hasattr(mlp_fp, "linear_fc2") and isinstance(
            mlp_fp.linear_fc2, torch.nn.Module
        )

        self.linear_fc1 = PTQWrapper(
            mlp_fp.linear_fc1, qcfg=linear_fc1_cfg, fp_name=f"{fp_name}.linear_fc1"
        )
        self.linear_fc2 = PTQWrapper(
            mlp_fp.linear_fc2, qcfg=linear_fc2_cfg, fp_name=f"{fp_name}.linear_fc2"
        )

        # ----- activation ---------------------------------------------
        assert hasattr(mlp_fp, "act_fn") and isinstance(mlp_fp.act_fn, torch.nn.Module)
        self.act_fn = PTQWrapper(
            mlp_fp.act_fn, qcfg=act_cfg, fp_name=f"{fp_name}.act_fn"
        )

        # ----- local observers ----------------------------------------
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")

    def forward(self, hidden_state):

        # self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

        # 1) quantize input once
        x_q = self._fq(hidden_state, self.obs_act_in)

        # 2) linear_fc1
        fc1 = self.linear_fc1(x_q)

        # 3) activation on linear_fc1
        a = self.act_fn(fc1)

        # 4) linear_fc2
        h = self._fq(self.linear_fc2(a), self.obs_act_out)

        return h

    def _all_observers(self) -> Iterable:
        yield self.obs_act_in
        yield self.obs_act_out
        # recurse into children that are QuantModuleBase
        for m in (self.linear_fc1, self.linear_fc2, self.act_fn):
            yield from m._all_observers()
