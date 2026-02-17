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
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextMLP",
)
class QuantQwen3VLTextMLP(QuantModuleBase):
    def __init__(
        self,
        mlp_fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # Configure child modules with specific quantization configs if provided
        gate_proj_cfg = qcfg.child("gate_proj") if qcfg else None
        up_proj_cfg = qcfg.child("up_proj") if qcfg else None
        down_proj_cfg = qcfg.child("down_proj") if qcfg else None
        act_fn_cfg = qcfg.child("act_fn") if qcfg else None

        # ----- wrap Linear layers -------------------------------
        assert hasattr(mlp_fp, "gate_proj") and isinstance(mlp_fp.gate_proj, nn.Module)
        assert hasattr(mlp_fp, "up_proj") and isinstance(mlp_fp.up_proj, nn.Module)
        assert hasattr(mlp_fp, "down_proj") and isinstance(mlp_fp.down_proj, nn.Module)

        self.gate_proj = PTQWrapper(
            mlp_fp.gate_proj, qcfg=gate_proj_cfg, fp_name=f"{fp_name}.gate_proj"
        )
        self.up_proj = PTQWrapper(
            mlp_fp.up_proj, qcfg=up_proj_cfg, fp_name=f"{fp_name}.up_proj"
        )
        self.down_proj = PTQWrapper(
            mlp_fp.down_proj, qcfg=down_proj_cfg, fp_name=f"{fp_name}.down_proj"
        )

        # ----- activation function ---------------------------------------------
        assert hasattr(mlp_fp, "act_fn") and isinstance(mlp_fp.act_fn, nn.Module)
        self.act_fn = PTQWrapper(
            mlp_fp.act_fn, qcfg=act_fn_cfg, fp_name=f"{fp_name}.act_fn"
        )

        # ----- local observers for intermediate activations --------------------
        # Observer for input activations
        mk = self._make_obs
        self.obs_act_in = mk("act_in")

        # Observers for intermediate values in the gating mechanism
        self.obs_gate_proj_out = mk("gate_proj_out")
        self.obs_up_proj_out = mk("up_proj_out")
        self.obs_act_fn_out = mk("act_fn_out")
        self.obs_gated_out = mk("gated_out")

        # Observer for final output
        self.obs_act_out = mk("act_out")

    def forward(self, hidden_state):
        # Quantize input once
        x_q = self._fq(hidden_state, self.obs_act_in)

        # Apply gate projection
        gate_proj_out = self.gate_proj(x_q)
        gate_proj_out = self._fq(gate_proj_out, self.obs_gate_proj_out)

        # Apply up projection
        up_proj_out = self.up_proj(x_q)
        up_proj_out = self._fq(up_proj_out, self.obs_up_proj_out)

        # Apply activation function to gate projection
        act_fn_out = self.act_fn(gate_proj_out)
        act_fn_out = self._fq(act_fn_out, self.obs_act_fn_out)

        # Gating mechanism: multiply activated gate with up projection
        gated_out = act_fn_out * up_proj_out
        gated_out = self._fq(gated_out, self.obs_gated_out)

        # Apply down projection
        h = self.down_proj(gated_out)
        h = self._fq(h, self.obs_act_out)

        return h

    def _all_observers(self) -> Iterable:
        # Yield local observers
        yield self.obs_act_in
        yield self.obs_gate_proj_out
        yield self.obs_up_proj_out
        yield self.obs_act_fn_out
        yield self.obs_gated_out
        yield self.obs_act_out

        # Recurse into children that are QuantModuleBase
        for m in (self.gate_proj, self.up_proj, self.down_proj, self.act_fn):
            yield from m._all_observers()
