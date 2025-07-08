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

import torch
import torch.nn as nn

from tico.experimental.quantization.custom.dtypes import DType
from tico.experimental.quantization.custom.observers import MinMaxObserver
from tico.experimental.quantization.custom.qscheme import QScheme
from tico.experimental.quantization.custom.wrappers.mode import Mode
from tico.experimental.quantization.custom.wrappers.ptq_wrapper import PTQWrapper
from tico.experimental.quantization.custom.wrappers.quant_silu import QuantSiLU


class QuantLlamaMLP(nn.Module):
    def __init__(self, mlp_fp: nn.Module):
        super().__init__()

        # --- helper to wrap Linears ---------------------------------
        def wrap_linear(m: nn.Linear) -> PTQWrapper:
            return PTQWrapper(
                module=m,
                act_obs=MinMaxObserver(dtype=DType.uint(8)),  # A8
                weight_obs=MinMaxObserver(  # W8
                    dtype=DType.uint(8),
                    qscheme=QScheme.PER_CHANNEL_AFFINE,
                    channel_axis=0,  # out-features
                ),
            )

        assert hasattr(mlp_fp, "gate_proj") and isinstance(mlp_fp.gate_proj, nn.Linear)
        assert hasattr(mlp_fp, "up_proj") and isinstance(mlp_fp.up_proj, nn.Linear)
        assert hasattr(mlp_fp, "down_proj") and isinstance(mlp_fp.down_proj, nn.Linear)
        self.gate_proj = wrap_linear(mlp_fp.gate_proj)
        self.up_proj = wrap_linear(mlp_fp.up_proj)
        self.down_proj = wrap_linear(mlp_fp.down_proj)

        assert hasattr(mlp_fp, "act_fn")
        # --- activation ---------------------------------------------
        if isinstance(mlp_fp.act_fn, torch.nn.SiLU):
            # Need internal quant for sigmoid + mul
            self.act_fn = QuantSiLU(dtype=DType.uint(8))  # type: ignore[assignment]
        else:
            assert isinstance(mlp_fp.act_fn, torch.nn.Module)
            # Any other weight-less activation
            self.act_fn = PTQWrapper(
                module=mlp_fp.act_fn,
                act_obs=MinMaxObserver(dtype=DType.uint(8)),
                weight_obs=None,  # no parameters
            )  # type: ignore[assignment]

        # Observer for outer product (act * up)
        self.mul_obs = MinMaxObserver(dtype=DType.uint(8))
        # Observer for input
        self.input_obs = MinMaxObserver(dtype=DType.uint(8))

        self._mode: Mode = Mode.NO_QUANT

    def enable_calibration(self):
        self._mode = Mode.CALIB
        for m in (self.gate_proj, self.up_proj, self.down_proj, self.act_fn):
            m.enable_calibration()  # type: ignore[operator]
        for obs in (self.input_obs, self.mul_obs):
            obs.enabled = True
            obs.reset()

    def freeze_qparams(self):
        self._mode = Mode.QUANT
        for m in (self.gate_proj, self.up_proj, self.down_proj, self.act_fn):
            m.freeze_qparams()  # type: ignore[operator]
        for obs in (self.input_obs, self.mul_obs):
            obs.enabled = False
            obs.compute_qparams()

    def forward(self, x: torch.Tensor):
        if self._mode is Mode.CALIB:
            self.input_obs.collect(x.detach())
        elif self._mode is Mode.QUANT:
            x = self.input_obs.fake_quant(x)

        g = self.gate_proj(x)
        u = self.up_proj(x)
        a = self.act_fn(g)
        h = a * u

        if self._mode is Mode.CALIB:
            self.mul_obs.collect(h.detach())
        elif self._mode is Mode.QUANT:
            h = self.mul_obs.fake_quant(h)

        return self.down_proj(h)

    def extra_repr(self):
        return f"mode={self._mode.name.lower()}"
