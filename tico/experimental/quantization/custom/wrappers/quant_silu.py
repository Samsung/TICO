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
from tico.experimental.quantization.custom.wrappers.mode import Mode


class QuantSiLU(nn.Module):
    """
    QuantSiLU — drop-in replacement for nn.SiLU that quantizes
    both intermediate tensors:
        • s  = sigmoid(x)   (logistic)
        • y  = x * s        (mul)
    """

    def __init__(self, dtype: DType = DType.uint(8)):
        super().__init__()
        self.sig_obs = MinMaxObserver(dtype=dtype)  # after sigmoid
        self.mul_obs = MinMaxObserver(dtype=dtype)  # after x * sigmoid
        self._mode = Mode.NO_QUANT

    def enable_calibration(self):
        self._mode = Mode.CALIB
        for obs in (self.sig_obs, self.mul_obs):
            obs.enabled = True
            obs.reset()

    def freeze_qparams(self):
        self._mode = Mode.QUANT
        for obs in (self.sig_obs, self.mul_obs):
            obs.enabled = False
            obs.compute_qparams()

    def forward(self, x: torch.Tensor):
        s = torch.sigmoid(x)
        if self._mode is Mode.CALIB:
            self.sig_obs.collect(s.detach())
        elif self._mode is Mode.QUANT:
            s = self.sig_obs.fake_quant(s)

        y = x * s
        if self._mode is Mode.CALIB:
            self.mul_obs.collect(y.detach())
        elif self._mode is Mode.QUANT:
            y = self.mul_obs.fake_quant(y)

        return y

    def extra_repr(self) -> str:
        return f"mode={self._mode.name.lower()}"
