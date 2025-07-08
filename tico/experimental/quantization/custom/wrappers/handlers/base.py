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

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn

from tico.experimental.quantization.custom.observers.base import ObserverBase
from tico.experimental.quantization.custom.wrappers.mode import Mode


class BaseHandler(ABC):
    """
    Generic interface every concrete handler must implement.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        act_obs: ObserverBase,
        weight_obs: Optional[ObserverBase],
    ):
        self.module = module
        self.act_obs = act_obs
        self.weight_obs = weight_obs

        # One-shot weight stats (if any)
        if self.weight_obs is not None and hasattr(module, "weight"):
            self.weight_obs.collect(self._weight())

    def enable_calibration(self):
        self.act_obs.enabled = True
        self.act_obs.reset()

    def freeze_qparams(self):
        self.act_obs.enabled = False
        self.act_obs.compute_qparams()
        if self.weight_obs is not None:
            self.weight_obs.compute_qparams()

    def _weight(self):
        return getattr(self.module, "weight")

    def _fake_quant_weight(self, w: torch.Tensor):
        return self.weight_obs.fake_quant(w) if self.weight_obs else w

    @abstractmethod
    def forward(self, x: torch.Tensor, *, mode: Mode) -> torch.Tensor:
        ...
