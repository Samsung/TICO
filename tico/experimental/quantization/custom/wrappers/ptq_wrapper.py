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

from tico.experimental.quantization.custom.observers.base import ObserverBase
from tico.experimental.quantization.custom.wrappers.handlers import (
    BaseHandler,
    HANDLER_REGISTRY,
)
from tico.experimental.quantization.custom.wrappers.mode import Mode


class PTQWrapper(torch.nn.Module):
    """
    Generic fake-quant wrapper using pluggable Handlers.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        act_obs: ObserverBase,
        weight_obs: Optional[ObserverBase] = None,
    ):
        super().__init__()
        self.module = module
        self.act_obs = act_obs
        self.weight_obs = weight_obs
        self._mode = Mode.NO_QUANT

        # Select handler
        handler_cls = HANDLER_REGISTRY.get(type(module))
        if handler_cls is None:
            raise NotImplementedError(f"No handler for {type(module).__name__}")

        self.hdl: BaseHandler = handler_cls(
            module,
            act_obs=act_obs,
            weight_obs=weight_obs,
        )

    def enable_calibration(self) -> None:
        """Start (re-)calibration; disables fake-quant during stats gathering."""
        self._mode = Mode.CALIB
        self.hdl.enable_calibration()

    def freeze_qparams(self) -> None:
        """Stop collecting and enable fake-quant in the forward path."""
        self._mode = Mode.QUANT
        self.hdl.freeze_qparams()

    def forward(self, x: torch.Tensor):
        return self.hdl.forward(x, mode=self._mode)

    def extra_repr(self) -> str:
        return f"mode={str(self._mode)}"
