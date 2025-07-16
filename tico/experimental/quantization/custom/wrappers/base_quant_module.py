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
from typing import Iterable, Optional

import torch.nn as nn

from tico.experimental.quantization.custom.mode import Mode

from tico.experimental.quantization.custom.observers import MinMaxObserver
from tico.experimental.quantization.custom.observers.base import ObserverBase
from tico.experimental.quantization.custom.quant_config import QuantConfig


class QuantModuleBase(nn.Module, ABC):
    """
    Abstract parent for EVERY wrapper.

    Responsibilities
    ----------------
    • Own *one* Mode enum (`NO_QUANT / CALIB / QUANT`)
    • Own a QuantConfig describing default / per-observer dtypes
    • Expose a canonical lifecycle:
          enable_calibration()
          freeze_qparams()
    • Provide helper `_fq(x, observer)` (“fake-quant or collect”) so
      subclasses write arithmetic code without boilerplate.
    """

    def __init__(self, qcfg: Optional[QuantConfig] = None) -> None:
        super().__init__()
        self.qcfg = qcfg or QuantConfig()
        self._mode: Mode = Mode.NO_QUANT  # default state

    def _child_quant_modules(self):
        """Yield direct children that are QuantModuleBase."""
        for m in self.children():
            if isinstance(m, QuantModuleBase):
                yield m

    def enable_calibration(self) -> None:
        self._mode = Mode.CALIB
        for obs in self._all_observers():
            obs.enabled = True
            obs.reset()

        # propagate to children
        for child in self._child_quant_modules():
            child.enable_calibration()

    def freeze_qparams(self) -> None:
        self._mode = Mode.QUANT
        for obs in self._all_observers():
            obs.enabled = False
            obs.compute_qparams()

        # propagate to children
        for child in self._child_quant_modules():
            child.freeze_qparams()

    def _fq(self, x, obs: ObserverBase):
        """Fake-quant or collect."""
        if self._mode is Mode.CALIB:
            obs.collect(x.detach())
            return x
        if self._mode is Mode.QUANT:
            return obs.fake_quant(x)
        return x  # NO_QUANT

    @abstractmethod
    def _all_observers(self) -> Iterable[ObserverBase]:
        """Return every observer owned by this module."""
        ...

    def _make_obs(
        self,
        name: str,
        *,
        default_factory=MinMaxObserver,
        **default_kwargs,
    ) -> ObserverBase:
        """
        Create an observer called *name*.

        1.  Start from `default_kwargs` (e.g. qscheme, channel_axis).
        2.  Overlay the user's overrides from `QuantConfig`.
        3.  If the overrides contain a `"factory"` key, use that class
            instead of `default_factory`.
        """
        kw = default_kwargs.copy()
        kw_cfg = self.qcfg.get_kwargs(name).copy()

        factory = kw_cfg.pop("factory", default_factory)
        kw.update(kw_cfg)  # user wins

        return factory(**kw)

    # nice repr
    def extra_repr(self) -> str:
        return f"mode={self._mode.name.lower()}"
