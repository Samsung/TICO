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
from typing import Iterable, Optional, Tuple

import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase


class QuantModuleBase(nn.Module, ABC):
    """
    Abstract parent for EVERY wrapper.

    Responsibilities
    ----------------
    • Own *one* Mode enum (`NO_QUANT / CALIB / QUANT`)
    • Own a PTQConfig describing default / per-observer dtypes
    • Expose a canonical lifecycle:
          enable_calibration()
          freeze_qparams()
    • Provide helper `_fq(x, observer)` (“fake-quant or collect”) so
      subclasses write arithmetic code without boilerplate.
    """

    def __init__(
        self, qcfg: Optional[PTQConfig] = None, *, fp_name: Optional[str] = None
    ) -> None:
        super().__init__()
        self.qcfg = qcfg or PTQConfig()
        self._mode: Mode = Mode.NO_QUANT  # default state
        self.fp_name = fp_name

    def _child_quant_modules(self):
        """
        Yield immediate QuantModuleBase *descendants*, skipping over pure containers
        (e.g., ModuleList/Sequential/ModuleDict). Once a QuantModuleBase is found,
        do NOT descend into it here—let recursion happen level by level.
        """
        seen = set()
        stack = list(self.children())  # start from direct children

        while stack:
            m = stack.pop()
            if isinstance(m, QuantModuleBase):
                if id(m) not in seen:
                    seen.add(id(m))
                    yield m
                # IMPORTANT: do not recurse into `m` here; its own call will handle its subtree
            elif isinstance(m, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
                # `m` is a container or a non-quant leaf: keep descending until we hit quant modules
                stack.extend(list(m.children()))

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

    def named_observers(self) -> Iterable[Tuple[str, ObserverBase]]:
        for obs in self._all_observers():
            yield obs.name, obs

    def get_observer(self, name: str) -> Optional[ObserverBase]:
        for obs in self._all_observers():
            if obs.name == name:
                return obs
        return None

    def _make_obs(
        self,
        name: str,
        **default_kwargs,
    ) -> ObserverBase:
        """
        Instantiate an observer named *name*.

        Precedence (3-tier) for keys:
           • observer:  user > wrapper-default > PTQConfig.default_observer
           • dtype:     user > wrapper-default > PTQConfig.default_dtype
           • qscheme:   user > wrapper-default > PTQConfig.default_qscheme

        Other kwargs (e.g., qscheme, channel_axis, etc.) remain:
           user override > wrapper-default
        """
        _UNSPEC = object()

        wrapper_defaults = default_kwargs.copy()
        user_cfg = self.qcfg.get_kwargs(name).copy()

        def pick3(user_val, wrap_val, global_val):
            return (
                user_val
                if user_val is not _UNSPEC
                else wrap_val
                if wrap_val is not _UNSPEC
                else global_val
            )

        # 1) resolve observer class
        user_observer = user_cfg.pop("observer", _UNSPEC)
        wrapper_observer = wrapper_defaults.pop("observer", _UNSPEC)
        obs_cls = pick3(user_observer, wrapper_observer, self.qcfg.default_observer)

        # 2) resolve dtype
        user_dtype = user_cfg.pop("dtype", _UNSPEC)
        wrapper_dtype = wrapper_defaults.pop("dtype", _UNSPEC)
        final_dtype = pick3(user_dtype, wrapper_dtype, self.qcfg.default_dtype)

        # 3) resolve qscheme
        user_qscheme = user_cfg.pop("qscheme", _UNSPEC)
        wrapper_qscheme = wrapper_defaults.pop("qscheme", _UNSPEC)
        final_qscheme = pick3(user_qscheme, wrapper_qscheme, self.qcfg.default_qscheme)

        # 4) merge remaining kwargs: user_cfg wins
        final_kw = wrapper_defaults
        final_kw.update(user_cfg)
        final_kw["dtype"] = final_dtype
        final_kw["qscheme"] = final_qscheme

        return obs_cls(**final_kw, name=name)

    # nice repr
    def extra_repr(self) -> str:
        return f"mode={self._mode.name.lower()}"
