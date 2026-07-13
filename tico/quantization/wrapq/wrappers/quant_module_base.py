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
from typing import Any, cast, Iterable, Optional, Tuple

import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.utils import auto_qscheme_for, dtype_is_unsigned
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.qscheme import QScheme


def _symmetric_qscheme_like(qscheme: QScheme) -> QScheme:
    """Return the symmetric qscheme with the same tensor/channel granularity."""
    if qscheme.is_per_channel():
        return QScheme.PER_CHANNEL_SYMM
    return QScheme.PER_TENSOR_SYMM


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

    def _named_child_quant_modules(self):
        """
        Yield immediate QuantModuleBase descendants with their module-tree names,
        skipping over pure containers.

        This mirrors _child_quant_modules(), but preserves stable prefixes so
        recursive observer names remain unique, e.g. `q_proj.weight` and
        `k_proj.weight` instead of multiple ambiguous `weight` entries.
        """
        seen = set()
        stack = list(reversed(list(self.named_children())))

        while stack:
            prefix, m = stack.pop()
            if isinstance(m, QuantModuleBase):
                if id(m) not in seen:
                    seen.add(id(m))
                    yield prefix, m
                # Do not recurse into `m` here; its own named_observers() will
                # handle its subtree and keep names relative to that child.
            elif isinstance(m, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
                children = list(m.named_children())
                for child_name, child in reversed(children):
                    stack.append((f"{prefix}.{child_name}", child))

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
        """Return observers owned directly by this module.

        Child QuantModuleBase instances are traversed by the base lifecycle
        methods, so subclasses must not yield child observers here.
        """
        ...

    def named_observers(
        self, *, recurse: bool = True
    ) -> Iterable[Tuple[str, ObserverBase]]:
        for obs in self._all_observers():
            yield obs.name, obs

        if not recurse:
            return

        for child_name, child in self._named_child_quant_modules():
            for name, obs in child.named_observers(recurse=True):
                yield f"{child_name}.{name}", obs

    def get_observer(
        self, name: str, *, recurse: bool = True
    ) -> Optional[ObserverBase]:
        """Return an observer by name, or None when it is not found."""
        for obs_name, obs in self.named_observers(recurse=recurse):
            if obs_name == name:
                return obs
        return None

    def _make_obs(
        self,
        name: str,
        **default_kwargs,
    ) -> ObserverBase:
        """
        Instantiate an observer named `name`.

        Precedence for observer, dtype, qscheme, and other kwargs is:
            1. explicit observer override in PTQConfig.overrides
            2. wrapper defaults passed by the wrapper implementation
            3. role-level QuantSpec from PTQConfig.activation or PTQConfig.weight

        ``QuantSpec.to_kwargs(mark_replace=True)`` adds the internal
        ``__quant_spec_replace_role__`` marker to an explicit observer override.
        This method removes the marker before observer construction. When the
        marker is set, the role-level activation or weight QuantSpec is discarded
        entirely, while wrapper defaults may still fill fields omitted from the
        explicit observer override.

        For example, an MX activation override must not inherit the ``dtype`` or
        ``qscheme`` from an affine role-level activation policy. Clearing
        ``role_cfg`` prevents incompatible affine settings from being passed to
        the MX observer.
        """
        _UNSPEC: Any = object()

        wrapper_defaults = default_kwargs.copy()
        role_cfg = self.qcfg.get_role_kwargs(name).copy()
        user_cfg = self.qcfg.get_kwargs(name).copy()
        replace_role = bool(user_cfg.pop("__quant_spec_replace_role__", False))
        if replace_role:
            role_cfg = {}

        def pick3(user_val: Any, wrap_val: Any, role_val: Any) -> Any:
            return (
                user_val
                if user_val is not _UNSPEC
                else wrap_val
                if wrap_val is not _UNSPEC
                else role_val
            )

        user_observer = user_cfg.pop("observer", _UNSPEC)
        role_observer = role_cfg.pop("observer", _UNSPEC)
        wrapper_observer = wrapper_defaults.pop("observer", _UNSPEC)
        obs_cls = pick3(user_observer, wrapper_observer, role_observer)

        if obs_cls is _UNSPEC:
            raise ValueError(f"No observer class configured for observer {name!r}.")

        obs_type = cast(type[ObserverBase], obs_cls)

        user_dtype = user_cfg.pop("dtype", _UNSPEC)
        role_dtype = role_cfg.pop("dtype", _UNSPEC)
        wrapper_dtype = wrapper_defaults.pop("dtype", _UNSPEC)
        final_dtype_raw = pick3(user_dtype, wrapper_dtype, role_dtype)
        final_dtype = (
            None if final_dtype_raw is _UNSPEC else cast(DType, final_dtype_raw)
        )

        user_qscheme = user_cfg.pop("qscheme", _UNSPEC)
        role_qscheme = role_cfg.pop("qscheme", _UNSPEC)
        wrapper_qscheme = wrapper_defaults.pop("qscheme", _UNSPEC)
        final_qscheme_raw = pick3(user_qscheme, wrapper_qscheme, role_qscheme)
        final_qscheme = (
            None if final_qscheme_raw is _UNSPEC else cast(QScheme, final_qscheme_raw)
        )

        if final_dtype is not None:
            if final_qscheme is None:
                final_qscheme = auto_qscheme_for(final_dtype, name)

            if dtype_is_unsigned(final_dtype) and final_qscheme.is_symmetric():
                raise ValueError(
                    f"Invalid quantization config for observer {name!r}: unsigned "
                    f"dtype {final_dtype!r} cannot be paired with symmetric "
                    f"qscheme {final_qscheme!r}."
                )

        if (
            user_qscheme is _UNSPEC
            and wrapper_qscheme is _UNSPEC
            and final_dtype is not None
            and final_qscheme is not None
            and final_dtype.signed
            and not final_qscheme.is_symmetric()
        ):
            final_qscheme = _symmetric_qscheme_like(final_qscheme)

        final_kw = role_cfg
        final_kw.update(wrapper_defaults)
        final_kw.update(user_cfg)
        if final_dtype is not None:
            final_kw["dtype"] = final_dtype
        if final_qscheme is not None:
            final_kw["qscheme"] = final_qscheme

        return obs_type(**final_kw, name=name)

    def extra_repr(self) -> str:
        return f"mode={self._mode.name.lower()}"
