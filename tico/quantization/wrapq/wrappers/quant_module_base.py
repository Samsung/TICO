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
from typing import Any, cast, Iterable, Iterator, Optional, Tuple

import torch
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


def _validate_local_loaded_state(module, incompatible_keys):
    """Check promised local capabilities even under an unprepared plain root.

    Cross-source checks wait for the prepared root's complete-tree post-load hook.
    """
    if module._mode is not Mode.QUANT:
        return
    groups = module.local_observer_groups()
    declared = set(groups) | set(module._conditional_quant_subtrees())
    if not module._ready_capabilities <= declared:
        raise RuntimeError("Unknown frozen capability; reprepare and recalibrate.")
    if not module._conditional_unavailable:
        for observer in groups.get("required", ()):
            observer.require_frozen()
        if "required" in groups and "required" not in module._ready_capabilities:
            raise RuntimeError("Corrupt required capability state; recalibrate.")
    for name in module._ready_capabilities:
        if name in groups:
            for observer in groups[name]:
                observer.require_frozen()
    for observer in module._all_observers():
        observer.validate_loaded_state()


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
        self._ready_capabilities: set[str] = set()
        self._frozen_schema = 1
        self._conditional_unavailable = False
        self.register_load_state_dict_post_hook(_validate_local_loaded_state)

    def _named_child_quant_modules(self) -> Iterator[Tuple[str, "QuantModuleBase"]]:
        """
        Yield immediate QuantModuleBase descendants with their module-tree names,
        skipping over pure containers.

        Stable prefixes keep recursive diagnostic names unique, e.g. `q_proj.weight` and
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
            else:
                children = list(m.named_children())
                for child_name, child in reversed(children):
                    stack.append((f"{prefix}.{child_name}", child))

    def enable_calibration(self) -> None:
        from tico.quantization.wrapq.lifecycle import enable_calibration

        enable_calibration(self)

    def freeze_qparams(self) -> None:
        from tico.quantization.wrapq.lifecycle import freeze_qparams

        freeze_qparams(self)

    def _prepare_calibration(self) -> None:
        """Observe local deterministic parameters after the common reset."""

    def local_observer_groups(self) -> dict[str, tuple[ObserverBase, ...]]:
        """Declare required sites and optional execution capabilities."""
        return {"required": tuple(self._all_observers())}

    def _conditional_quant_subtrees(self) -> dict[str, tuple[nn.Module, ...]]:
        return {}

    def _freeze_dependencies(self) -> tuple[ObserverBase, ...]:
        return ()

    def _validate_freeze(self, plan, available: set[str]) -> None:
        """Check model semantics without mutating the complete candidate plan."""

    def _validate_loaded_state(self) -> None:
        """Check semantic references after the complete subtree has loaded."""

    def has_capability(self, name: str) -> bool:
        return name in self._ready_capabilities

    def require_capability(self, name: str) -> None:
        if self._mode is Mode.QUANT and name not in self._ready_capabilities:
            raise RuntimeError(
                f"{type(self).__name__}: capability {name!r} is unavailable; "
                "reprepare and recalibrate its quantization sites."
            )

    def get_extra_state(self):
        return {
            "version": 1,
            "mode": self._mode.name,
            "capabilities": sorted(self._ready_capabilities),
            "conditional_unavailable": self._conditional_unavailable,
        }

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("version") != 1:
            raise RuntimeError("Unsupported WrapQ state; reprepare and recalibrate.")
        if not isinstance(state.get("capabilities"), list) or not all(
            isinstance(name, str) for name in state["capabilities"]
        ):
            raise RuntimeError("Corrupt WrapQ capabilities; reprepare and recalibrate.")
        if not isinstance(state.get("conditional_unavailable"), bool):
            raise RuntimeError("Corrupt WrapQ conditional readiness; recalibrate.")
        self._conditional_unavailable = state["conditional_unavailable"]
        try:
            self._mode = Mode[state["mode"]]
            self._ready_capabilities = set(state["capabilities"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Corrupt WrapQ frozen state; reprepare and recalibrate."
            ) from exc
        self._frozen_schema = 1

    def load_state_dict(self, state_dict, strict=True, assign=False):
        from tico.quantization.wrapq.lifecycle import validate_frozen_model

        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        validate_frozen_model(self)
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Range-only legacy state can be recalibrated but cannot claim QUANT.
        if prefix + "_extra_state" not in state_dict:
            if any(
                key.startswith(prefix)
                and key.endswith("._extra_state")
                and isinstance(value, dict)
                and "frozen" in value
                for key, value in state_dict.items()
            ):
                raise RuntimeError(
                    "Incomplete WrapQ wrapper state; reprepare and recalibrate "
                    "instead of loading versioned observers without their mode."
                )
            self._mode = Mode.NO_QUANT
            self._ready_capabilities = set()
            self._conditional_unavailable = False
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if (
            prefix + "_extra_state" not in state_dict
            and prefix + "_extra_state" in missing_keys
        ):
            missing_keys.remove(prefix + "_extra_state")

    def _fq(self, x: torch.Tensor, obs: ObserverBase) -> torch.Tensor:
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

        The common coordinator traverses child QuantModuleBase owners, so
        subclasses must not yield child observers here.
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

    def _make_per_tensor_obs(self, name: str, **defaults) -> ObserverBase:
        """Declare a scalar grid while retaining configured symmetry and overrides.

        Norm parameters use a scalar grid even when the weight-role default is
        per-channel. An explicit observer override still controls its own policy.
        """
        role = self.qcfg.get_role_kwargs(name).copy()
        user = self.qcfg.get_kwargs(name).copy()
        if user.get("__quant_spec_replace_role__", False):
            role = {}
        axis = user.get(
            "channel_axis", defaults.get("channel_axis", role.get("channel_axis"))
        )
        if axis is not None:
            return self._make_obs(name, **defaults)
        dtype = user.get("dtype", defaults.get("dtype", role.get("dtype")))
        scheme = defaults.get("qscheme", role.get("qscheme"))
        if dtype is not None:
            symmetric = dtype.signed or (scheme is not None and scheme.is_symmetric())
            defaults["qscheme"] = (
                QScheme.PER_TENSOR_SYMM if symmetric else QScheme.PER_TENSOR_ASYMM
            )
        return self._make_obs(name, **defaults)

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
                else wrap_val if wrap_val is not _UNSPEC else role_val
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
