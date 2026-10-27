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
from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.wrapq.dtypes import DType, UINT8
from tico.quantization.wrapq.qscheme import QScheme


class ObserverNotReadyError(RuntimeError):
    """A required observation or semantic extent is missing."""


@dataclass(frozen=True)
class FrozenObserverState:
    """An owned, validated candidate. Public reads clone its tensor payloads."""

    kind: str
    policy: tuple
    provenance: str
    locked: bool = False
    fake_quant_enabled: bool = True
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None
    observed_shape: Optional[tuple[int, ...]] = None
    clipping_bounds: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def clone(self) -> "FrozenObserverState":
        return replace(
            self,
            scale=None if self.scale is None else self.scale.detach().clone(),
            zero_point=(
                None if self.zero_point is None else self.zero_point.detach().clone()
            ),
            clipping_bounds=(
                None
                if self.clipping_bounds is None
                else (
                    self.clipping_bounds[0].detach().clone(),
                    self.clipping_bounds[1].detach().clone(),
                )
            ),
        )


class ObserverBase(nn.Module, ABC):
    """Statistics hooks plus the common validated frozen-state protocol."""

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        name: str,
        dtype: DType = UINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_ASYMM,
        channel_axis: Optional[int] = None,
    ):
        super().__init__()
        self.name = name
        self.dtype = dtype
        self.qscheme = qscheme
        self.channel_axis = channel_axis if qscheme.is_per_channel() else None
        self.enabled = True
        self.fake_quant_enabled = True
        self._frozen = False
        self._frozen_policy: Optional[tuple] = None
        self._requires_recalibration = False
        self._provenance: Optional[str] = None
        self._observed_shape: Optional[tuple[int, ...]] = None

    def _policy(self) -> tuple:
        return (
            self.__class__.__module__,
            self.__class__.__qualname__,
            self.dtype.bits,
            self.dtype.signed,
            self.qscheme.name,
            self.channel_axis,
        )

    def policy_signature(self) -> tuple:
        return self._policy()

    def validate_loaded_state(self) -> None:
        if not hasattr(self, "_frozen"):
            raise RuntimeError(
                f"Observer {self.name!r}: legacy whole-module state cannot prove exact frozen semantics; reprepare and recalibrate."
            )
        if any(
            not isinstance(getattr(self, field, None), bool)
            for field in ("_frozen", "enabled", "fake_quant_enabled")
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: corrupt frozen flags; reprepare and recalibrate."
            )
        if self._frozen:
            if self.enabled:
                raise RuntimeError(
                    f"Observer {self.name!r}: frozen collection cannot be enabled."
                )
            self.validate_candidate(self.require_frozen())

    def observe_shape(self, shape) -> None:
        """Record semantic rank/channel extent, including for imported grids."""
        shape = tuple(int(n) for n in shape)
        if self.channel_axis is not None:
            axis = self.channel_axis
            if not -len(shape) <= axis < len(shape):
                raise RuntimeError(
                    f"Observer {self.name!r}: channel axis {axis} is invalid for shape {shape}."
                )
            if self._observed_shape is not None:
                old = self._observed_shape
                if len(old) != len(shape) or old[axis] != shape[axis]:
                    raise RuntimeError(
                        f"Observer {self.name!r}: incompatible observed channel extent."
                    )
        if not self._frozen:
            self._observed_shape = shape

    def _reset_frozen_state(self) -> None:
        self._requires_recalibration = False
        self._frozen = False
        self._frozen_policy = None
        self._provenance = None
        self._observed_shape = None

    def set_fake_quant_enabled(self, enabled: bool) -> None:
        if enabled and self._frozen and self._provenance == "bypassed":
            raise RuntimeError(
                f"Observer {self.name!r} has no frozen grid; recalibrate before enabling fake quantization."
            )
        self.fake_quant_enabled = bool(enabled)

    def enable_fake_quant(self) -> None:
        self.set_fake_quant_enabled(True)

    def disable_fake_quant(self) -> None:
        self.set_fake_quant_enabled(False)

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def collect(self, x: torch.Tensor) -> None:
        if self._requires_recalibration:
            raise ObserverNotReadyError(
                f"Observer {self.name!r}: restored statistics require a fresh calibration cycle; reset and recalibrate."
            )
        if self._frozen or x.numel() == 0:
            return
        # Imported locks suppress statistics, but still need the site's extent.
        self.observe_shape(x.shape)
        if self.enabled:
            self._update_stats(x)

    @abstractmethod
    def _update_stats(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_qparams(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def derive_candidate(self) -> FrozenObserverState:
        raise RuntimeError(
            f"Observer {type(self).__name__} must implement the pure derive_candidate protocol; migrate legacy compute_qparams overrides."
        )

    def validate_candidate(self, state: FrozenObserverState) -> None:
        if not isinstance(state.locked, bool) or not isinstance(
            state.fake_quant_enabled, bool
        ):
            raise RuntimeError(f"Observer {self.name!r}: invalid frozen flags.")
        if state.observed_shape is not None and (
            not isinstance(state.observed_shape, tuple)
            or not all(isinstance(n, int) and n >= 0 for n in state.observed_shape)
        ):
            raise RuntimeError(f"Observer {self.name!r}: invalid frozen shape schema.")
        if state.policy != self._policy():
            raise RuntimeError(
                f"Observer {self.name!r}: incompatible frozen policy; reprepare and recalibrate."
            )
        if state.kind not in ("identity", "mx", "bypass", "affine"):
            raise RuntimeError(f"Observer {self.name!r}: invalid frozen state kind.")
        if state.provenance not in ("calibrated", "imported", "stateless", "bypassed"):
            raise RuntimeError(f"Observer {self.name!r}: invalid frozen provenance.")

    def install_candidate(self, state: FrozenObserverState) -> None:
        self.validate_candidate(state)
        self._frozen = True
        self._frozen_policy = state.policy
        self._provenance = state.provenance
        self._observed_shape = state.observed_shape
        self.fake_quant_enabled = state.fake_quant_enabled
        self.enabled = False

    @property
    def frozen_state(self) -> Optional[FrozenObserverState]:
        if not self._frozen:
            return None
        if self._provenance is None:
            raise RuntimeError(
                f"Observer {self.name!r}: frozen state lacks provenance."
            )
        return FrozenObserverState(
            kind=self._state_kind(),
            policy=self._frozen_policy or self._policy(),
            provenance=self._provenance,
            fake_quant_enabled=self.fake_quant_enabled,
            observed_shape=self._observed_shape,
        )

    def _state_kind(self) -> str:
        return "bypass"

    def _check_frozen_policy(self) -> None:
        if self._frozen_policy is not None and self._frozen_policy != self._policy():
            raise RuntimeError(
                f"Observer {self.name!r}: frozen policy changed; reprepare and recalibrate."
            )

    def require_frozen(self) -> FrozenObserverState:
        state = self.frozen_state
        if state is None:
            raise ObserverNotReadyError(
                f"Observer {self.name!r} is not frozen; calibrate and convert first."
            )
        self._check_frozen_policy()
        return state

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "policy": self._frozen_policy or self._policy(),
            "frozen": self._frozen,
            "provenance": self._provenance,
            "observed_shape": self._observed_shape,
            "enabled": self.enabled,
            "fake_quant_enabled": self.fake_quant_enabled,
            "requires_recalibration": self._requires_recalibration,
        }

    def set_extra_state(self, state) -> None:
        if not isinstance(state, dict) or state.get("version") != self.STATE_VERSION:
            raise RuntimeError(
                f"Observer {self.name!r}: unsupported frozen-state version; reprepare and recalibrate."
            )
        if state.get("policy") != self._policy():
            raise RuntimeError(
                f"Observer {self.name!r}: incompatible frozen policy; reprepare and recalibrate."
            )
        for field in (
            "frozen",
            "enabled",
            "fake_quant_enabled",
            "requires_recalibration",
        ):
            if not isinstance(state.get(field), bool):
                raise RuntimeError(
                    f"Observer {self.name!r}: corrupt frozen-state field {field}."
                )
        self._frozen_policy = (
            state["policy"]
            if state["frozen"] or state.get("provenance") == "imported"
            else None
        )
        self._requires_recalibration = state["requires_recalibration"]
        self._frozen = state["frozen"]
        if self._frozen and self._requires_recalibration:
            raise RuntimeError(
                f"Observer {self.name!r}: frozen state requires recalibration."
            )
        if "provenance" not in state or "observed_shape" not in state:
            raise RuntimeError(
                f"Observer {self.name!r}: incomplete frozen-state schema."
            )
        self._provenance = state["provenance"]
        self._observed_shape = state["observed_shape"]
        self.enabled = state["enabled"]
        self.fake_quant_enabled = state["fake_quant_enabled"]
        if self._frozen:
            snapshot = self.frozen_state
            if snapshot is None or self.enabled:
                raise RuntimeError(f"Observer {self.name!r}: corrupt frozen readiness.")
            self.validate_candidate(snapshot)

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
        legacy = prefix + "_extra_state" not in state_dict
        if legacy:
            self._reset_frozen_state()
            self._requires_recalibration = True
            self.enabled = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if legacy and prefix + "_extra_state" in missing_keys:
            missing_keys.remove(prefix + "_extra_state")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, dtype={self.dtype}, "
            f"qscheme={self.qscheme}, channel_axis={self.channel_axis}, "
            f"enabled={self.enabled}, fake_quant_enabled={self.fake_quant_enabled})"
        )
