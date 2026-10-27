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

import math
from dataclasses import replace
from typing import Optional, Tuple

import torch

from tico.quantization.wrapq.dtypes import DType, UINT8
from tico.quantization.wrapq.observers.base import (
    FrozenObserverState,
    ObserverBase,
    ObserverNotReadyError,
)
from tico.quantization.wrapq.qscheme import QScheme


class AffineObserverBase(ObserverBase):
    """Base for affine observers (min/max → scale/zp)."""

    min_val: torch.Tensor
    max_val: torch.Tensor
    _cached_scale: torch.Tensor
    _cached_zp: torch.Tensor
    _frozen_minimum: torch.Tensor
    _frozen_maximum: torch.Tensor

    def __init__(
        self,
        *,
        name: str,
        dtype: DType = UINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_ASYMM,
        channel_axis: Optional[int] = None,
    ):
        super().__init__(
            name=name, dtype=dtype, qscheme=qscheme, channel_axis=channel_axis
        )

        # Register internal statistics as buffers so they:
        #  - move correctly with `model.to(device)`
        #  - are included in state_dict (if persistent)
        #  - follow PyTorch module semantics
        #
        # Shapes may later expand (e.g. per-channel),
        # but the buffers themselves remain tracked.
        self.register_buffer("min_val", torch.tensor(math.inf))
        self.register_buffer("max_val", torch.tensor(-math.inf))

        # Cached quantization parameters.
        # Exact grids cannot be reconstructed from ranges (imported/Percentile state).
        self.register_buffer("_cached_scale", torch.tensor([]), persistent=True)
        self.register_buffer(
            "_cached_zp", torch.tensor([], dtype=torch.int), persistent=True
        )

        # Clipping provenance participates in device moves and small-state rollback.
        # Its serialized representation lives in the versioned extra state.
        self.register_buffer("_frozen_minimum", torch.tensor([]), persistent=False)
        self.register_buffer("_frozen_maximum", torch.tensor([]), persistent=False)

        # This flag is set by load_qparams(lock=True) to protect externally
        # injected qparams (e.g. GPTQ scales/zero-points)
        # Cleared by reset() so the observer can be reused for a fresh
        # calibration cycle.
        self._qparams_locked = False

        self.reset()

    @property
    def _frozen_clipping_bounds(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self._frozen_minimum.numel() == 0:
            return None
        return self._frozen_minimum, self._frozen_maximum

    @_frozen_clipping_bounds.setter
    def _frozen_clipping_bounds(
        self, bounds: Optional[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        if bounds is None:
            self._frozen_minimum = self._frozen_minimum.new_empty((0,))
            self._frozen_maximum = self._frozen_maximum.new_empty((0,))
        else:
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or not all(isinstance(t, torch.Tensor) for t in bounds)
            ):
                raise RuntimeError(
                    f"Observer {self.name!r}: corrupt clipping provenance."
                )
            self._frozen_minimum = (
                bounds[0].detach().clone().to(device=self._frozen_minimum.device)
            )
            self._frozen_maximum = (
                bounds[1].detach().clone().to(device=self._frozen_maximum.device)
            )

    def reset(self) -> None:
        """
        Reset running min/max and drop cached qparams.

        Keep all grids as registered buffers; reset begins a fresh calibration
        cycle and deliberately clears imported locks and frozen provenance.
        """
        assert isinstance(self.min_val, torch.Tensor)
        assert isinstance(self.max_val, torch.Tensor)
        self.min_val.fill_(math.inf)
        self.max_val.fill_(-math.inf)
        # Clear cached qparams while keeping buffer registration intact
        self._cached_scale = self._cached_scale.new_empty((0,))  # type: ignore[has-type]
        self._cached_zp = self._cached_zp.new_empty((0,), dtype=torch.int)  # type: ignore[has-type]
        self._qparams_locked = False
        self._frozen_clipping_bounds = None
        self._reset_frozen_state()

    def load_qparams(self, scale: torch.Tensor, zp: torch.Tensor, *, lock: bool = True):
        """
        Inject externally computed qparams and optionally lock the observer.

        Per-channel qparams are canonicalized to the one-dimensional form
        required by ``torch.fake_quantize_per_channel_affine``. External
        quantizers may keep the same qparams in broadcast-ready shapes such as
        ``(C, 1)`` for Linear weights or ``(C, 1, 1, 1)`` for Conv2d weights.

        When locked, subsequent `collect()` calls are ignored.
        """
        scale = scale.detach().clone()
        zp = zp.detach().clone()
        if self.channel_axis is not None:
            scale, zp = scale.reshape(-1), zp.reshape(-1)
        state = FrozenObserverState(
            "affine",
            self._policy(),
            "imported",
            bool(lock),
            self.fake_quant_enabled,
            scale,
            zp,
            self._observed_shape,
        )
        self._validate_grid(state, require_extent=False)
        # Validate zero points before integer conversion; no truncation is allowed.
        state = replace(state, zero_point=zp.to(torch.int))
        self._cached_scale = scale
        self._cached_zp = zp.to(torch.int)
        self._qparams_locked = bool(lock)
        self._requires_recalibration = False
        self._frozen_policy = state.policy
        self._provenance = "imported"
        self._frozen_clipping_bounds = None
        self._frozen = bool(lock) and (
            self.channel_axis is None or self._observed_shape is not None
        )
        if lock:
            self.enabled = False

    @property
    def has_qparams(self) -> bool:
        # This reports an owned validated grid, not calibration readiness.
        return self._cached_scale.numel() != 0 and self._provenance in (
            "calibrated",
            "imported",
        )

    def _statistics_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.min_val, self.max_val

    def _validate_grid(
        self, state: FrozenObserverState, *, require_extent: bool = True
    ) -> None:
        super().validate_candidate(state)
        if state.provenance not in ("calibrated", "imported") or (
            state.locked and state.provenance != "imported"
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: invalid affine provenance or lock."
            )
        scale, zp = state.scale, state.zero_point
        if (
            scale is None
            or zp is None
            or not scale.is_floating_point()
            or not scale.numel()
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: scale must be a nonempty floating tensor."
            )
        if (
            scale.shape != zp.shape
            or not bool(torch.isfinite(scale).all())
            or not bool((scale > 0).all())
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: invalid scale/zero-point shape or finite positive scale."
            )
        if not bool(torch.isfinite(zp).all()) or not bool((zp == zp.round()).all()):
            raise RuntimeError(
                f"Observer {self.name!r}: zero point must be finite and integer."
            )
        if not bool(((zp >= self.dtype.qmin) & (zp <= self.dtype.qmax)).all()):
            raise RuntimeError(
                f"Observer {self.name!r}: zero point is outside the quantization range."
            )
        if self.qscheme.is_symmetric() and bool((zp != 0).any()):
            raise RuntimeError(
                f"Observer {self.name!r}: symmetric zero point must be zero."
            )
        if self.qscheme.is_per_channel():
            if scale.dtype not in (torch.float32, torch.bfloat16):
                raise RuntimeError(
                    f"Observer {self.name!r}: per-channel scale dtype must be float32 or bfloat16; received {scale.dtype}."
                )
            if scale.dtype == torch.bfloat16 and scale.device.type == "cpu":
                raise RuntimeError(
                    f"Observer {self.name!r}: CPU per-channel fake quantization does not support bfloat16 scales; provide float32 qparams explicitly."
                )
            if (
                self.channel_axis is None
                or not isinstance(self.channel_axis, int)
                or scale.ndim != 1
            ):
                raise RuntimeError(
                    f"Observer {self.name!r}: per-channel policy requires an axis and one-dimensional grid."
                )
            shape = state.observed_shape
            if shape is None:
                if require_extent:
                    raise ObserverNotReadyError(
                        f"Observer {self.name!r}: per-channel extent is unknown; observe the site or declare its shape before freezing."
                    )
            elif (
                not -len(shape) <= self.channel_axis < len(shape)
                or shape[self.channel_axis] != scale.numel()
            ):
                raise RuntimeError(
                    f"Observer {self.name!r}: grid does not match the observed channel extent."
                )
        elif scale.numel() != 1:
            raise RuntimeError(
                f"Observer {self.name!r}: per-tensor policy requires a scalar grid."
            )
        if not isinstance(self.dtype.bits, int) or not 2 <= self.dtype.bits <= 16:
            raise RuntimeError(f"Observer {self.name!r}: unsupported affine bit width.")
        bounds = state.clipping_bounds
        if state.provenance == "calibrated" and bounds is None:
            raise RuntimeError(
                f"Observer {self.name!r}: calibrated grid lacks range provenance."
            )
        if bounds is not None:
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or not all(isinstance(t, torch.Tensor) for t in bounds)
            ):
                raise RuntimeError(
                    f"Observer {self.name!r}: corrupt clipping provenance."
                )
            minimum, maximum = bounds
            if (
                minimum.shape != maximum.shape
                or minimum.numel() != scale.numel()
                or not bool(torch.isfinite(minimum).all())
                or not bool(torch.isfinite(maximum).all())
                or not bool((minimum <= maximum).all())
            ):
                raise RuntimeError(
                    f"Observer {self.name!r}: invalid clipping provenance."
                )

    def validate_candidate(self, state: FrozenObserverState) -> None:
        if state.kind == "bypass":
            super().validate_candidate(state)
            if state.fake_quant_enabled or state.provenance != "bypassed":
                raise RuntimeError(f"Observer {self.name!r}: invalid bypass readiness.")
            return
        if state.kind != "affine":
            raise RuntimeError(f"Observer {self.name!r}: expected affine frozen state.")
        self._validate_grid(state)

    def derive_candidate(self) -> FrozenObserverState:
        if self._requires_recalibration:
            raise ObserverNotReadyError(
                f"Observer {self.name!r}: restored statistics require a fresh calibration cycle; reset and recalibrate."
            )
        if type(self).compute_qparams is not AffineObserverBase.compute_qparams:
            raise RuntimeError(
                f"Observer {type(self).__name__}: migrate compute_qparams override to pure _statistics_bounds/derive_candidate hook."
            )
        if self._frozen or self._qparams_locked:
            state = self._grid_state()
            self.validate_candidate(state)
            return state.clone()
        try:
            minimum, maximum = self._statistics_bounds()
        except ObserverNotReadyError:
            if not self.fake_quant_enabled:
                return FrozenObserverState(
                    "bypass",
                    self._policy(),
                    "bypassed",
                    fake_quant_enabled=False,
                    observed_shape=self._observed_shape,
                )
            raise
        valid = (
            minimum.numel() != 0
            and minimum.shape == maximum.shape
            and bool(torch.isfinite(minimum).all())
            and bool(torch.isfinite(maximum).all())
            and bool((minimum <= maximum).all())
        )
        if not valid:
            if not self.fake_quant_enabled:
                return FrozenObserverState(
                    "bypass",
                    self._policy(),
                    "bypassed",
                    fake_quant_enabled=False,
                    observed_shape=self._observed_shape,
                )
            error = (
                ObserverNotReadyError
                if bool(torch.isposinf(minimum).all())
                and bool(torch.isneginf(maximum).all())
                else RuntimeError
            )
            raise error(
                f"Observer {self.name!r}: missing or invalid calibration statistics; collect finite values before freezing."
            )
        scale, zp = self._derive_grid(minimum, maximum)
        state = FrozenObserverState(
            "affine",
            self._policy(),
            "calibrated",
            False,
            self.fake_quant_enabled,
            scale.detach().clone(),
            zp.detach().clone(),
            self._observed_shape,
            (minimum.detach().clone(), maximum.detach().clone()),
        )
        self.validate_candidate(state)
        return state

    def _grid_state(self) -> FrozenObserverState:
        return FrozenObserverState(
            "bypass" if self._provenance == "bypassed" else "affine",
            self._frozen_policy or self._policy(),
            self._provenance or "unready",
            self._qparams_locked,
            self.fake_quant_enabled,
            (
                None
                if self._provenance == "bypassed"
                else self._cached_scale.detach().clone()
            ),
            (
                None
                if self._provenance == "bypassed"
                else self._cached_zp.detach().clone()
            ),
            self._observed_shape,
            getattr(self, "_frozen_clipping_bounds", None),
        )

    @property
    def frozen_state(self) -> Optional[FrozenObserverState]:
        return self._grid_state().clone() if self._frozen else None

    def install_candidate(self, state: FrozenObserverState) -> None:
        self.validate_candidate(state)
        owned = state.clone()
        if owned.kind == "affine":
            assert owned.scale is not None and owned.zero_point is not None
            self._cached_scale = owned.scale
            self._cached_zp = owned.zero_point.to(torch.int)
        else:
            self._cached_scale = self._cached_scale.new_empty((0,))
            self._cached_zp = self._cached_zp.new_empty((0,))
        self._qparams_locked = owned.locked
        self._frozen_clipping_bounds = owned.clipping_bounds
        super().install_candidate(owned)

    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Explicitly finalize through the same validation as model conversion."""
        self.install_candidate(self.derive_candidate())
        return self._cached_scale.detach().clone(), self._cached_zp.detach().clone()

    def _derive_grid(self, minimum, maximum) -> Tuple[torch.Tensor, torch.Tensor]:
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        rng = maximum - minimum
        # The floor bounds the final scale, not the observed range, so a
        # degenerate (all-zero) channel quantizes with the floor scale instead
        # of floor / qmax. The value matches the ONE quantizer: a dead weight
        # channel can still carry a real bias, whose integer value scales with
        # 1 / (input_scale * weight_scale); floors far below 1e-8 push such
        # biases past integer-backend accumulator rescale limits.
        eps = 1e-8

        if self.qscheme.is_symmetric():
            max_abs = torch.maximum(maximum.abs(), minimum.abs())
            scale = torch.clamp(max_abs / qmax, min=eps)
            zp = torch.zeros_like(scale, dtype=torch.int)
            return scale, zp

        if (self.channel_axis is None) and torch.all(rng.abs() < 1e-8):
            C = minimum
            if torch.allclose(C, torch.zeros_like(C)):
                scale = torch.ones_like(C)
                zp = torch.zeros_like(C, dtype=torch.int)
            elif (C > 0).all():
                scale = torch.clamp(C, min=eps)
                zp = torch.zeros_like(C, dtype=torch.int)
            else:
                scale = torch.clamp(C.abs(), min=eps)
                zp = torch.full_like(C, qmax, dtype=torch.int)
        else:
            # Force the range to include 0
            rng = torch.where(0 < minimum, maximum, rng)
            rng = torch.where(0 > maximum, -minimum, rng)

            scale = torch.clamp(rng / (qmax - qmin), min=eps)
            zp = torch.round(qmin - minimum / scale).clamp(qmin, qmax).to(torch.int)

        return scale, zp

    def get_extra_state(self):
        state = super().get_extra_state()
        state.update(
            locked=self._qparams_locked,
            clipping_bounds=getattr(self, "_frozen_clipping_bounds", None),
        )
        return state

    def set_extra_state(self, state):
        if not isinstance(state, dict) or not isinstance(state.get("locked"), bool):
            raise RuntimeError(
                f"Observer {self.name!r}: corrupt frozen lock; reprepare and recalibrate."
            )
        if self._cached_zp.dtype != torch.int:
            raise RuntimeError(
                f"Observer {self.name!r}: frozen zero-point storage must be int32."
            )
        self._qparams_locked = state["locked"]
        self._frozen_clipping_bounds = state.get("clipping_bounds")
        super().set_extra_state(state)
        if self._qparams_locked and self._provenance != "imported":
            raise RuntimeError(
                f"Observer {self.name!r}: locked grid lacks imported provenance."
            )
        if self._cached_scale.numel() and self._provenance not in (
            "calibrated",
            "imported",
        ):
            raise RuntimeError(f"Observer {self.name!r}: grid lacks frozen provenance.")
        if self._cached_scale.numel() and not self._frozen:
            self._validate_grid(self._grid_state(), require_extent=False)

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
        if legacy and any(
            prefix + name in state_dict for name in ("_cached_scale", "_cached_zp")
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: frozen grids lack schema/provenance; reprepare and recalibrate."
            )
        if not legacy and any(
            prefix + name not in state_dict for name in ("_cached_scale", "_cached_zp")
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: incomplete frozen grid, including under non-strict load."
            )
        for name in ("min_val", "max_val", "_cached_scale", "_cached_zp"):
            key = prefix + name
            if key in state_dict:
                value = state_dict[key]
                setattr(
                    self,
                    name,
                    getattr(self, name).new_empty(value.shape, dtype=value.dtype),
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if legacy:
            self._cached_scale = self._cached_scale.new_empty((0,))
            self._cached_zp = self._cached_zp.new_empty((0,), dtype=torch.int)
            self._qparams_locked = False
            for name in ("_cached_scale", "_cached_zp"):
                if prefix + name in missing_keys:
                    missing_keys.remove(prefix + name)

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        self._check_frozen_policy()
        if not self.fake_quant_enabled:
            return x
        if not self.has_qparams:
            raise RuntimeError(
                "Call compute_qparams()/freeze_qparams() or load_qparams() first."
            )

        if (
            self.channel_axis is not None
            and x.device.type == "cpu"
            and self._cached_scale.dtype == torch.bfloat16
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: CPU per-channel fake quantization does not support bfloat16 scales; provide float32 qparams explicitly."
            )

        # A direct imported per-channel observer may have no site schema yet.
        # Validate this concrete use without promoting or mutating frozen readiness.
        if self.channel_axis is not None and (
            not -x.ndim <= self.channel_axis < x.ndim
            or x.shape[self.channel_axis] != self._cached_scale.numel()
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: input does not match the grid channel extent."
            )

        scale = self._cached_scale.to(x.device)
        zp = self._cached_zp.to(x.device, dtype=torch.int)

        if self.channel_axis is None:
            return torch.fake_quantize_per_tensor_affine(
                x,
                scale=scale,
                zero_point=zp,
                quant_min=self.dtype.qmin,
                quant_max=self.dtype.qmax,
            )
        else:
            return torch.fake_quantize_per_channel_affine(
                x,
                scale=scale,
                zero_point=zp,
                axis=self.channel_axis,
                quant_min=self.dtype.qmin,
                quant_max=self.dtype.qmax,
            )
