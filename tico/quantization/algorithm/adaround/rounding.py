# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

"""Learnable floor/ceil decisions for fixed affine weight qparams."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from tico.quantization.wrapq.control import (
    iter_quantization_sites,
    QuantizationSite,
    SiteRole,
)
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase


@dataclass(frozen=True)
class AdaRoundWeightGroup:
    """Identify one independently rounded Conv2d weight tensor."""

    name: str
    site_path: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AdaRound weight-group name must be non-empty.")
        if not self.site_path:
            raise ValueError("AdaRound weight-group site_path must be non-empty.")


@dataclass(frozen=True)
class AdaRoundWeightStatistics:
    """Summarize one finalized hard-rounding decision tensor."""

    site_path: str
    element_count: int
    round_up_count: int
    changed_from_nearest_count: int
    clipped_code_count: int

    @property
    def round_up_ratio(self) -> float:
        return self.round_up_count / self.element_count

    @property
    def changed_from_nearest_ratio(self) -> float:
        return self.changed_from_nearest_count / self.element_count

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "site_path": self.site_path,
            "element_count": self.element_count,
            "round_up_count": self.round_up_count,
            "round_up_ratio": self.round_up_ratio,
            "changed_from_nearest_count": self.changed_from_nearest_count,
            "changed_from_nearest_ratio": self.changed_from_nearest_ratio,
            "clipped_code_count": self.clipped_code_count,
        }


class AdaRoundWeightQuantizer(AffineObserverBase):
    """Replace affine round-to-nearest with learnable soft rounding.

    Scale, zero-point, bit width, qscheme, and channel axis are copied from the
    calibrated weight observer and never optimized. Only one alpha value per
    weight element is trainable. Hard mode maps alpha >= 0 to ceil and alpha < 0
    to floor. Soft mode uses the rectified stretched sigmoid from AdaRound.
    """

    def __init__(
        self,
        original: AffineObserverBase,
        weight: torch.Tensor,
        *,
        gamma: float = -0.1,
        zeta: float = 1.1,
        initialization_epsilon: float = 1.0e-6,
    ) -> None:
        if not math.isfinite(gamma) or not math.isfinite(zeta) or gamma >= 0.0:
            raise ValueError("AdaRound gamma must be finite and negative.")
        if zeta <= 1.0 or not math.isfinite(zeta):
            raise ValueError("AdaRound zeta must be finite and greater than one.")
        if not 0.0 < initialization_epsilon < 0.5:
            raise ValueError("initialization_epsilon must be in (0, 0.5).")
        if not original.has_qparams:
            raise ValueError("AdaRound requires frozen weight qparams.")
        if not weight.is_floating_point():
            raise TypeError("AdaRound requires a floating-point weight tensor.")

        super().__init__(
            name=original.name,
            dtype=original.dtype,
            qscheme=original.qscheme,
            channel_axis=original.channel_axis,
        )
        scale, zero_point = original.compute_qparams()
        self.load_qparams(scale, zero_point, lock=True)
        self.min_val = original.min_val.detach().clone()
        self.max_val = original.max_val.detach().clone()
        self.enabled = original.enabled
        self.fake_quant_enabled = original.fake_quant_enabled
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.initialization_epsilon = float(initialization_epsilon)
        self.hard = False

        scale_broadcast, zero_point_broadcast = _broadcast_qparams(
            weight,
            scale,
            zero_point,
            channel_axis=original.channel_axis,
        )
        normalized = weight.detach() / scale_broadcast + zero_point_broadcast
        floor_codes = torch.floor(normalized)
        fraction = normalized - floor_codes
        sigmoid_target = (fraction - self.gamma) / (self.zeta - self.gamma)
        sigmoid_target = sigmoid_target.clamp(
            self.initialization_epsilon,
            1.0 - self.initialization_epsilon,
        )
        alpha = torch.log(sigmoid_target / (1.0 - sigmoid_target))
        nearest_codes = torch.round(normalized).clamp(
            self.dtype.qmin,
            self.dtype.qmax,
        )
        nearest_rounds_up = nearest_codes > floor_codes
        sign_epsilon = self.initialization_epsilon
        alpha = torch.where(
            nearest_rounds_up & (alpha <= 0.0),
            alpha.abs() + sign_epsilon,
            alpha,
        )
        alpha = torch.where(
            (~nearest_rounds_up) & (alpha >= 0.0),
            -alpha.abs() - sign_epsilon,
            alpha,
        )

        self.register_buffer("_floor_codes", floor_codes, persistent=False)
        self.register_buffer(
            "_scale_broadcast",
            scale_broadcast.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "_zero_point_broadcast",
            zero_point_broadcast.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "_nearest_codes",
            nearest_codes,
            persistent=False,
        )
        self.alpha = nn.Parameter(alpha)

    def reset(self) -> None:
        """Reset affine buffers; alpha belongs to one immutable weight tensor."""
        super().reset()

    def _update_stats(self, x: torch.Tensor) -> None:
        """AdaRound never changes the calibrated weight range."""
        del x

    def set_hard(self, hard: bool) -> None:
        self.hard = bool(hard)

    def soft_rounding(self) -> torch.Tensor:
        stretched = torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def hard_rounding(self) -> torch.Tensor:
        return (self.alpha >= 0.0).to(dtype=self.alpha.dtype)

    def rounding_regularizer(self, beta: float) -> torch.Tensor:
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("AdaRound beta must be finite and positive.")
        rounding = self.soft_rounding()
        return (1.0 - (2.0 * rounding - 1.0).abs().pow(beta)).mean()

    def quantized_codes(self, *, hard: bool | None = None) -> torch.Tensor:
        use_hard = self.hard if hard is None else bool(hard)
        rounding = self.hard_rounding() if use_hard else self.soft_rounding()
        return (self._floor_codes + rounding).clamp(
            self.dtype.qmin,
            self.dtype.qmax,
        )

    def hard_weight(self) -> torch.Tensor:
        codes = self.quantized_codes(hard=True)
        return (codes - self._zero_point_broadcast) * self._scale_broadcast

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return x
        if x.shape != self.alpha.shape:
            raise ValueError(
                "AdaRound weight shape changed after initialization: "
                f"{tuple(x.shape)} != {tuple(self.alpha.shape)}."
            )
        codes = self.quantized_codes()
        return (codes - self._zero_point_broadcast) * self._scale_broadcast

    def statistics(self, site_path: str) -> AdaRoundWeightStatistics:
        codes = self.quantized_codes(hard=True).detach()
        raw_codes = self._floor_codes + self.hard_rounding().detach()
        changed = codes != self._nearest_codes
        clipped = raw_codes != codes
        round_up = self.hard_rounding().detach()
        return AdaRoundWeightStatistics(
            site_path=site_path,
            element_count=round_up.numel(),
            round_up_count=int(round_up.sum().cpu().item()),
            changed_from_nearest_count=int(changed.sum().cpu().item()),
            clipped_code_count=int(clipped.sum().cpu().item()),
        )


@dataclass
class _AdaRoundBinding:
    group: AdaRoundWeightGroup
    owner: nn.Module
    attribute_names: tuple[str, ...]
    weight_module: nn.Conv2d
    original_observer: AffineObserverBase
    original_weight: torch.Tensor
    original_enabled: bool
    original_fake_quant_enabled: bool
    proxy: AdaRoundWeightQuantizer


class AdaRoundWeightSet:
    """Temporarily install learnable weight rounders and commit atomically."""

    def __init__(
        self,
        model: nn.Module,
        groups: Sequence[AdaRoundWeightGroup],
        *,
        gamma: float,
        zeta: float,
        initialization_epsilon: float,
    ) -> None:
        definitions = tuple(groups)
        if not definitions:
            raise ValueError("AdaRound requires at least one weight group.")
        names = tuple(group.name for group in definitions)
        paths = tuple(group.site_path for group in definitions)
        if len(set(names)) != len(names):
            raise ValueError("AdaRound weight-group names must be unique.")
        if len(set(paths)) != len(paths):
            raise ValueError("AdaRound weight sites must be unique.")

        sites = {site.path: site for site in iter_quantization_sites(model)}
        bindings: list[_AdaRoundBinding] = []
        try:
            for group in definitions:
                site = sites.get(group.site_path)
                if site is None:
                    raise KeyError(f"Unknown AdaRound weight site {group.site_path!r}.")
                binding = _build_binding(
                    group,
                    site,
                    gamma=gamma,
                    zeta=zeta,
                    initialization_epsilon=initialization_epsilon,
                )
                _replace_observer_attributes(
                    binding.owner,
                    binding.attribute_names,
                    expected=binding.original_observer,
                    replacement=binding.proxy,
                    site_path=group.site_path,
                )
                bindings.append(binding)
        except Exception:
            for binding in reversed(bindings):
                _replace_observer_attributes(
                    binding.owner,
                    binding.attribute_names,
                    expected=binding.proxy,
                    replacement=binding.original_observer,
                    site_path=binding.group.site_path,
                )
            raise

        self.bindings = tuple(bindings)
        self._closed = False

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(binding.proxy.alpha for binding in self.bindings)

    def set_hard(self, hard: bool) -> None:
        for binding in self.bindings:
            binding.proxy.set_hard(hard)

    def state_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            binding.group.name: binding.proxy.alpha.detach().cpu().clone()
            for binding in self.bindings
        }

    def load_state_snapshot(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = tuple(binding.group.name for binding in self.bindings)
        if tuple(state) != expected:
            raise ValueError(
                f"AdaRound state keys differ: {tuple(state)} != {expected}."
            )
        with torch.no_grad():
            for binding in self.bindings:
                value = state[binding.group.name]
                if value.shape != binding.proxy.alpha.shape:
                    raise ValueError(
                        f"AdaRound alpha shape mismatch for {binding.group.name!r}."
                    )
                binding.proxy.alpha.copy_(
                    value.to(
                        device=binding.proxy.alpha.device,
                        dtype=binding.proxy.alpha.dtype,
                    )
                )

    def rounding_regularizer(self, beta: float) -> torch.Tensor:
        weighted = [
            binding.proxy.rounding_regularizer(beta) * binding.proxy.alpha.numel()
            for binding in self.bindings
        ]
        total = sum(binding.proxy.alpha.numel() for binding in self.bindings)
        return torch.stack(weighted).sum() / total

    def statistics(self) -> tuple[AdaRoundWeightStatistics, ...]:
        return tuple(
            binding.proxy.statistics(binding.group.site_path)
            for binding in self.bindings
        )

    def finalize(self) -> tuple[AdaRoundWeightStatistics, ...]:
        if self._closed:
            raise RuntimeError("AdaRound weight set is already closed.")
        self.set_hard(True)
        statistics = self.statistics()
        with torch.no_grad():
            for binding in self.bindings:
                binding.weight_module.weight.copy_(binding.proxy.hard_weight())
        self._restore_observers()
        self._closed = True
        return statistics

    def restore(self) -> None:
        if self._closed:
            return
        with torch.no_grad():
            for binding in self.bindings:
                binding.weight_module.weight.copy_(binding.original_weight)
        self._restore_observers()
        self._closed = True

    def _restore_observers(self) -> None:
        for binding in self.bindings:
            binding.original_observer.enabled = binding.original_enabled
            binding.original_observer.fake_quant_enabled = (
                binding.original_fake_quant_enabled
            )
            _replace_observer_attributes(
                binding.owner,
                binding.attribute_names,
                expected=binding.proxy,
                replacement=binding.original_observer,
                site_path=binding.group.site_path,
            )


def _build_binding(
    group: AdaRoundWeightGroup,
    site: QuantizationSite,
    *,
    gamma: float,
    zeta: float,
    initialization_epsilon: float,
) -> _AdaRoundBinding:
    if site.role is not SiteRole.PARAMETER or site.observer_name != "weight":
        raise ValueError(
            f"AdaRound site {site.path!r} must be a weight parameter site."
        )
    if not isinstance(site.observer, AffineObserverBase):
        raise TypeError(f"AdaRound site {site.path!r} is not affine.")
    weight_module = getattr(site.module, "module", None)
    if not isinstance(weight_module, nn.Conv2d):
        raise TypeError(
            f"AdaRound currently supports Conv2d weights, got "
            f"{type(weight_module).__name__} at {site.path!r}."
        )
    if site.observer.channel_axis != 0:
        raise ValueError(
            f"Conv2d AdaRound expects weight channel_axis=0 at {site.path!r}."
        )
    attributes = _observer_attribute_names(site.module, site.observer)
    if not attributes:
        raise RuntimeError(
            f"No registered observer attribute aliases site {site.path!r}."
        )
    proxy = AdaRoundWeightQuantizer(
        site.observer,
        weight_module.weight,
        gamma=gamma,
        zeta=zeta,
        initialization_epsilon=initialization_epsilon,
    )
    return _AdaRoundBinding(
        group=group,
        owner=site.module,
        attribute_names=attributes,
        weight_module=weight_module,
        original_observer=site.observer,
        original_weight=weight_module.weight.detach().clone(),
        original_enabled=bool(site.observer.enabled),
        original_fake_quant_enabled=bool(site.observer.fake_quant_enabled),
        proxy=proxy,
    )


def _broadcast_qparams(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    channel_axis: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if channel_axis is None:
        return (
            scale.to(device=weight.device, dtype=weight.dtype),
            zero_point.to(device=weight.device, dtype=weight.dtype),
        )
    axis = channel_axis % weight.ndim
    shape = [1] * weight.ndim
    shape[axis] = -1
    return (
        scale.reshape(shape).to(device=weight.device, dtype=weight.dtype),
        zero_point.reshape(shape).to(device=weight.device, dtype=weight.dtype),
    )


def _observer_attribute_names(
    owner: nn.Module,
    observer: ObserverBase,
) -> tuple[str, ...]:
    return tuple(name for name, child in owner._modules.items() if child is observer)


def _replace_observer_attributes(
    owner: nn.Module,
    attribute_names: tuple[str, ...],
    *,
    expected: ObserverBase,
    replacement: ObserverBase,
    site_path: str,
) -> None:
    mismatched = tuple(
        name for name in attribute_names if getattr(owner, name, None) is not expected
    )
    if mismatched:
        raise RuntimeError(
            f"Observer aliases {mismatched} for {site_path!r} no longer reference "
            "the expected object."
        )
    replaced: list[str] = []
    try:
        for name in attribute_names:
            setattr(owner, name, replacement)
            replaced.append(name)
    except Exception:
        for name in replaced:
            setattr(owner, name, expected)
        raise
