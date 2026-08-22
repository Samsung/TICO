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

"""Joint depthwise/pointwise AdaRound with learnable per-channel scales."""

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


_SUPPORTED_FAMILIES = frozenset({"regular_conv", "depthwise_conv"})


@dataclass(frozen=True)
class JointAdaRoundWeightGroup:
    """Identify one Conv2d weight in a joint reconstruction window."""

    name: str
    site_path: str
    family: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Joint AdaRound group name must be non-empty.")
        if not self.site_path:
            raise ValueError("Joint AdaRound site_path must be non-empty.")
        if self.family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                "Joint AdaRound family must be 'regular_conv' or " "'depthwise_conv'."
            )


@dataclass(frozen=True)
class JointAdaRoundWeightStatistics:
    """Summarize one finalized scale-and-rounding decision tensor."""

    site_path: str
    family: str
    element_count: int
    channel_count: int
    round_up_count: int
    changed_from_initial_nearest_count: int
    changed_from_final_nearest_count: int
    clipped_code_count: int
    changed_scale_channel_count: int
    initial_scale_minimum: float
    initial_scale_maximum: float
    final_scale_minimum: float
    final_scale_maximum: float
    scale_ratio_minimum: float
    scale_ratio_maximum: float
    scale_ratio_mean: float

    @property
    def round_up_ratio(self) -> float:
        return self.round_up_count / max(self.element_count, 1)

    @property
    def changed_from_initial_nearest_ratio(self) -> float:
        return self.changed_from_initial_nearest_count / max(
            self.element_count,
            1,
        )

    @property
    def changed_from_final_nearest_ratio(self) -> float:
        return self.changed_from_final_nearest_count / max(
            self.element_count,
            1,
        )

    @property
    def changed_scale_channel_ratio(self) -> float:
        return self.changed_scale_channel_count / max(self.channel_count, 1)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "site_path": self.site_path,
            "family": self.family,
            "element_count": self.element_count,
            "channel_count": self.channel_count,
            "round_up_count": self.round_up_count,
            "round_up_ratio": self.round_up_ratio,
            "changed_from_initial_nearest_count": (
                self.changed_from_initial_nearest_count
            ),
            "changed_from_initial_nearest_ratio": (
                self.changed_from_initial_nearest_ratio
            ),
            "changed_from_final_nearest_count": (self.changed_from_final_nearest_count),
            "changed_from_final_nearest_ratio": (self.changed_from_final_nearest_ratio),
            "clipped_code_count": self.clipped_code_count,
            "changed_scale_channel_count": self.changed_scale_channel_count,
            "changed_scale_channel_ratio": self.changed_scale_channel_ratio,
            "initial_scale_minimum": self.initial_scale_minimum,
            "initial_scale_maximum": self.initial_scale_maximum,
            "final_scale_minimum": self.final_scale_minimum,
            "final_scale_maximum": self.final_scale_maximum,
            "scale_ratio_minimum": self.scale_ratio_minimum,
            "scale_ratio_maximum": self.scale_ratio_maximum,
            "scale_ratio_mean": self.scale_ratio_mean,
        }


class LearnableScaleAdaRoundWeightQuantizer(AffineObserverBase):
    """Learn per-channel scale and element-wise floor/ceil decisions jointly.

    The calibrated zero-point, dtype, qscheme, and channel axis remain fixed.
    A bounded log-scale correction keeps every learned scale positive and near
    its calibrated initialization. Integer cells are recomputed from the
    current scale on every forward pass. A straight-through floor gives the
    scale parameter a useful reconstruction gradient while hard checkpoints
    always use exact integer floor/ceil decisions.
    """

    def __init__(
        self,
        original: AffineObserverBase,
        weight: torch.Tensor,
        *,
        gamma: float = -0.1,
        zeta: float = 1.1,
        initialization_epsilon: float = 1.0e-6,
        max_scale_ratio: float = 1.25,
    ) -> None:
        _validate_quantizer_arguments(
            original,
            weight,
            gamma=gamma,
            zeta=zeta,
            initialization_epsilon=initialization_epsilon,
            max_scale_ratio=max_scale_ratio,
        )
        super().__init__(
            name=original.name,
            dtype=original.dtype,
            qscheme=original.qscheme,
            channel_axis=original.channel_axis,
        )
        initial_scale, zero_point = original.compute_qparams()
        initial_scale = initial_scale.detach().clone()
        zero_point = zero_point.detach().clone().to(torch.int)
        self.load_qparams(initial_scale, zero_point, lock=True)
        self.min_val = original.min_val.detach().clone()
        self.max_val = original.max_val.detach().clone()
        self.enabled = original.enabled
        self.fake_quant_enabled = original.fake_quant_enabled
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.initialization_epsilon = float(initialization_epsilon)
        self.max_scale_ratio = float(max_scale_ratio)
        self.hard = False

        self.register_buffer(
            "_reference_weight",
            weight.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "_initial_scale",
            initial_scale,
            persistent=False,
        )
        self.register_buffer(
            "_fixed_zero_point",
            zero_point,
            persistent=False,
        )

        scale_broadcast, zero_point_broadcast = _broadcast_qparams(
            self._reference_weight,
            initial_scale,
            zero_point,
            channel_axis=original.channel_axis,
        )
        normalized = self._reference_weight / scale_broadcast
        normalized = normalized + zero_point_broadcast
        floor_codes = torch.floor(normalized)
        fraction = normalized - floor_codes
        sigmoid_target = (fraction - self.gamma) / (self.zeta - self.gamma)
        sigmoid_target = sigmoid_target.clamp(
            self.initialization_epsilon,
            1.0 - self.initialization_epsilon,
        )
        alpha = torch.log(sigmoid_target / (1.0 - sigmoid_target))
        initial_nearest = torch.round(normalized).clamp(
            self.dtype.qmin,
            self.dtype.qmax,
        )
        nearest_rounds_up = initial_nearest > floor_codes
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

        self.register_buffer(
            "_initial_nearest_codes",
            initial_nearest,
            persistent=False,
        )
        self.alpha = nn.Parameter(alpha)
        self.raw_log_scale_delta = nn.Parameter(torch.zeros_like(initial_scale))

    def reset(self) -> None:
        """Reset affine buffers; trainable state belongs to one frozen weight."""
        super().reset()

    def _update_stats(self, x: torch.Tensor) -> None:
        """Do not alter calibrated statistics during reconstruction."""
        del x

    def set_hard(self, hard: bool) -> None:
        self.hard = bool(hard)

    def bounded_log_scale_delta(self) -> torch.Tensor:
        maximum = math.log(self.max_scale_ratio)
        return maximum * torch.tanh(self.raw_log_scale_delta)

    def scale_ratio(self) -> torch.Tensor:
        return torch.exp(self.bounded_log_scale_delta())

    def learned_scale(self) -> torch.Tensor:
        return self._initial_scale * self.scale_ratio()

    def compute_qparams(self):
        """Return the current learned scale and fixed calibrated zero-point."""
        return self.learned_scale(), self._fixed_zero_point

    def soft_rounding(self) -> torch.Tensor:
        stretched = torch.sigmoid(self.alpha) * (self.zeta - self.gamma)
        return (stretched + self.gamma).clamp(0.0, 1.0)

    def hard_rounding(self) -> torch.Tensor:
        return (self.alpha >= 0.0).to(dtype=self.alpha.dtype)

    def rounding_regularizer(self, beta: float) -> torch.Tensor:
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("AdaRound beta must be finite and positive.")
        rounding = self.soft_rounding()
        return (1.0 - (2.0 * rounding - 1.0).abs().pow(beta)).mean()

    def scale_regularizer(self) -> torch.Tensor:
        return self.bounded_log_scale_delta().square().mean()

    def quantized_codes(self, *, hard: bool | None = None) -> torch.Tensor:
        use_hard = self.hard if hard is None else bool(hard)
        scale, zero_point = self.compute_qparams()
        scale_broadcast, zero_point_broadcast = _broadcast_qparams(
            self._reference_weight,
            scale,
            zero_point,
            channel_axis=self.channel_axis,
        )
        normalized = self._reference_weight / scale_broadcast
        normalized = normalized + zero_point_broadcast
        exact_floor = torch.floor(normalized)
        if use_hard:
            floor_codes = exact_floor
            rounding = self.hard_rounding()
        else:
            # Exact floor in the forward pass, identity in the backward pass.
            floor_codes = normalized + (exact_floor - normalized).detach()
            rounding = self.soft_rounding()
        return (floor_codes + rounding).clamp(
            self.dtype.qmin,
            self.dtype.qmax,
        )

    def hard_weight(self) -> torch.Tensor:
        scale, zero_point = self.compute_qparams()
        scale_broadcast, zero_point_broadcast = _broadcast_qparams(
            self._reference_weight,
            scale,
            zero_point,
            channel_axis=self.channel_axis,
        )
        codes = self.quantized_codes(hard=True)
        return (codes - zero_point_broadcast) * scale_broadcast

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return x
        if x.shape != self._reference_weight.shape:
            raise ValueError(
                "Joint AdaRound weight shape changed: "
                f"{tuple(x.shape)} != {tuple(self._reference_weight.shape)}."
            )
        scale, zero_point = self.compute_qparams()
        scale_broadcast, zero_point_broadcast = _broadcast_qparams(
            self._reference_weight,
            scale,
            zero_point,
            channel_axis=self.channel_axis,
        )
        codes = self.quantized_codes()
        return (codes - zero_point_broadcast) * scale_broadcast

    def statistics(
        self,
        site_path: str,
        family: str,
    ) -> JointAdaRoundWeightStatistics:
        with torch.no_grad():
            final_scale = self.learned_scale().detach()
            scale_broadcast, zero_point_broadcast = _broadcast_qparams(
                self._reference_weight,
                final_scale,
                self._fixed_zero_point,
                channel_axis=self.channel_axis,
            )
            normalized = self._reference_weight / scale_broadcast
            normalized = normalized + zero_point_broadcast
            floor_codes = torch.floor(normalized)
            raw_codes = floor_codes + self.hard_rounding().detach()
            codes = raw_codes.clamp(self.dtype.qmin, self.dtype.qmax)
            final_nearest = torch.round(normalized).clamp(
                self.dtype.qmin,
                self.dtype.qmax,
            )
            scale_ratio = final_scale / self._initial_scale
            round_up = self.hard_rounding().detach()
            changed_scale = (scale_ratio - 1.0).abs() > 1.0e-6
            return JointAdaRoundWeightStatistics(
                site_path=site_path,
                family=family,
                element_count=round_up.numel(),
                channel_count=final_scale.numel(),
                round_up_count=int(round_up.sum().cpu().item()),
                changed_from_initial_nearest_count=int(
                    (codes != self._initial_nearest_codes).sum().cpu().item()
                ),
                changed_from_final_nearest_count=int(
                    (codes != final_nearest).sum().cpu().item()
                ),
                clipped_code_count=int((raw_codes != codes).sum().cpu().item()),
                changed_scale_channel_count=int(changed_scale.sum().cpu().item()),
                initial_scale_minimum=float(self._initial_scale.min().cpu().item()),
                initial_scale_maximum=float(self._initial_scale.max().cpu().item()),
                final_scale_minimum=float(final_scale.min().cpu().item()),
                final_scale_maximum=float(final_scale.max().cpu().item()),
                scale_ratio_minimum=float(scale_ratio.min().cpu().item()),
                scale_ratio_maximum=float(scale_ratio.max().cpu().item()),
                scale_ratio_mean=float(scale_ratio.mean().cpu().item()),
            )


@dataclass
class _JointBinding:
    group: JointAdaRoundWeightGroup
    owner: nn.Module
    attribute_names: tuple[str, ...]
    weight_module: nn.Conv2d
    original_observer: AffineObserverBase
    original_weight: torch.Tensor
    original_scale: torch.Tensor
    original_zero_point: torch.Tensor
    original_qparams_locked: bool
    original_enabled: bool
    original_fake_quant_enabled: bool
    proxy: LearnableScaleAdaRoundWeightQuantizer


class JointAdaRoundWeightSet:
    """Install joint scale/rounding proxies and commit or roll back atomically."""

    def __init__(
        self,
        model: nn.Module,
        groups: Sequence[JointAdaRoundWeightGroup],
        *,
        gamma: float,
        zeta: float,
        initialization_epsilon: float,
        max_scale_ratio: float,
    ) -> None:
        definitions = tuple(groups)
        _validate_group_definitions(definitions)
        sites = {site.path: site for site in iter_quantization_sites(model)}
        bindings: list[_JointBinding] = []
        try:
            for group in definitions:
                site = sites.get(group.site_path)
                if site is None:
                    raise KeyError(
                        f"Unknown joint AdaRound weight site {group.site_path!r}."
                    )
                binding = _build_binding(
                    group,
                    site,
                    gamma=gamma,
                    zeta=zeta,
                    initialization_epsilon=initialization_epsilon,
                    max_scale_ratio=max_scale_ratio,
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

    def alpha_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(binding.proxy.alpha for binding in self.bindings)

    def scale_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(binding.proxy.raw_log_scale_delta for binding in self.bindings)

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return (*self.alpha_parameters(), *self.scale_parameters())

    def set_hard(self, hard: bool) -> None:
        for binding in self.bindings:
            binding.proxy.set_hard(hard)

    def state_snapshot(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            binding.group.name: {
                "alpha": binding.proxy.alpha.detach().cpu().clone(),
                "raw_log_scale_delta": (
                    binding.proxy.raw_log_scale_delta.detach().cpu().clone()
                ),
            }
            for binding in self.bindings
        }

    def load_state_snapshot(
        self,
        state: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
        expected = tuple(binding.group.name for binding in self.bindings)
        if tuple(state) != expected:
            raise ValueError(
                f"Joint AdaRound state keys differ: {tuple(state)} != {expected}."
            )
        with torch.no_grad():
            for binding in self.bindings:
                values = state[binding.group.name]
                for name, parameter in (
                    ("alpha", binding.proxy.alpha),
                    (
                        "raw_log_scale_delta",
                        binding.proxy.raw_log_scale_delta,
                    ),
                ):
                    value = values.get(name)
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"Joint AdaRound state {name!r} for "
                            f"{binding.group.name!r} is not a Tensor."
                        )
                    if value.shape != parameter.shape:
                        raise ValueError(
                            f"Joint AdaRound {name} shape mismatch for "
                            f"{binding.group.name!r}."
                        )
                    parameter.copy_(
                        value.to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )

    def rounding_regularizer(self, beta: float) -> torch.Tensor:
        weighted = [
            binding.proxy.rounding_regularizer(beta) * binding.proxy.alpha.numel()
            for binding in self.bindings
        ]
        total = sum(binding.proxy.alpha.numel() for binding in self.bindings)
        return torch.stack(weighted).sum() / max(total, 1)

    def scale_regularizer(self) -> torch.Tensor:
        weighted = [
            binding.proxy.scale_regularizer()
            * binding.proxy.raw_log_scale_delta.numel()
            for binding in self.bindings
        ]
        total = sum(
            binding.proxy.raw_log_scale_delta.numel() for binding in self.bindings
        )
        return torch.stack(weighted).sum() / max(total, 1)

    def statistics(self) -> tuple[JointAdaRoundWeightStatistics, ...]:
        return tuple(
            binding.proxy.statistics(
                binding.group.site_path,
                binding.group.family,
            )
            for binding in self.bindings
        )

    def finalize(self) -> tuple[JointAdaRoundWeightStatistics, ...]:
        if self._closed:
            raise RuntimeError("Joint AdaRound weight set is already closed.")
        self.set_hard(True)
        statistics = self.statistics()
        committed = tuple(
            (
                binding,
                binding.proxy.hard_weight().detach().clone(),
                binding.proxy.learned_scale().detach().clone(),
                binding.proxy._fixed_zero_point.detach().clone(),
            )
            for binding in self.bindings
        )
        try:
            with torch.no_grad():
                for binding, weight, _, _ in committed:
                    binding.weight_module.weight.copy_(weight)
            for binding, _, scale, zero_point in committed:
                device = binding.original_observer.min_val.device
                binding.original_observer.load_qparams(
                    scale.to(device=device),
                    zero_point.to(device=device),
                    lock=True,
                )
                binding.original_observer.fake_quant_enabled = (
                    binding.original_fake_quant_enabled
                )
            self._restore_observers()
        except Exception:
            self._restore_original_state()
            raise
        self._closed = True
        return statistics

    def restore(self) -> None:
        if self._closed:
            return
        self._restore_original_state()
        self._closed = True

    def _restore_original_state(self) -> None:
        with torch.no_grad():
            for binding in self.bindings:
                binding.weight_module.weight.copy_(binding.original_weight)
        for binding in self.bindings:
            device = binding.original_observer.min_val.device
            binding.original_observer.load_qparams(
                binding.original_scale.to(device=device),
                binding.original_zero_point.to(device=device),
                lock=binding.original_qparams_locked,
            )
            binding.original_observer.enabled = binding.original_enabled
            binding.original_observer.fake_quant_enabled = (
                binding.original_fake_quant_enabled
            )
        self._restore_observers()

    def _restore_observers(self) -> None:
        for binding in self.bindings:
            current = tuple(
                getattr(binding.owner, name, None) for name in binding.attribute_names
            )
            if all(value is binding.original_observer for value in current):
                continue
            if not all(value is binding.proxy for value in current):
                raise RuntimeError(
                    f"Observer aliases for {binding.group.site_path!r} are in "
                    "a mixed state and cannot be restored transactionally."
                )
            _replace_observer_attributes(
                binding.owner,
                binding.attribute_names,
                expected=binding.proxy,
                replacement=binding.original_observer,
                site_path=binding.group.site_path,
            )


def _validate_quantizer_arguments(
    original: AffineObserverBase,
    weight: torch.Tensor,
    *,
    gamma: float,
    zeta: float,
    initialization_epsilon: float,
    max_scale_ratio: float,
) -> None:
    if not original.has_qparams:
        raise ValueError("Joint AdaRound requires frozen weight qparams.")
    if not weight.is_floating_point():
        raise TypeError("Joint AdaRound requires floating-point weights.")
    if not math.isfinite(gamma) or gamma >= 0.0:
        raise ValueError("AdaRound gamma must be finite and negative.")
    if not math.isfinite(zeta) or zeta <= 1.0:
        raise ValueError("AdaRound zeta must be finite and greater than one.")
    if not 0.0 < initialization_epsilon < 0.5:
        raise ValueError("initialization_epsilon must be in (0, 0.5).")
    if not math.isfinite(max_scale_ratio) or max_scale_ratio <= 1.0:
        raise ValueError("max_scale_ratio must be finite and greater than one.")


def _validate_group_definitions(
    groups: Sequence[JointAdaRoundWeightGroup],
) -> None:
    if not groups:
        raise ValueError("Joint AdaRound requires at least one weight group.")
    names = tuple(group.name for group in groups)
    paths = tuple(group.site_path for group in groups)
    if len(set(names)) != len(names):
        raise ValueError("Joint AdaRound group names must be unique.")
    if len(set(paths)) != len(paths):
        raise ValueError("Joint AdaRound weight sites must be unique.")


def _build_binding(
    group: JointAdaRoundWeightGroup,
    site: QuantizationSite,
    *,
    gamma: float,
    zeta: float,
    initialization_epsilon: float,
    max_scale_ratio: float,
) -> _JointBinding:
    if site.role is not SiteRole.PARAMETER or site.observer_name != "weight":
        raise ValueError(
            f"Joint AdaRound site {site.path!r} must be a weight parameter site."
        )
    if not isinstance(site.observer, AffineObserverBase):
        raise TypeError(f"Joint AdaRound site {site.path!r} is not affine.")
    weight_module = getattr(site.module, "module", None)
    if not isinstance(weight_module, nn.Conv2d):
        raise TypeError(
            "Joint AdaRound supports Conv2d weights, got "
            f"{type(weight_module).__name__} at {site.path!r}."
        )
    expected_family = _conv_family(weight_module)
    if group.family != expected_family:
        raise ValueError(
            f"Group {group.name!r} declares {group.family!r}, but "
            f"{site.path!r} is {expected_family!r}."
        )
    if site.observer.channel_axis != 0:
        raise ValueError(
            f"Conv2d joint AdaRound expects channel_axis=0 at {site.path!r}."
        )
    attributes = _observer_attribute_names(site.module, site.observer)
    if not attributes:
        raise RuntimeError(
            f"No registered observer attribute aliases site {site.path!r}."
        )
    original_scale, original_zero_point = site.observer.compute_qparams()
    proxy = LearnableScaleAdaRoundWeightQuantizer(
        site.observer,
        weight_module.weight,
        gamma=gamma,
        zeta=zeta,
        initialization_epsilon=initialization_epsilon,
        max_scale_ratio=max_scale_ratio,
    )
    return _JointBinding(
        group=group,
        owner=site.module,
        attribute_names=attributes,
        weight_module=weight_module,
        original_observer=site.observer,
        original_weight=weight_module.weight.detach().clone(),
        original_scale=original_scale.detach().clone(),
        original_zero_point=original_zero_point.detach().clone(),
        original_qparams_locked=bool(getattr(site.observer, "_qparams_locked", False)),
        original_enabled=bool(site.observer.enabled),
        original_fake_quant_enabled=bool(site.observer.fake_quant_enabled),
        proxy=proxy,
    )


def _conv_family(module: nn.Conv2d) -> str:
    if (
        module.groups == module.in_channels
        and module.out_channels % module.in_channels == 0
    ):
        return "depthwise_conv"
    return "regular_conv"


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
            f"Observer aliases {mismatched} for {site_path!r} no longer "
            "reference the expected object."
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
