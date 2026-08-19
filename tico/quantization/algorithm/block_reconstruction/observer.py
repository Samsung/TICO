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

"""Differentiable affine qparams installed temporarily during reconstruction."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from torch import nn

from tico.quantization.algorithm.block_reconstruction.qdrop import (
    maybe_qdrop_activation,
)
from tico.quantization.wrapq.control import iter_quantization_sites, QuantizationSite
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase


@dataclass(frozen=True)
class AffineObserverGroup:
    """Tie one learnable qparam pair to semantically equivalent observer sites."""

    name: str
    site_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Affine observer group names must be non-empty.")
        if not self.site_paths:
            raise ValueError("Affine observer groups require at least one site path.")
        if len(set(self.site_paths)) != len(self.site_paths):
            raise ValueError(
                f"Affine observer group {self.name!r} contains duplicate site paths."
            )


class LearnableAffineObserver(ObserverBase):
    """Use STE affine fake quantization with learnable per-tensor qparams."""

    def __init__(
        self,
        original: AffineObserverBase,
        *,
        optimize_scale: bool,
        optimize_zero_point: bool,
        minimum_scale: float = 1.0e-12,
    ) -> None:
        if original.channel_axis is not None:
            raise ValueError(
                "Block reconstruction currently supports per-tensor affine "
                "activation observers only."
            )
        if minimum_scale <= 0.0 or not math.isfinite(minimum_scale):
            raise ValueError("minimum_scale must be finite and positive.")

        scale, zero_point = original.compute_qparams()
        if scale.numel() != 1 or zero_point.numel() != 1:
            raise ValueError("Learnable affine qparams must be scalar values.")
        scale_value = scale.detach().to(dtype=torch.float32).reshape(())
        zero_point_value = zero_point.detach().to(dtype=torch.float32).reshape(())
        if not torch.isfinite(scale_value) or bool(scale_value <= 0):
            raise ValueError("The initial affine scale must be finite and positive.")
        if not torch.isfinite(zero_point_value):
            raise ValueError("The initial affine zero-point must be finite.")

        super().__init__(
            name=original.name,
            dtype=original.dtype,
            qscheme=original.qscheme,
            channel_axis=None,
        )
        object.__setattr__(self, "original", original)
        self.minimum_scale = float(minimum_scale)
        self.log_scale = nn.Parameter(
            scale_value.log(),
            requires_grad=optimize_scale,
        )
        if original.qscheme.is_symmetric():
            self.register_buffer(
                "_fixed_zero_point",
                torch.zeros_like(zero_point_value),
            )
            self.zero_point_parameter: nn.Parameter | None = None
        else:
            self.register_buffer(
                "_fixed_zero_point",
                torch.empty(0, dtype=torch.float32),
            )
            self.zero_point_parameter = nn.Parameter(
                zero_point_value,
                requires_grad=optimize_zero_point,
            )
        self.enabled = False
        self.fake_quant_enabled = original.fake_quant_enabled

    @property
    def scale(self) -> torch.Tensor:
        """Return the positive differentiable scale used in fake quantization."""
        return self.log_scale.exp().clamp_min(self.minimum_scale)

    @property
    def projected_zero_point(self) -> torch.Tensor:
        """Return an STE-rounded and clamped zero-point for the forward pass."""
        if self.zero_point_parameter is None:
            return self._fixed_zero_point
        return _round_ste(self.zero_point_parameter).clamp(
            self.dtype.qmin,
            self.dtype.qmax,
        )

    def reset(self) -> None:
        """Reject calibration resets while qparams are being optimized."""
        raise RuntimeError("A learnable reconstruction observer cannot be reset.")

    def _update_stats(self, x: torch.Tensor) -> None:
        """Reject statistics collection after initial qparams were frozen."""
        del x
        raise RuntimeError(
            "A learnable reconstruction observer cannot collect calibration data."
        )

    def compute_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected backend-compatible qparams."""
        zero_point = torch.round(self.projected_zero_point).to(dtype=torch.int)
        return self.scale, zero_point

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """Apply affine fake quantization with straight-through rounding."""
        if not self.fake_quant_enabled:
            return x
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        zero_point = self.projected_zero_point.to(device=x.device, dtype=x.dtype)
        quantized = _round_ste(x / scale) + zero_point
        quantized = quantized.clamp(self.dtype.qmin, self.dtype.qmax)
        dequantized = (quantized - zero_point) * scale
        return maybe_qdrop_activation(x, dequantized)

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return only enabled qparam parameters."""
        parameters = []
        if self.log_scale.requires_grad:
            parameters.append(self.log_scale)
        if (
            self.zero_point_parameter is not None
            and self.zero_point_parameter.requires_grad
        ):
            parameters.append(self.zero_point_parameter)
        return tuple(parameters)

    def state_snapshot(self) -> dict[str, torch.Tensor]:
        """Return a detached qparam-optimization state snapshot."""
        state = {"log_scale": self.log_scale.detach().clone()}
        if self.zero_point_parameter is not None:
            state["zero_point"] = self.zero_point_parameter.detach().clone()
        return state

    def load_state_snapshot(self, state: Mapping[str, torch.Tensor]) -> None:
        """Restore a qparam-optimization state snapshot."""
        self.log_scale.data.copy_(state["log_scale"].to(self.log_scale))
        if self.zero_point_parameter is not None:
            self.zero_point_parameter.data.copy_(
                state["zero_point"].to(self.zero_point_parameter)
            )

    def qparams_dict(self) -> dict[str, float | int]:
        """Return scalar projected qparams for JSON diagnostics."""
        scale, zero_point = self.compute_qparams()
        return {
            "scale": float(scale.detach().cpu().item()),
            "zero_point": int(zero_point.detach().cpu().item()),
        }


@dataclass(frozen=True)
class LearnableObserverBinding:
    """Describe one temporary observer replacement."""

    site_path: str
    owner: nn.Module
    observer_name: str
    attribute_names: tuple[str, ...]
    original: AffineObserverBase
    proxy: LearnableAffineObserver


@dataclass(frozen=True)
class LearnableObserverGroupBinding:
    """Bind one learnable qparam pair to one or more original observers."""

    group: AffineObserverGroup
    proxy: LearnableAffineObserver
    sites: tuple[LearnableObserverBinding, ...]


class LearnableObserverSet:
    """Install, snapshot, commit, and restore tied learnable observer groups."""

    def __init__(
        self,
        model: nn.Module,
        groups: Iterable[AffineObserverGroup],
        *,
        optimize_scale: bool = True,
        optimize_zero_point: bool = True,
        minimum_scale: float = 1.0e-12,
    ) -> None:
        requested = tuple(groups)
        if not requested:
            raise ValueError("At least one affine observer group is required.")
        group_names = tuple(group.name for group in requested)
        if len(set(group_names)) != len(group_names):
            raise ValueError("Affine observer group names must be unique.")
        all_paths = tuple(path for group in requested for path in group.site_paths)
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("Observer site paths cannot belong to multiple groups.")

        sites = {site.path: site for site in iter_quantization_sites(model)}
        missing = tuple(path for path in all_paths if path not in sites)
        if missing:
            raise KeyError(f"Unknown quantization observer sites: {missing}.")

        installed: list[LearnableObserverBinding] = []
        group_bindings: list[LearnableObserverGroupBinding] = []
        try:
            for group in requested:
                originals: list[tuple[str, QuantizationSite, tuple[str, ...]]] = []
                for path in group.site_paths:
                    site = sites[path]
                    if not isinstance(site.observer, AffineObserverBase):
                        raise TypeError(
                            f"Quantization site {path!r} is not an affine observer."
                        )
                    attribute_names = _observer_attribute_names(site)
                    originals.append((path, site, attribute_names))

                representative = originals[0][1].observer
                assert isinstance(representative, AffineObserverBase)
                _validate_tied_observers(
                    group,
                    tuple(
                        item[1].observer
                        for item in originals
                        if isinstance(item[1].observer, AffineObserverBase)
                    ),
                )
                proxy = LearnableAffineObserver(
                    representative,
                    optimize_scale=optimize_scale,
                    optimize_zero_point=optimize_zero_point,
                    minimum_scale=minimum_scale,
                )
                bindings: list[LearnableObserverBinding] = []
                for path, site, attribute_names in originals:
                    assert isinstance(site.observer, AffineObserverBase)
                    binding = LearnableObserverBinding(
                        site_path=path,
                        owner=site.module,
                        observer_name=site.observer_name,
                        attribute_names=attribute_names,
                        original=site.observer,
                        proxy=proxy,
                    )
                    installed.append(binding)
                    bindings.append(binding)
                    _replace_observer(binding, proxy)
                group_bindings.append(
                    LearnableObserverGroupBinding(
                        group=group,
                        proxy=proxy,
                        sites=tuple(bindings),
                    )
                )
        except Exception:
            for binding in reversed(installed):
                _replace_observer(binding, binding.original)
            raise

        self.groups = tuple(group_bindings)
        self._closed = False

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return all enabled qparam parameters exactly once."""
        return tuple(
            parameter
            for group in self.groups
            for parameter in group.proxy.trainable_parameters()
        )

    def state_snapshot(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return a detached optimization state keyed by group name."""
        return {group.group.name: group.proxy.state_snapshot() for group in self.groups}

    def load_state_snapshot(
        self,
        state: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
        """Restore a previously captured optimization state."""
        for group in self.groups:
            group.proxy.load_state_snapshot(state[group.group.name])

    def qparams_dict(self) -> dict[str, dict[str, object]]:
        """Return projected qparams and tied site paths by group name."""
        return {
            group.group.name: {
                **group.proxy.qparams_dict(),
                "site_paths": list(group.group.site_paths),
            }
            for group in self.groups
        }

    def finalize(self) -> dict[str, dict[str, object]]:
        """Persist learned qparams into every original observer and restore them."""
        self._ensure_open()
        qparams = self.qparams_dict()
        for group in self.groups:
            scale, zero_point = group.proxy.compute_qparams()
            for binding in group.sites:
                _store_qparams(binding.original, scale, zero_point)
                binding.original.fake_quant_enabled = group.proxy.fake_quant_enabled
                _replace_observer(binding, binding.original)
        self._closed = True
        return qparams

    def restore(self) -> None:
        """Restore original observers without changing their qparams."""
        if self._closed:
            return
        for group in self.groups:
            for binding in group.sites:
                _replace_observer(binding, binding.original)
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The learnable observer set is already closed.")


def _observer_attribute_names(site: QuantizationSite) -> tuple[str, ...]:
    """Return direct module attributes that reference a reported observer.

    QuantizationSite.observer_name is a logical observer name such as
    ``act_out``. WrapQ modules commonly register the actual child module under
    an attribute such as ``obs_act_out``. Resolve replacements by object
    identity instead of assuming those two names are identical.
    """
    names = tuple(
        name for name, child in site.module._modules.items() if child is site.observer
    )
    if names:
        return names
    registered = tuple(site.module._modules)
    raise RuntimeError(
        f"Quantization site {site.path!r} reports observer "
        f"{site.observer_name!r}, but its owner does not directly register that "
        f"observer; registered child modules={registered}."
    )


def _replace_observer(
    binding: LearnableObserverBinding,
    observer: ObserverBase,
) -> None:
    """Replace every direct attribute alias for one observer binding."""
    for attribute_name in binding.attribute_names:
        setattr(binding.owner, attribute_name, observer)


def _validate_tied_observers(
    group: AffineObserverGroup,
    observers: tuple[AffineObserverBase, ...],
) -> None:
    representative = observers[0]
    reference_scale, reference_zp = representative.compute_qparams()
    for observer in observers[1:]:
        if observer.dtype != representative.dtype:
            raise ValueError(f"Tied group {group.name!r} mixes activation dtypes.")
        if observer.qscheme != representative.qscheme:
            raise ValueError(f"Tied group {group.name!r} mixes activation qschemes.")
        if observer.channel_axis is not None:
            raise ValueError(
                f"Tied group {group.name!r} contains per-channel activation qparams."
            )
        scale, zero_point = observer.compute_qparams()
        if not torch.allclose(scale, reference_scale, rtol=1.0e-6, atol=1.0e-8):
            raise ValueError(
                f"Tied group {group.name!r} starts from inconsistent scales."
            )
        if not torch.equal(zero_point, reference_zp):
            raise ValueError(
                f"Tied group {group.name!r} starts from inconsistent zero-points."
            )
        if observer.fake_quant_enabled != representative.fake_quant_enabled:
            raise ValueError(
                f"Tied group {group.name!r} mixes fake-quant enabled states."
            )


def _store_qparams(
    observer: AffineObserverBase,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> None:
    stored_scale = scale.detach().to(device=observer.min_val.device)
    stored_zero_point = zero_point.detach().to(
        device=observer.min_val.device,
        dtype=torch.int,
    )
    qmin, qmax = observer.dtype.qmin, observer.dtype.qmax
    if observer.qscheme.is_symmetric():
        observer.min_val.copy_((-qmax * stored_scale).to(observer.min_val))
        observer.max_val.copy_((qmax * stored_scale).to(observer.max_val))
    else:
        observer.min_val.copy_(
            ((qmin - stored_zero_point) * stored_scale).to(observer.min_val)
        )
        observer.max_val.copy_(
            ((qmax - stored_zero_point) * stored_scale).to(observer.max_val)
        )
    observer.load_qparams(stored_scale, stored_zero_point, lock=True)


def _round_ste(value: torch.Tensor) -> torch.Tensor:
    return value + (torch.round(value) - value).detach()
