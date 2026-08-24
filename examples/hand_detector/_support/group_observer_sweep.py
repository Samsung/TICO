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

"""Group-specific activation-observer override sweep helpers."""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from tico.quantization.analysis import (
    evaluate_models,
    ModelInput,
    OutputAdapter,
    QuantizationBoundaries,
    QuantizationProfile,
)
from tico.quantization.config.specs import affine, QuantSpec
from tico.quantization.wrapq.control import (
    FakeQuantState,
    iter_quantization_sites,
    SiteRole,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.qscheme import QScheme

from torch import nn

from examples.hand_detector._support.sensitivity import ActivationSensitivityGroup


_ACTIVATION_ROLES = frozenset(
    {
        SiteRole.ACTIVATION_INPUT,
        SiteRole.ACTIVATION_OUTPUT,
        SiteRole.ACTIVATION,
    }
)


@dataclass(frozen=True)
class GroupObserverPolicy:
    """Describe one observer override applied to a single activation group."""

    name: str
    observer: type[ObserverBase]
    observer_kwargs: Mapping[str, object] = field(default_factory=dict)
    percentile: float | None = None

    def quant_spec(self, bit_width: int) -> QuantSpec:
        """Return a role-replacing activation QuantSpec for this policy."""
        dtype, qscheme = _activation_dtype_qscheme(bit_width)
        return affine(
            dtype,
            qscheme=qscheme,
            observer=self.observer,
            **dict(self.observer_kwargs),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible policy metadata."""
        value: dict[str, object] = {
            "name": self.name,
            "observer": self.observer.__name__,
        }
        if self.percentile is not None:
            value["percentile"] = self.percentile
        return value


@dataclass(frozen=True)
class EvaluatedGroupObserverPolicy:
    """Store one group-specific override evaluation under E:internal-full."""

    policy: GroupObserverPolicy
    outputs: Mapping[str, Mapping[str, float | int | None]]
    enabled_site_count: int


def build_group_observer_policies(
    *,
    percentiles: Sequence[float],
    global_percentile: float,
    max_samples: int,
    samples_per_batch: int,
    seed: int,
    include_minmax: bool = True,
) -> tuple[GroupObserverPolicy, ...]:
    """Build MinMax and percentile policies, excluding the global control."""
    _validate_percentile(global_percentile, "global_percentile")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if samples_per_batch <= 0:
        raise ValueError("samples_per_batch must be positive.")

    policies: list[GroupObserverPolicy] = []
    if include_minmax and not math.isclose(
        global_percentile,
        100.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        policies.append(
            GroupObserverPolicy(
                name="minmax",
                observer=MinMaxObserver,  # type: ignore[type-abstract]
            )
        )

    seen: set[float] = set()
    for percentile in percentiles:
        value = float(percentile)
        _validate_percentile(value, "percentile")
        if any(
            math.isclose(value, existing, rel_tol=0.0, abs_tol=1.0e-12)
            for existing in seen
        ):
            continue
        seen.add(value)
        if math.isclose(
            value,
            global_percentile,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            continue
        if include_minmax and math.isclose(
            value,
            100.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            continue
        policies.append(
            GroupObserverPolicy(
                name=_percentile_name(value),
                observer=PercentileObserver,
                observer_kwargs={
                    "percentile": value,
                    "max_samples": max_samples,
                    "samples_per_batch": samples_per_batch,
                    "seed": seed,
                },
                percentile=value,
            )
        )

    if not policies:
        raise ValueError(
            "No group-specific observer policies remain after removing the "
            "global percentile control."
        )
    return tuple(policies)


def activation_group_override_paths(
    model: nn.Module,
    group: ActivationSensitivityGroup,
) -> tuple[str, ...]:
    """Map prepared observer sites back to PTQConfig floating-point paths."""
    sites_by_path = {site.path: site for site in iter_quantization_sites(model)}
    override_paths: list[str] = []
    for site_path in group.site_paths:
        site = sites_by_path.get(site_path)
        if site is None:
            raise KeyError(
                f"Activation group {group.name!r} references unknown site "
                f"{site_path!r}."
            )
        if site.role not in _ACTIVATION_ROLES:
            raise ValueError(
                f"Activation group {group.name!r} contains non-activation site "
                f"{site.path!r} with role {site.role.value!r}."
            )
        fp_name = getattr(site.module, "fp_name", None)
        if not fp_name:
            raise ValueError(
                f"Quantization site {site.path!r} does not preserve an original "
                "floating-point module path."
            )
        override_paths.append(f"{fp_name}.{site.observer_name}")

    unique_paths = tuple(dict.fromkeys(override_paths))
    if len(unique_paths) != len(group.site_paths):
        raise RuntimeError(
            f"Activation group {group.name!r} maps multiple sites to the same "
            "PTQConfig override path."
        )
    return unique_paths


def make_group_observer_overrides(
    policy: GroupObserverPolicy,
    *,
    bit_width: int,
    override_paths: Sequence[str],
) -> dict[str, QuantSpec]:
    """Assign one role-replacing observer policy to every group site."""
    if not override_paths:
        raise ValueError("Group observer overrides require at least one path.")
    spec = policy.quant_spec(bit_width)
    return {path: spec for path in override_paths}


def validate_group_observer_overrides(
    model: nn.Module,
    policy: GroupObserverPolicy,
    override_paths: Sequence[str],
) -> None:
    """Verify that every requested PTQConfig path constructed the policy observer."""
    expected = frozenset(override_paths)
    if not expected:
        raise ValueError("Observer override validation requires at least one path.")

    observed: dict[str, ObserverBase] = {}
    for site in iter_quantization_sites(model):
        fp_name = getattr(site.module, "fp_name", None)
        if not fp_name:
            continue
        path = f"{fp_name}.{site.observer_name}"
        if path in expected:
            if path in observed:
                raise RuntimeError(
                    f"Multiple quantization sites map to override path {path!r}."
                )
            observed[path] = site.observer

    missing = tuple(sorted(expected - observed.keys()))
    if missing:
        raise RuntimeError(
            f"Group observer overrides were not materialized for paths: {missing}."
        )
    wrong_types = tuple(
        sorted(
            path
            for path, observer in observed.items()
            if not isinstance(observer, policy.observer)
        )
    )
    if wrong_types:
        raise RuntimeError(
            f"Group observer policy {policy.name!r} produced unexpected observer "
            f"types at paths: {wrong_types}."
        )
    if policy.observer is PercentileObserver and policy.percentile is not None:
        mismatched = tuple(
            sorted(
                path
                for path, observer in observed.items()
                if not math.isclose(
                    float(observer.percentile),  # type: ignore[attr-defined]
                    policy.percentile,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
        )
        if mismatched:
            raise RuntimeError(
                f"Group observer policy {policy.name!r} has mismatched percentile "
                f"settings at paths: {mismatched}."
            )


def evaluate_internal_full(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    samples: Sequence[ModelInput],
    *,
    boundaries: QuantizationBoundaries,
    output_adapter: OutputAdapter | None = None,
) -> tuple[dict[str, dict[str, float | int | None]], int]:
    """Evaluate one candidate under E without an extra float-parity pass."""
    if not samples:
        raise ValueError("Group observer evaluation requires at least one sample.")
    selector = boundaries.selector_for(QuantizationProfile.INTERNAL_FULL)
    sites = tuple(iter_quantization_sites(candidate_model))
    enabled_site_count = sum(selector(site) for site in sites)
    if enabled_site_count == 0:
        raise ValueError("E:internal-full did not select any quantization sites.")

    with FakeQuantState(candidate_model) as state:
        state.set_all(False)
        state.set_where(selector, True)
        outputs = evaluate_models(
            reference_model,
            candidate_model,
            samples,
            output_adapter=output_adapter,
        )
    return _copy_outputs(outputs), enabled_site_count


def build_group_observer_sweep_result(
    *,
    group: ActivationSensitivityGroup,
    override_paths: Sequence[str],
    global_percentile: float,
    baseline_outputs: Mapping[str, Mapping[str, float | int | None]],
    baseline_site_count: int,
    evaluations: Sequence[EvaluatedGroupObserverPolicy],
    score_output: str,
) -> dict[str, object]:
    """Rank one group's observer overrides against the unchanged baseline."""
    if score_output not in baseline_outputs:
        raise KeyError(
            f"Unknown score output {score_output!r}; available outputs: "
            f"{tuple(baseline_outputs)}."
        )
    baseline_score = _mae(baseline_outputs, score_output)
    baseline_name = f"global_{_percentile_name(global_percentile)}"
    rows: list[dict[str, object]] = [
        {
            "name": baseline_name,
            "observer": "PercentileObserver",
            "percentile": global_percentile,
            "is_global_baseline": True,
            "enabled_site_count": baseline_site_count,
            "score": baseline_score,
            "score_improvement": 0.0,
            "regressor_mae_improvement": 0.0,
            "classifier_mae_improvement": 0.0,
            "outputs": _copy_outputs(baseline_outputs),
        }
    ]
    for evaluation in evaluations:
        if evaluation.enabled_site_count != baseline_site_count:
            raise RuntimeError(
                f"Observer override {evaluation.policy.name!r} changed the E site "
                f"count from {baseline_site_count} to "
                f"{evaluation.enabled_site_count}."
            )
        row = evaluation.policy.to_dict()
        score = _mae(evaluation.outputs, score_output)
        row.update(
            {
                "is_global_baseline": False,
                "enabled_site_count": evaluation.enabled_site_count,
                "score": score,
                "score_improvement": baseline_score - score,
                "regressor_mae_improvement": _mae(baseline_outputs, "regressors")
                - _mae(evaluation.outputs, "regressors"),
                "classifier_mae_improvement": _mae(baseline_outputs, "classifiers")
                - _mae(evaluation.outputs, "classifiers"),
                "outputs": _copy_outputs(evaluation.outputs),
            }
        )
        rows.append(row)

    ranked = sorted(rows, key=lambda row: float(cast(float, row["score"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["is_best"] = rank == 1

    value = group.to_dict()
    best = ranked[0]
    value.update(
        {
            "override_paths": list(override_paths),
            "score_output": score_output,
            "best_candidate": best["name"],
            "best_score_improvement": best["score_improvement"],
            "best_regressor_mae_improvement": best["regressor_mae_improvement"],
            "best_classifier_mae_improvement": best["classifier_mae_improvement"],
            "candidates": ranked,
        }
    )
    return value


def print_group_observer_sweep(
    *,
    dtype_name: str,
    global_percentile: float,
    baseline_outputs: Mapping[str, Mapping[str, float | int | None]],
    baseline_site_count: int,
    group_results: Sequence[Mapping[str, Any]],
    score_output: str,
) -> None:
    """Print candidate rankings for every independently overridden group."""
    print(f"\n{dtype_name.upper()} group-specific activation observer sweep")
    print(
        f"Global baseline P{global_percentile:g} E:internal-full: "
        f"REG_MAE={_mae(baseline_outputs, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(baseline_outputs, 'classifiers'):.6e}, "
        f"SITES={baseline_site_count}"
    )
    print(
        "Each candidate changes only the listed group; every other activation "
        f"uses P{global_percentile:g}. Rankings use {score_output} MAE."
    )
    for group_result in group_results:
        print(
            f"\n{group_result['group']} "
            f"({group_result['site_count']} activation sites, "
            f"{len(group_result['override_paths'])} override paths)"
        )
        print(
            f"{'candidate':25s} {'REG_MAE':>13s} {'GAIN_REG':>13s} "
            f"{'CLS_MAE':>13s} {'GAIN_CLS':>13s}"
        )
        for candidate in group_result["candidates"]:
            marker = "*" if candidate["is_best"] else " "
            outputs = candidate["outputs"]
            print(
                f"{marker}{str(candidate['name']):24s} "
                f"{_mae(outputs, 'regressors'):13.6e} "
                f"{float(candidate['regressor_mae_improvement']):13.6e} "
                f"{_mae(outputs, 'classifiers'):13.6e} "
                f"{float(candidate['classifier_mae_improvement']):13.6e}"
            )
    print("* lowest selected-output MAE within each group, including no override.")


def _activation_dtype_qscheme(bit_width: int) -> tuple[DType, QScheme]:
    if bit_width == 8:
        return DType.uint(8), QScheme.PER_TENSOR_ASYMM
    if bit_width == 16:
        return DType.int(16), QScheme.PER_TENSOR_SYMM
    raise ValueError(f"Expected bit width 8 or 16, but received {bit_width}.")


def _validate_percentile(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError(f"{name} must be finite and in (0, 100].")


def _percentile_name(percentile: float) -> str:
    return f"percentile_{percentile:g}".replace(".", "_")


def _mae(
    outputs: Mapping[str, Mapping[str, float | int | None]],
    output_name: str,
) -> float:
    value = outputs[output_name].get("mae")
    if not isinstance(value, (float, int)):
        raise TypeError(f"Output {output_name!r} does not contain numeric MAE.")
    return float(value)


def _copy_outputs(
    outputs: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    return {name: dict(metrics) for name, metrics in outputs.items()}
