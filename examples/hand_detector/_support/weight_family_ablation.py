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

"""Parameter-family numerical floors under the hand detector's P2 profile."""

from __future__ import annotations

import math

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from examples.hand_detector._support.weight_precision_sensitivity import (
    build_w8a16_candidate,
    build_weight_sensitivity_groups,
    WeightSensitivityGroup,
)
from tico.quantization.analysis import evaluate_models, OutputAdapter, SiteSelector
from tico.quantization.wrapq.control import FakeQuantState
from torch import nn


MetricSummary = Mapping[str, Mapping[str, float | int | None]]
_FAMILY_ORDER = ("regular_conv", "depthwise_conv", "prelu_slope")
_KIND_TO_FAMILY = {
    "conv2d_weight": "regular_conv",
    "depthwise_conv2d_weight": "depthwise_conv",
    "prelu_slope": "prelu_slope",
}


@dataclass(frozen=True)
class WeightFamily:
    """Describe one complete, non-overlapping parameter family."""

    name: str
    site_paths: tuple[str, ...]
    site_count: int
    parameter_element_count: int
    parameter_kinds: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.name,
            "site_count": self.site_count,
            "site_paths": list(self.site_paths),
            "parameter_element_count": self.parameter_element_count,
            "parameter_kinds": list(self.parameter_kinds),
        }


@dataclass(frozen=True)
class WeightFamilyProfile:
    """Describe which parameter families stay floating point."""

    name: str
    float_families: tuple[str, ...]
    explanation: str

    def to_dict(self) -> dict[str, object]:
        float_set = frozenset(self.float_families)
        return {
            "name": self.name,
            "regular_conv": ("float" if "regular_conv" in float_set else "uint8"),
            "depthwise_conv": ("float" if "depthwise_conv" in float_set else "uint8"),
            "prelu_slope": ("float" if "prelu_slope" in float_set else "uint8"),
            "float_families": list(self.float_families),
            "explanation": self.explanation,
        }


WEIGHT_FAMILY_PROFILES = (
    WeightFamilyProfile(
        name="F0",
        float_families=(),
        explanation="P2 baseline: every parameter family uses nearest W8.",
    ),
    WeightFamilyProfile(
        name="F1",
        float_families=("regular_conv",),
        explanation="Keep regular/pointwise and output-head Conv2d weights FP32.",
    ),
    WeightFamilyProfile(
        name="F2",
        float_families=("depthwise_conv",),
        explanation="Keep only depthwise Conv2d weights FP32.",
    ),
    WeightFamilyProfile(
        name="F3",
        float_families=("prelu_slope",),
        explanation="Keep only channel-wise PReLU slopes FP32.",
    ),
    WeightFamilyProfile(
        name="F4",
        float_families=("regular_conv", "depthwise_conv"),
        explanation="Keep every Conv2d weight FP32 while PReLU remains W8.",
    ),
)


def run_weight_family_ablation(
    float_model: nn.Module,
    calibration_samples: Sequence[torch.Tensor],
    evaluation_samples: Sequence[torch.Tensor],
    *,
    uint8_percentile: float,
    int16_observer: str,
    int16_percentile: float,
    max_samples: int,
    samples_per_batch: int,
    sampling_seed: int,
    target_regressor_mae: float,
    target_classifier_mae: float,
    output_adapter: OutputAdapter,
) -> dict[str, Any]:
    """Evaluate F0-F4 and an all-floating P3 parameter reference."""
    _validate_arguments(
        calibration_samples,
        evaluation_samples,
        target_regressor_mae=target_regressor_mae,
        target_classifier_mae=target_classifier_mae,
    )
    candidate, p2_metadata = build_w8a16_candidate(
        float_model,
        calibration_samples,
        uint8_percentile=uint8_percentile,
        int16_observer=int16_observer,
        int16_percentile=int16_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        sampling_seed=sampling_seed,
    )
    families = build_weight_families(candidate)
    family_by_name = {family.name: family for family in families}
    full_parameter_count = sum(family.parameter_element_count for family in families)

    with _WeightFamilyEvaluator(
        float_model,
        candidate,
        evaluation_samples,
        families,
        output_adapter=output_adapter,
    ) as evaluator:
        p3_outputs = evaluator.evaluate(frozenset(_FAMILY_ORDER))
        raw_results: list[tuple[WeightFamilyProfile, MetricSummary]] = []
        for profile in WEIGHT_FAMILY_PROFILES:
            raw_results.append(
                (
                    profile,
                    evaluator.evaluate(frozenset(profile.float_families)),
                )
            )

    f0_outputs = next(
        outputs for profile, outputs in raw_results if profile.name == "F0"
    )
    profiles = [
        _build_profile_result(
            profile,
            outputs,
            f0_outputs=f0_outputs,
            p3_outputs=p3_outputs,
            family_by_name=family_by_name,
            full_parameter_count=full_parameter_count,
            target_regressor_mae=target_regressor_mae,
            target_classifier_mae=target_classifier_mae,
        )
        for profile, outputs in raw_results
    ]
    summary = build_weight_family_summary(
        profiles,
        p3_outputs=p3_outputs,
        target_regressor_mae=target_regressor_mae,
        target_classifier_mae=target_classifier_mae,
    )
    return {
        "analysis": "w8a16_weight_family_ablation",
        "metadata": {
            **p2_metadata,
            "target_regressor_mae": target_regressor_mae,
            "target_classifier_mae": target_classifier_mae,
            "family_count": len(families),
            "parameter_site_count": sum(family.site_count for family in families),
            "parameter_element_count": full_parameter_count,
            "profile_count": len(WEIGHT_FAMILY_PROFILES),
        },
        "family_definitions": [family.to_dict() for family in families],
        "p3_all_float_reference": {
            "outputs": _copy_outputs(p3_outputs),
            "target_feasible": _target_feasible(
                p3_outputs,
                target_regressor_mae,
                target_classifier_mae,
            ),
        },
        "profiles": profiles,
        "summary": summary,
        "recommendation": _recommendation(summary),
    }


def build_weight_families(model: nn.Module) -> tuple[WeightFamily, ...]:
    """Aggregate site-level groups into Conv, depthwise, and PReLU families."""
    site_groups = build_weight_sensitivity_groups(model, granularity="site")
    if not site_groups:
        raise ValueError("Weight-family ablation requires parameter sites.")

    grouped: dict[str, list[WeightSensitivityGroup]] = defaultdict(list)
    for group in site_groups:
        if group.site_count != 1 or len(group.parameter_breakdown) != 1:
            raise RuntimeError(
                "Site-level weight groups must contain exactly one parameter site."
            )
        kind = group.parameter_breakdown[0].kind
        family = _KIND_TO_FAMILY.get(kind)
        if family is None:
            raise ValueError(
                f"Unsupported hand-detector parameter kind {kind!r} at "
                f"{group.name!r}."
            )
        grouped[family].append(group)

    missing = tuple(name for name in _FAMILY_ORDER if not grouped.get(name))
    if missing:
        raise RuntimeError(
            f"Weight-family ablation is missing expected families: {missing}."
        )

    families: list[WeightFamily] = []
    all_paths: list[str] = []
    for name in _FAMILY_ORDER:
        members = grouped[name]
        paths = tuple(path for member in members for path in member.site_paths)
        kinds = tuple(
            sorted(
                {item.kind for member in members for item in member.parameter_breakdown}
            )
        )
        families.append(
            WeightFamily(
                name=name,
                site_paths=paths,
                site_count=sum(member.site_count for member in members),
                parameter_element_count=sum(
                    member.parameter_element_count for member in members
                ),
                parameter_kinds=kinds,
            )
        )
        all_paths.extend(paths)

    expected = tuple(sorted(path for group in site_groups for path in group.site_paths))
    actual = tuple(sorted(all_paths))
    if expected != actual or len(actual) != len(set(actual)):
        raise RuntimeError(
            "Weight families must cover every parameter site exactly once."
        )
    return tuple(families)


def build_weight_family_summary(
    profiles: Sequence[Mapping[str, Any]],
    *,
    p3_outputs: MetricSummary,
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> dict[str, Any]:
    """Derive family gains, interactions, and the smallest feasible oracle."""
    by_name = {str(profile["name"]): profile for profile in profiles}
    expected = {profile.name for profile in WEIGHT_FAMILY_PROFILES}
    missing = tuple(sorted(expected.difference(by_name)))
    if missing:
        raise ValueError(f"Missing weight-family profile results: {missing}.")

    f0 = by_name["F0"]
    f1 = by_name["F1"]
    f2 = by_name["F2"]
    f3 = by_name["F3"]
    f4 = by_name["F4"]
    f0_reg = _mae(f0["outputs"], "regressors")
    regular_gain = f0_reg - _mae(f1["outputs"], "regressors")
    depthwise_gain = f0_reg - _mae(f2["outputs"], "regressors")
    prelu_gain = f0_reg - _mae(f3["outputs"], "regressors")
    all_conv_gain = f0_reg - _mae(f4["outputs"], "regressors")
    p3_reg = _mae(p3_outputs, "regressors")

    feasible = [profile for profile in profiles if bool(profile["target_feasible"])]
    feasible.sort(
        key=lambda profile: (
            int(profile["float_parameter_element_count"]),
            _mae(profile["outputs"], "regressors"),
            str(profile["name"]),
        )
    )
    best = feasible[0] if feasible else None
    return {
        "regular_conv_regressor_gain": regular_gain,
        "depthwise_conv_regressor_gain": depthwise_gain,
        "prelu_slope_regressor_gain": prelu_gain,
        "all_conv_regressor_gain": all_conv_gain,
        "regular_depthwise_interaction": (
            all_conv_gain - regular_gain - depthwise_gain
        ),
        "residual_regressor_gap_after_all_conv_float": (
            _mae(f4["outputs"], "regressors") - p3_reg
        ),
        "p3_regressor_mae": p3_reg,
        "p3_classifier_mae": _mae(p3_outputs, "classifiers"),
        "p3_target_feasible": _target_feasible(
            p3_outputs,
            target_regressor_mae,
            target_classifier_mae,
        ),
        "best_target_feasible_profile": (
            str(best["name"]) if best is not None else None
        ),
        "best_target_feasible_float_families": (
            list(best["float_families"]) if best is not None else None
        ),
        "best_target_feasible_float_parameter_ratio": (
            float(best["float_parameter_ratio"]) if best is not None else None
        ),
    }


def print_weight_family_ablation(report: Mapping[str, Any]) -> None:
    """Print family-oracle floors and interaction diagnostics."""
    p3 = report["p3_all_float_reference"]["outputs"]
    print("\nW8/A16 weight family ablation")
    print(
        "P3 all-weight-FP reference: "
        f"REG_MAE={_mae(p3, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(p3, 'classifiers'):.6e}"
    )
    print(
        f"{'profile':8s} {'REGULAR':>9s} {'DEPTHWISE':>10s} {'PRELU':>7s} "
        f"{'REG_MAE':>13s} {'GAIN_REG':>13s} {'CLS_MAE':>13s} "
        f"{'GAIN_CLS':>13s} {'FP_PARAMS':>11s} {'FP(%)':>8s} {'OK':>4s}"
    )
    for profile in report["profiles"]:
        outputs = profile["outputs"]
        print(
            f"{profile['name']:8s} "
            f"{profile['regular_conv']:>9s} "
            f"{profile['depthwise_conv']:>10s} "
            f"{profile['prelu_slope']:>7s} "
            f"{_mae(outputs, 'regressors'):13.6e} "
            f"{float(profile['regressor_mae_improvement_vs_f0']):13.6e} "
            f"{_mae(outputs, 'classifiers'):13.6e} "
            f"{float(profile['classifier_mae_improvement_vs_f0']):13.6e} "
            f"{int(profile['float_parameter_element_count']):11d} "
            f"{100.0 * float(profile['float_parameter_ratio']):8.3f} "
            f"{'yes' if profile['target_feasible'] else 'no':>4s}"
        )

    summary = report["summary"]
    print("\nDerived family effects relative to F0")
    print(
        "Regular Conv FP gain:       "
        f"{float(summary['regular_conv_regressor_gain']):.6e}"
    )
    print(
        "Depthwise Conv FP gain:     "
        f"{float(summary['depthwise_conv_regressor_gain']):.6e}"
    )
    print(
        "PReLU slope FP gain:        "
        f"{float(summary['prelu_slope_regressor_gain']):.6e}"
    )
    print(
        "All Conv FP gain:           "
        f"{float(summary['all_conv_regressor_gain']):.6e}"
    )
    print(
        "Regular/depthwise interaction: "
        f"{float(summary['regular_depthwise_interaction']):.6e}"
    )
    print(
        "Residual gap F4->P3:        "
        f"{float(summary['residual_regressor_gap_after_all_conv_float']):.6e}"
    )
    print("Recommendation: " + str(report["recommendation"]))


class _WeightFamilyEvaluator(AbstractContextManager["_WeightFamilyEvaluator"]):
    """Evaluate arbitrary floating-family subsets from one prepared P2 model."""

    def __init__(
        self,
        reference_model: nn.Module,
        candidate_model: nn.Module,
        samples: Sequence[torch.Tensor],
        families: Sequence[WeightFamily],
        *,
        output_adapter: OutputAdapter,
    ) -> None:
        self.reference_model = reference_model
        self.candidate_model = candidate_model
        self.samples = samples
        self.families = tuple(families)
        self.output_adapter = output_adapter
        self._family_paths = {
            family.name: frozenset(family.site_paths) for family in self.families
        }
        self._cache: dict[frozenset[str], dict[str, dict[str, float | int | None]]] = {}
        self._context = FakeQuantState(candidate_model)
        self._state: FakeQuantState | None = None

    def __enter__(self) -> "_WeightFamilyEvaluator":
        self._state = self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._context.__exit__(exc_type, exc_value, traceback)
        self._state = None
        return None

    def evaluate(
        self,
        float_families: frozenset[str],
    ) -> dict[str, dict[str, float | int | None]]:
        cached = self._cache.get(float_families)
        if cached is not None:
            return _copy_outputs(cached)
        if self._state is None:
            raise RuntimeError("Weight-family evaluator is not active.")
        unknown = tuple(sorted(set(float_families).difference(self._family_paths)))
        if unknown:
            raise KeyError(f"Unknown floating weight families: {unknown}.")

        self._state.set_all(True)
        float_paths = frozenset(
            path for name in float_families for path in self._family_paths[name]
        )
        if float_paths:
            self._state.set_where(
                SiteSelector(
                    lambda site, paths=float_paths: site.path in paths,
                    "weight_family_float_paths",
                ),
                False,
            )
        outputs = evaluate_models(
            self.reference_model,
            self.candidate_model,
            self.samples,
            output_adapter=self.output_adapter,
        )
        value = _copy_outputs(outputs)
        self._cache[float_families] = value
        return _copy_outputs(value)


def _build_profile_result(
    profile: WeightFamilyProfile,
    outputs: MetricSummary,
    *,
    f0_outputs: MetricSummary,
    p3_outputs: MetricSummary,
    family_by_name: Mapping[str, WeightFamily],
    full_parameter_count: int,
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> dict[str, Any]:
    value = profile.to_dict()
    float_count = sum(
        family_by_name[name].parameter_element_count for name in profile.float_families
    )
    quantized_count = full_parameter_count - float_count
    f0_reg = _mae(f0_outputs, "regressors")
    f0_cls = _mae(f0_outputs, "classifiers")
    denominator = f0_reg - _mae(p3_outputs, "regressors")
    reg_gain = f0_reg - _mae(outputs, "regressors")
    value.update(
        {
            "outputs": _copy_outputs(outputs),
            "regressor_mae_improvement_vs_f0": reg_gain,
            "classifier_mae_improvement_vs_f0": (f0_cls - _mae(outputs, "classifiers")),
            "regressor_gap_recovery_ratio": (
                reg_gain / denominator if denominator > 0.0 else 0.0
            ),
            "float_parameter_element_count": float_count,
            "float_parameter_ratio": float_count / max(full_parameter_count, 1),
            "quantized_parameter_element_count": quantized_count,
            "quantized_parameter_ratio": (
                quantized_count / max(full_parameter_count, 1)
            ),
            "target_feasible": _target_feasible(
                outputs,
                target_regressor_mae,
                target_classifier_mae,
            ),
        }
    )
    return value


def _recommendation(summary: Mapping[str, Any]) -> str:
    best = summary["best_target_feasible_profile"]
    if best == "F0":
        return "Nearest W8 already meets both targets; no family exemption is needed."
    if best == "F1":
        return (
            "The regular-Conv family oracle reaches the targets; prioritize broad "
            "regular/pointwise and output-head Conv2d W8 optimization."
        )
    if best == "F2":
        return (
            "The depthwise-only oracle reaches the targets; prioritize depthwise "
            "weight optimization."
        )
    if best == "F3":
        return (
            "The PReLU-only oracle reaches the targets; investigate channel-wise "
            "slope quantization."
        )
    if best == "F4":
        return (
            "Regular Conv alone is insufficient but all Conv weights reach the "
            "targets; optimize regular and depthwise Conv families together."
        )
    if bool(summary["p3_target_feasible"]):
        return (
            "No tested family oracle reaches the targets although all-FP weights "
            "do; broad joint parameter optimization is required."
        )
    return (
        "Even all-FP weights miss at least one target under the current A16/output "
        "policy; revisit activation or output precision."
    )


def _target_feasible(
    outputs: MetricSummary,
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> bool:
    return (
        _mae(outputs, "regressors") < target_regressor_mae
        and _mae(outputs, "classifiers") < target_classifier_mae
    )


def _mae(outputs: MetricSummary, name: str) -> float:
    value = outputs[name]["mae"]
    if value is None:
        raise ValueError(f"Output {name!r} does not contain MAE.")
    return float(value)


def _copy_outputs(outputs: MetricSummary) -> dict[str, dict[str, float | int | None]]:
    return {name: dict(metrics) for name, metrics in outputs.items()}


def _validate_arguments(
    calibration_samples: Sequence[torch.Tensor],
    evaluation_samples: Sequence[torch.Tensor],
    *,
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> None:
    if not calibration_samples or not evaluation_samples:
        raise ValueError(
            "Weight-family ablation requires calibration and evaluation data."
        )
    for name, value in (
        ("target_regressor_mae", target_regressor_mae),
        ("target_classifier_mae", target_classifier_mae),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
