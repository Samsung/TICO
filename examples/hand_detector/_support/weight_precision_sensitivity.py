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

"""W8 parameter sensitivity under the hand detector's P2 W8/A16 profile."""

from __future__ import annotations

import copy
import math
import re

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tico.quantization import convert as freeze_quantization, prepare
from tico.quantization.analysis import (
    evaluate_models,
    OutputAdapter,
    QuantizationGroup,
    QuantizationSensitivity,
    SensitivityMode,
    SensitivityPathResult,
    SensitivityResult,
    SiteSelector,
)
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, QuantSpec
from tico.quantization.wrapq.control import (
    FakeQuantState,
    iter_quantization_sites,
    QuantizationSite,
    SiteRole,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn

from examples.hand_detector._support.precision_ablation import (
    discover_output_observer_paths,
    PHASE1_PROFILES,
    summarize_precision_inventory,
    validate_profile_inventory,
)

# Reuse the semantic partition used by activation sensitivity so names remain
# stable across the two analyses.
# pylint: disable=protected-access
from examples.hand_detector._support.sensitivity import (
    _find_detector,
    _partition_operations,
)


MetricSummary = Mapping[str, Mapping[str, float | int | None]]
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_SUPPORTED_GRANULARITIES = frozenset({"semantic", "site"})


@dataclass(frozen=True)
class ParameterKindSummary:
    """Summarize one parameter kind inside a sensitivity group."""

    kind: str
    site_count: int
    element_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "site_count": self.site_count,
            "element_count": self.element_count,
        }


@dataclass(frozen=True)
class WeightSensitivityGroup:
    """Describe one semantic block or individual W8 parameter site."""

    group: QuantizationGroup
    granularity: str
    semantic_group: str
    block_kind: str
    operation_positions: tuple[int, ...]
    operation_indices: tuple[int, ...]
    operation_names: tuple[str, ...]
    site_paths: tuple[str, ...]
    parameter_element_count: int
    parameter_breakdown: tuple[ParameterKindSummary, ...]

    @property
    def name(self) -> str:
        return self.group.name

    @property
    def site_count(self) -> int:
        return len(self.site_paths)

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.name,
            "granularity": self.granularity,
            "semantic_group": self.semantic_group,
            "kind": self.block_kind,
            "operation_positions": list(self.operation_positions),
            "operation_indices": list(self.operation_indices),
            "operation_names": list(self.operation_names),
            "site_count": self.site_count,
            "site_paths": list(self.site_paths),
            "parameter_element_count": self.parameter_element_count,
            "parameter_breakdown": [
                value.to_dict() for value in self.parameter_breakdown
            ],
        }


def run_weight_precision_sensitivity(
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
    requested_groups: Sequence[str] | None,
    granularity: str,
    run_greedy: bool,
    greedy_include_all_groups: bool,
    greedy_candidate_count: int,
    max_greedy_steps: int,
    minimum_improvement: float,
    auxiliary_tolerance: float,
    target_regressor_mae: float,
    output_adapter: OutputAdapter,
) -> dict[str, Any]:
    """Run independent ranking and an optional constrained greedy path."""
    _validate_run_arguments(
        evaluation_samples,
        greedy_candidate_count=greedy_candidate_count,
        max_greedy_steps=max_greedy_steps,
        minimum_improvement=minimum_improvement,
        auxiliary_tolerance=auxiliary_tolerance,
        target_regressor_mae=target_regressor_mae,
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
    all_groups = build_weight_sensitivity_groups(
        candidate,
        granularity=granularity,
    )
    groups = select_weight_sensitivity_groups(all_groups, requested_groups)
    site_count, element_count = parameter_totals(groups)

    runner = QuantizationSensitivity(
        float_model,
        candidate,
        output_adapter=output_adapter,
    )
    baseline, independent_results = runner.run(
        evaluation_samples,
        tuple(group.group for group in groups),
        mode=SensitivityMode.LEAVE_ONE_FLOAT,
        score_output="regressors",
        score_metric="mae",
        baseline_selector=SiteSelector.all(),
    )
    independent = build_independent_report(
        baseline=baseline,
        results=independent_results,
        groups=groups,
        target_regressor_mae=target_regressor_mae,
        auxiliary_tolerance=auxiliary_tolerance,
    )

    payload: dict[str, Any] = {
        "analysis": "w8a16_weight_precision_sensitivity",
        "metadata": {
            **p2_metadata,
            "granularity": granularity,
            "requested_groups": (
                list(requested_groups) if requested_groups is not None else None
            ),
            "group_count": len(groups),
            "parameter_site_count": site_count,
            "parameter_element_count": element_count,
            "greedy_enabled": run_greedy,
            "greedy_include_all_groups": greedy_include_all_groups,
            "greedy_candidate_count": greedy_candidate_count,
            "max_greedy_steps": max_greedy_steps,
            "minimum_improvement": minimum_improvement,
            "auxiliary_output": "classifiers",
            "auxiliary_tolerance": auxiliary_tolerance,
            "target_regressor_mae": target_regressor_mae,
        },
        "baseline": _copy_outputs(baseline),
        "group_definitions": [group.to_dict() for group in groups],
        "independent": independent,
    }

    if run_greedy:
        if greedy_include_all_groups:
            eligible_names = [str(value["group"]) for value in independent]
        else:
            eligible_names = [
                str(value["group"])
                for value in independent
                if bool(value["eligible"])
                and float(value["regressor_mae_improvement"]) > 0.0
            ]
        if greedy_candidate_count > 0:
            eligible_names = eligible_names[:greedy_candidate_count]
        greedy_groups = select_weight_sensitivity_groups(groups, eligible_names)
        greedy_results = _run_constrained_greedy(
            float_model,
            candidate,
            evaluation_samples,
            greedy_groups,
            output_adapter=output_adapter,
            max_steps=max_greedy_steps,
            minimum_improvement=minimum_improvement,
            auxiliary_tolerance=auxiliary_tolerance,
            target_regressor_mae=target_regressor_mae,
        )
        greedy_path, greedy_summary = build_path_report(
            baseline=baseline,
            results=greedy_results,
            groups=greedy_groups,
            target_regressor_mae=target_regressor_mae,
        )
        payload["greedy"] = {
            "candidate_groups": [group.name for group in greedy_groups],
            "steps": greedy_path,
            "summary": greedy_summary,
        }
    return payload


def build_w8a16_candidate(
    float_model: nn.Module,
    calibration_samples: Sequence[torch.Tensor],
    *,
    uint8_percentile: float,
    int16_observer: str,
    int16_percentile: float,
    max_samples: int,
    samples_per_batch: int,
    sampling_seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the P2 baseline: W8, internal A16, reg-I16, and cls-U8."""
    _validate_candidate_arguments(
        calibration_samples,
        uint8_percentile=uint8_percentile,
        int16_observer=int16_observer,
        int16_percentile=int16_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
    )
    uint8_spec = _uint8_activation_spec(
        percentile=uint8_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        seed=sampling_seed,
    )
    int16_spec = _int16_activation_spec(
        observer=int16_observer,
        percentile=int16_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        seed=sampling_seed,
    )
    output_paths = discover_output_observer_paths(float_model, uint8_spec)
    candidate = copy.deepcopy(float_model).eval()
    candidate = prepare(
        candidate,
        PTQConfig(
            activation=int16_spec,
            weight=_uint8_weight_spec(),
            overrides={
                output_paths.regressors: int16_spec,
                output_paths.classifiers: uint8_spec,
            },
            strict_wrap=False,
        ),
        inplace=True,
    ).eval()
    _calibrate(candidate, calibration_samples)
    candidate = freeze_quantization(candidate, inplace=True).eval()

    inventory = summarize_precision_inventory(candidate, output_paths)
    p2 = next(profile for profile in PHASE1_PROFILES if profile.name == "P2")
    validate_profile_inventory(p2, inventory)
    return candidate, {
        "profile": p2.to_dict(),
        "output_override_paths": output_paths.to_dict(),
        "inventory": inventory,
        "uint8_percentile": uint8_percentile,
        "int16_observer": int16_observer,
        "int16_percentile": (
            int16_percentile if int16_observer == "percentile" else None
        ),
        "max_samples": max_samples,
        "samples_per_batch": samples_per_batch,
        "sampling_seed": sampling_seed,
    }


def build_weight_sensitivity_groups(
    model: nn.Module,
    *,
    granularity: str = "semantic",
) -> tuple[WeightSensitivityGroup, ...]:
    """Partition every W8 parameter observer into non-overlapping groups."""
    if granularity not in _SUPPORTED_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {tuple(sorted(_SUPPORTED_GRANULARITIES))}."
        )
    detector = _find_detector(model)
    operation_groups = _partition_operations(detector)
    operation_to_group = {
        position: group for group in operation_groups for position in group.positions
    }
    parameter_sites = tuple(
        site
        for site in iter_quantization_sites(model)
        if site.role is SiteRole.PARAMETER
    )
    if not parameter_sites:
        raise ValueError("No W8 parameter sites were found for sensitivity.")

    assigned: dict[str, list[tuple[QuantizationSite, int]]] = defaultdict(list)
    for site in parameter_sites:
        position = _site_position(site)
        operation_group = operation_to_group.get(position)
        if operation_group is None:
            raise RuntimeError(
                f"Parameter site {site.path!r} at layer {position} is not assigned "
                "to a semantic operation group."
            )
        assigned[operation_group.name].append((site, position))

    if granularity == "semantic":
        results = _build_semantic_groups(detector, operation_groups, assigned)
    else:
        results = _build_site_groups(detector, operation_groups, assigned)
    _validate_group_coverage(parameter_sites, results)
    return results


def select_weight_sensitivity_groups(
    groups: Sequence[WeightSensitivityGroup],
    names: Sequence[str] | None,
) -> tuple[WeightSensitivityGroup, ...]:
    """Resolve an optional ordered group-name subset without silent misses."""
    available = {group.name: group for group in groups}
    if names is None:
        return tuple(groups)
    requested = tuple(names)
    if not requested:
        return ()
    if len(set(requested)) != len(requested):
        raise ValueError("Weight sensitivity group names must be unique.")
    missing = tuple(name for name in requested if name not in available)
    if missing:
        raise KeyError(
            f"Unknown weight sensitivity groups: {missing}; available groups: "
            f"{tuple(available)}."
        )
    return tuple(available[name] for name in requested)


def build_independent_report(
    *,
    baseline: MetricSummary,
    results: Sequence[SensitivityResult],
    groups: Sequence[WeightSensitivityGroup],
    target_regressor_mae: float,
    auxiliary_tolerance: float = 0.0,
) -> list[dict[str, object]]:
    """Attach weight metadata and balanced eligibility to independent results."""
    metadata = {group.name: group for group in groups}
    baseline_cls = _mae(baseline, "classifiers")
    report: list[dict[str, object]] = []
    for rank, result in enumerate(results, start=1):
        group = metadata[result.group]
        reg_gain = _mae(baseline, "regressors") - _mae(
            result.outputs,
            "regressors",
        )
        cls_gain = baseline_cls - _mae(result.outputs, "classifiers")
        value = result.to_dict()
        value.update(group.to_dict())
        value.update(
            {
                "rank": rank,
                "regressor_mae_improvement": reg_gain,
                "classifier_mae_improvement": cls_gain,
                "eligible": cls_gain + auxiliary_tolerance >= 0.0,
                "regressor_target_reached": (
                    _mae(result.outputs, "regressors") < target_regressor_mae
                ),
                "regressor_gain_per_million_parameters": (
                    reg_gain * 1_000_000.0 / max(group.parameter_element_count, 1)
                ),
            }
        )
        report.append(value)
    return report


def build_path_report(
    *,
    baseline: MetricSummary,
    results: Sequence[SensitivityPathResult],
    groups: Sequence[WeightSensitivityGroup],
    target_regressor_mae: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Attach cumulative parameter counts and two-output gains to path steps."""
    metadata = {group.name: group for group in groups}
    baseline_reg = _mae(baseline, "regressors")
    baseline_cls = _mae(baseline, "classifiers")
    previous_reg = baseline_reg
    previous_cls = baseline_cls
    report: list[dict[str, object]] = []
    for result in results:
        group = metadata[result.group]
        selected = tuple(metadata[name] for name in result.selected_groups)
        reg = _mae(result.outputs, "regressors")
        cls = _mae(result.outputs, "classifiers")
        value = result.to_dict()
        value.update(group.to_dict())
        value.update(
            {
                "incremental_regressor_mae_improvement": previous_reg - reg,
                "cumulative_regressor_mae_improvement": baseline_reg - reg,
                "incremental_classifier_mae_improvement": previous_cls - cls,
                "cumulative_classifier_mae_improvement": baseline_cls - cls,
                "selected_parameter_element_count": sum(
                    item.parameter_element_count for item in selected
                ),
                "regressor_target_reached": reg < target_regressor_mae,
            }
        )
        report.append(value)
        previous_reg = reg
        previous_cls = cls

    target_steps = tuple(
        int(value["step"])
        for value in report
        if bool(value["regressor_target_reached"])
    )
    if report:
        best = min(report, key=lambda value: _mae(value["outputs"], "regressors"))
        best_step = int(best["step"])
        best_regressor_mae = _mae(best["outputs"], "regressors")
    else:
        best_step = 0
        best_regressor_mae = baseline_reg
    summary = {
        "target_regressor_mae": target_regressor_mae,
        "target_reached": bool(target_steps),
        "first_target_step": target_steps[0] if target_steps else None,
        "best_step": best_step,
        "best_regressor_mae": best_regressor_mae,
        "final_regressor_mae": (
            _mae(report[-1]["outputs"], "regressors") if report else baseline_reg
        ),
        "final_classifier_mae": (
            _mae(report[-1]["outputs"], "classifiers") if report else baseline_cls
        ),
    }
    return report, summary


def print_weight_precision_sensitivity(
    report: Mapping[str, Any],
    *,
    top_k: int,
) -> None:
    """Print independent ranking and an optional greedy path."""
    baseline = report["baseline"]
    metadata = report["metadata"]
    independent = report["independent"]
    _print_independent_report(
        baseline=baseline,
        results=independent,
        top_k=top_k,
        parameter_site_count=int(metadata["parameter_site_count"]),
        parameter_element_count=int(metadata["parameter_element_count"]),
    )
    greedy = report.get("greedy")
    if isinstance(greedy, Mapping):
        _print_path_report(
            baseline=baseline,
            steps=greedy["steps"],
            summary=greedy["summary"],
        )


def parameter_totals(groups: Sequence[WeightSensitivityGroup]) -> tuple[int, int]:
    """Return non-overlapping site and element totals for a group collection."""
    return (
        sum(group.site_count for group in groups),
        sum(group.parameter_element_count for group in groups),
    )


def _run_constrained_greedy(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    samples: Sequence[torch.Tensor],
    groups: Sequence[WeightSensitivityGroup],
    *,
    output_adapter: OutputAdapter,
    max_steps: int,
    minimum_improvement: float,
    auxiliary_tolerance: float,
    target_regressor_mae: float,
) -> list[SensitivityPathResult]:
    if not groups:
        return []
    sites = tuple(iter_quantization_sites(candidate_model))
    by_name = {group.name: group for group in groups}
    remaining = list(groups)
    selected: list[str] = []
    selected_paths: set[str] = set()
    results: list[SensitivityPathResult] = []
    step_limit = len(groups) if max_steps == 0 else min(max_steps, len(groups))

    with FakeQuantState(candidate_model) as state:
        _apply_weight_baseline(state)
        current_outputs = evaluate_models(
            reference_model,
            candidate_model,
            samples,
            output_adapter=output_adapter,
        )
        current_reg = _mae(current_outputs, "regressors")
        current_cls = _mae(current_outputs, "classifiers")
        baseline_reg = current_reg

        for step in range(1, step_limit + 1):
            best_group: WeightSensitivityGroup | None = None
            best_outputs: dict[str, dict[str, float | int | None]] = {}
            best_improvement = float("-inf")
            best_cls = float("inf")
            for group in remaining:
                candidate_paths = selected_paths.union(group.site_paths)
                _apply_weight_baseline(state)
                state.set_where(
                    _path_selector(candidate_paths, "greedy_candidate"),
                    False,
                )
                outputs = evaluate_models(
                    reference_model,
                    candidate_model,
                    samples,
                    output_adapter=output_adapter,
                )
                reg = _mae(outputs, "regressors")
                cls = _mae(outputs, "classifiers")
                improvement = current_reg - reg
                if cls > current_cls + auxiliary_tolerance:
                    continue
                if best_group is None or improvement > best_improvement:
                    best_group = group
                    best_outputs = outputs
                    best_improvement = improvement
                    best_cls = cls

            if best_group is None or best_improvement <= minimum_improvement:
                break
            group = best_group
            outputs = best_outputs
            improvement = best_improvement
            cls = best_cls
            selected.append(group.name)
            selected_paths.update(group.site_paths)
            reg = _mae(outputs, "regressors")
            selected_sites = tuple(
                sorted(site.path for site in sites if site.path in selected_paths)
            )
            results.append(
                SensitivityPathResult(
                    step=step,
                    group=group.name,
                    selected_groups=tuple(selected),
                    outputs=_copy_outputs(outputs),
                    score=reg,
                    cumulative_sensitivity=baseline_reg - reg,
                    incremental_sensitivity=improvement,
                    matched_sites=tuple(group.site_paths),
                    selected_sites=selected_sites,
                )
            )
            current_reg = reg
            current_cls = cls
            remaining.remove(by_name[group.name])
            if current_reg < target_regressor_mae:
                break
    return results


def _apply_weight_baseline(state: FakeQuantState) -> None:
    state.set_all(True)


def _path_selector(paths: Sequence[str] | set[str], name: str) -> SiteSelector:
    selected = frozenset(paths)
    return SiteSelector(
        lambda site, values=selected: site.path in values,
        f"weight_paths[{name}]",
    )


def _build_semantic_groups(
    detector: Any,
    operation_groups: Sequence[Any],
    assigned: Mapping[str, Sequence[tuple[QuantizationSite, int]]],
) -> tuple[WeightSensitivityGroup, ...]:
    results: list[WeightSensitivityGroup] = []
    for operation_group in operation_groups:
        grouped = tuple(assigned.get(operation_group.name, ()))
        if not grouped:
            continue
        sites = tuple(site for site, _ in grouped)
        paths = tuple(sorted(site.path for site in sites))
        operations = tuple(
            detector.operations[position] for position in operation_group.positions
        )
        results.append(
            WeightSensitivityGroup(
                group=QuantizationGroup(
                    operation_group.name,
                    _path_selector(paths, operation_group.name),
                ),
                granularity="semantic",
                semantic_group=operation_group.name,
                block_kind=operation_group.kind,
                operation_positions=operation_group.positions,
                operation_indices=tuple(int(op["index"]) for op in operations),
                operation_names=tuple(str(op["name"]) for op in operations),
                site_paths=paths,
                parameter_element_count=sum(
                    _parameter_tensor(site).numel() for site in sites
                ),
                parameter_breakdown=_parameter_breakdown(sites),
            )
        )
    return tuple(results)


def _build_site_groups(
    detector: Any,
    operation_groups: Sequence[Any],
    assigned: Mapping[str, Sequence[tuple[QuantizationSite, int]]],
) -> tuple[WeightSensitivityGroup, ...]:
    operation_group_by_name = {group.name: group for group in operation_groups}
    flattened: list[tuple[int, str, QuantizationSite]] = []
    for semantic_name, values in assigned.items():
        for site, position in values:
            flattened.append((position, semantic_name, site))
    flattened.sort(key=lambda value: (value[0], value[2].path))

    results: list[WeightSensitivityGroup] = []
    used_names: set[str] = set()
    for position, semantic_name, site in flattened:
        operation_group = operation_group_by_name[semantic_name]
        kind = _parameter_kind(site)
        base_name = f"layer_{position:03d}_{kind}"
        name = base_name
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base_name}_{suffix}"
        used_names.add(name)
        operation = detector.operations[position]
        paths = (site.path,)
        results.append(
            WeightSensitivityGroup(
                group=QuantizationGroup(name, _path_selector(paths, name)),
                granularity="site",
                semantic_group=semantic_name,
                block_kind=operation_group.kind,
                operation_positions=(position,),
                operation_indices=(int(operation["index"]),),
                operation_names=(str(operation["name"]),),
                site_paths=paths,
                parameter_element_count=_parameter_tensor(site).numel(),
                parameter_breakdown=_parameter_breakdown((site,)),
            )
        )
    return tuple(results)


def _parameter_breakdown(
    sites: Sequence[QuantizationSite],
) -> tuple[ParameterKindSummary, ...]:
    counts: dict[str, list[int]] = {}
    for site in sites:
        kind = _parameter_kind(site)
        values = counts.setdefault(kind, [0, 0])
        values[0] += 1
        values[1] += _parameter_tensor(site).numel()
    return tuple(
        ParameterKindSummary(kind, values[0], values[1])
        for kind, values in sorted(counts.items())
    )


def _parameter_kind(site: QuantizationSite) -> str:
    wrapped = getattr(site.module, "module", None)
    if isinstance(wrapped, nn.PReLU):
        return "prelu_slope"
    if isinstance(wrapped, nn.Conv2d):
        if (
            wrapped.groups == wrapped.in_channels
            and wrapped.out_channels % wrapped.in_channels == 0
        ):
            return "depthwise_conv2d_weight"
        return "conv2d_weight"
    return f"{type(wrapped).__name__.lower()}_weight"


def _parameter_tensor(site: QuantizationSite) -> torch.Tensor:
    wrapped = getattr(site.module, "module", None)
    weight = getattr(wrapped, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"Parameter site {site.path!r} does not expose module.weight Tensor."
        )
    return weight


def _site_position(site: QuantizationSite) -> int:
    module_name = getattr(site.module, "fp_name", None) or site.module_path
    match = _LAYER_PATTERN.search(module_name)
    if match is None:
        raise RuntimeError(
            f"Cannot map parameter site {site.path!r} to a detector layer."
        )
    return int(match.group(1))


def _validate_group_coverage(
    sites: Sequence[QuantizationSite],
    groups: Sequence[WeightSensitivityGroup],
) -> None:
    expected = tuple(sorted(site.path for site in sites))
    actual = tuple(sorted(path for group in groups for path in group.site_paths))
    if actual != expected:
        missing = tuple(sorted(set(expected) - set(actual)))
        duplicated = tuple(
            sorted(path for path in set(actual) if actual.count(path) > 1)
        )
        raise RuntimeError(
            "Weight sensitivity groups must cover every parameter site exactly "
            f"once; missing={missing}, duplicated={duplicated}."
        )


def _uint8_weight_spec() -> QuantSpec:
    return affine(
        DType.uint(8),
        qscheme=QScheme.PER_CHANNEL_ASYMM,
        observer=MinMaxObserver,
    )


def _uint8_activation_spec(
    *,
    percentile: float,
    max_samples: int,
    samples_per_batch: int,
    seed: int,
) -> QuantSpec:
    return affine(
        DType.uint(8),
        qscheme=QScheme.PER_TENSOR_ASYMM,
        observer=PercentileObserver,
        percentile=percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        seed=seed,
    )


def _int16_activation_spec(
    *,
    observer: str,
    percentile: float,
    max_samples: int,
    samples_per_batch: int,
    seed: int,
) -> QuantSpec:
    if observer == "minmax":
        return affine(
            DType.int(16),
            qscheme=QScheme.PER_TENSOR_SYMM,
            observer=MinMaxObserver,
        )
    if observer == "percentile":
        return affine(
            DType.int(16),
            qscheme=QScheme.PER_TENSOR_SYMM,
            observer=PercentileObserver,
            percentile=percentile,
            max_samples=max_samples,
            samples_per_batch=samples_per_batch,
            seed=seed,
        )
    raise ValueError("int16_observer must be 'minmax' or 'percentile'.")


def _calibrate(model: nn.Module, samples: Sequence[torch.Tensor]) -> None:
    model.eval()
    # Observer implementations may replace registered statistics buffers
    # while collecting. Tensors created under inference_mode become
    # inference tensors, which later reject load_state_dict's in-place
    # copies outside inference mode. no_grad avoids autograd overhead
    # without changing the mutability contract of observer buffers.
    with torch.no_grad():
        for sample in samples:
            model(sample)


def _validate_candidate_arguments(
    samples: Sequence[torch.Tensor],
    *,
    uint8_percentile: float,
    int16_observer: str,
    int16_percentile: float,
    max_samples: int,
    samples_per_batch: int,
) -> None:
    if not samples:
        raise ValueError("Weight sensitivity requires calibration samples.")
    for name, value in (
        ("uint8_percentile", uint8_percentile),
        ("int16_percentile", int16_percentile),
    ):
        if not math.isfinite(value) or not 0.0 < value <= 100.0:
            raise ValueError(f"{name} must be finite and in (0, 100].")
    if int16_observer not in {"minmax", "percentile"}:
        raise ValueError("int16_observer must be 'minmax' or 'percentile'.")
    if max_samples <= 0 or samples_per_batch <= 0:
        raise ValueError("Observer sample limits must be positive.")


def _validate_run_arguments(
    samples: Sequence[torch.Tensor],
    *,
    greedy_candidate_count: int,
    max_greedy_steps: int,
    minimum_improvement: float,
    auxiliary_tolerance: float,
    target_regressor_mae: float,
) -> None:
    if not samples:
        raise ValueError("Weight sensitivity requires evaluation samples.")
    if greedy_candidate_count < 0 or max_greedy_steps < 0:
        raise ValueError("Greedy candidate and step counts must be nonnegative.")
    for name, value in (
        ("minimum_improvement", minimum_improvement),
        ("auxiliary_tolerance", auxiliary_tolerance),
        ("target_regressor_mae", target_regressor_mae),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be nonnegative.")
    if auxiliary_tolerance < 0.0:
        raise ValueError("auxiliary_tolerance must be nonnegative.")
    if target_regressor_mae <= 0.0:
        raise ValueError("target_regressor_mae must be positive.")


def _print_independent_report(
    *,
    baseline: MetricSummary,
    results: Sequence[Mapping[str, Any]],
    top_k: int,
    parameter_site_count: int,
    parameter_element_count: int,
) -> None:
    if top_k < 0:
        raise ValueError("top_k must be nonnegative.")
    shown = results if top_k == 0 else results[:top_k]
    print("\nW8/A16 weight precision sensitivity")
    print(
        "Baseline P2: "
        f"REG_MAE={_mae(baseline, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(baseline, 'classifiers'):.6e}, "
        f"WEIGHT_SITES={parameter_site_count}, "
        f"WEIGHT_ELEMENTS={parameter_element_count}"
    )
    print("Groups are ranked by REG MAE improvement when their weights stay FP32.")
    print(
        f"{'rank':>4s} {'group':34s} {'REG_MAE':>13s} {'GAIN_REG':>13s} "
        f"{'CLS_MAE':>13s} {'GAIN_CLS':>13s} {'PARAMS':>11s} "
        f"{'SITES':>7s} {'OK':>3s}"
    )
    for result in shown:
        print(
            f"{int(result['rank']):4d} "
            f"{str(result['group'])[:34]:34s} "
            f"{_mae(result['outputs'], 'regressors'):13.6e} "
            f"{float(result['regressor_mae_improvement']):13.6e} "
            f"{_mae(result['outputs'], 'classifiers'):13.6e} "
            f"{float(result['classifier_mae_improvement']):13.6e} "
            f"{int(result['parameter_element_count']):11d} "
            f"{int(result['site_count']):7d} "
            f"{('yes' if result['eligible'] else 'no'):>3s}"
        )
    if top_k and len(results) > top_k:
        print(f"Showing {top_k} of {len(results)} groups; JSON contains all groups.")


def _print_path_report(
    *,
    baseline: MetricSummary,
    steps: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    print("\nW8/A16 weight constrained greedy path")
    print(
        "Baseline P2: "
        f"REG_MAE={_mae(baseline, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(baseline, 'classifiers'):.6e}"
    )
    print(
        f"{'step':>4s} {'added_group':34s} {'REG_MAE':>13s} "
        f"{'DELTA_REG':>13s} {'TOTAL_REG':>13s} {'CLS_MAE':>13s} "
        f"{'DELTA_CLS':>13s} {'TOTAL_CLS':>13s} {'FP_PARAMS':>11s}"
    )
    for step in steps:
        print(
            f"{int(step['step']):4d} "
            f"{str(step['group'])[:34]:34s} "
            f"{_mae(step['outputs'], 'regressors'):13.6e} "
            f"{float(step['incremental_regressor_mae_improvement']):13.6e} "
            f"{float(step['cumulative_regressor_mae_improvement']):13.6e} "
            f"{_mae(step['outputs'], 'classifiers'):13.6e} "
            f"{float(step['incremental_classifier_mae_improvement']):13.6e} "
            f"{float(step['cumulative_classifier_mae_improvement']):13.6e} "
            f"{int(step['selected_parameter_element_count']):11d}"
        )
    if bool(summary["target_reached"]):
        print(
            "Target reached at step "
            f"{int(summary['first_target_step'])}: "
            f"REG_MAE<{float(summary['target_regressor_mae']):g}."
        )
    else:
        print(
            "Target not reached; best REG_MAE="
            f"{float(summary['best_regressor_mae']):.6e}."
        )


def _copy_outputs(outputs: MetricSummary) -> dict[str, dict[str, Any]]:
    return {name: dict(metrics) for name, metrics in outputs.items()}


def _mae(outputs: MetricSummary, output_name: str) -> float:
    value = outputs[output_name]["mae"]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Output {output_name!r} MAE is not numeric.")
    return float(value)


__all__ = [
    "ParameterKindSummary",
    "WeightSensitivityGroup",
    "build_independent_report",
    "build_path_report",
    "build_w8a16_candidate",
    "build_weight_sensitivity_groups",
    "parameter_totals",
    "print_weight_precision_sensitivity",
    "run_weight_precision_sensitivity",
    "select_weight_sensitivity_groups",
]
