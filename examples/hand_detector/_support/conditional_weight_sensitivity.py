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

"""Conditional Conv-family W8 sensitivity under the P2 W8/A16 profile."""

from __future__ import annotations

import math

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
from tico.quantization.analysis import evaluate_models, OutputAdapter, SiteSelector
from tico.quantization.wrapq.control import FakeQuantState
from torch import nn

from examples.hand_detector._support.weight_precision_sensitivity import (
    build_w8a16_candidate,
    build_weight_sensitivity_groups,
    WeightSensitivityGroup,
)


MetricSummary = Mapping[str, Mapping[str, float | int | None]]
_SUPPORTED_GRANULARITIES = frozenset({"semantic", "site"})
_KIND_TO_FAMILY = {
    "conv2d_weight": "regular",
    "depthwise_conv2d_weight": "depthwise",
    "prelu_slope": "prelu",
}
_SCENARIOS = {
    "regular-float": {
        "baseline_profile": "F1",
        "baseline_float_family": "regular",
        "target_family": "depthwise",
    },
    "depthwise-float": {
        "baseline_profile": "F2",
        "baseline_float_family": "depthwise",
        "target_family": "regular",
    },
}


@dataclass(frozen=True)
class ConditionalWeightGroup:
    """Describe one target-family semantic block or individual parameter site."""

    name: str
    family: str
    granularity: str
    semantic_group: str
    block_kind: str
    operation_positions: tuple[int, ...]
    operation_indices: tuple[int, ...]
    operation_names: tuple[str, ...]
    site_paths: tuple[str, ...]
    parameter_element_count: int

    @property
    def site_count(self) -> int:
        """Return the number of parameter sites represented by this group."""
        return len(self.site_paths)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible group metadata."""
        return {
            "group": self.name,
            "family": self.family,
            "granularity": self.granularity,
            "semantic_group": self.semantic_group,
            "kind": self.block_kind,
            "operation_positions": list(self.operation_positions),
            "operation_indices": list(self.operation_indices),
            "operation_names": list(self.operation_names),
            "site_count": self.site_count,
            "site_paths": list(self.site_paths),
            "parameter_element_count": self.parameter_element_count,
        }


@dataclass(frozen=True)
class FamilyDefinition:
    """Describe all parameter sites assigned to one family."""

    name: str
    site_paths: tuple[str, ...]
    parameter_element_count: int

    @property
    def site_count(self) -> int:
        """Return the number of parameter sites in this family."""
        return len(self.site_paths)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible family metadata."""
        return {
            "family": self.name,
            "site_count": self.site_count,
            "site_paths": list(self.site_paths),
            "parameter_element_count": self.parameter_element_count,
        }


def run_conditional_weight_sensitivity(
    float_model: nn.Module,
    calibration_samples: Sequence[torch.Tensor],
    evaluation_samples: Sequence[torch.Tensor],
    *,
    baseline_family: str,
    uint8_percentile: float,
    int16_observer: str,
    int16_percentile: float,
    max_samples: int,
    samples_per_batch: int,
    sampling_seed: int,
    requested_groups: Sequence[str] | None,
    granularity: str,
    run_greedy: bool,
    max_greedy_steps: int,
    minimum_improvement: float,
    auxiliary_tolerance: float,
    target_regressor_mae: float,
    target_classifier_mae: float,
    output_adapter: OutputAdapter,
) -> dict[str, Any]:
    """Run independent and greedy sensitivity from F1 or F2.

    ``regular-float`` holds every regular Conv weight FP and diagnoses the
    remaining depthwise W8 error. ``depthwise-float`` does the complementary
    regular-Conv analysis. PReLU slopes stay W8 in both scenarios.
    """
    _validate_run_arguments(
        calibration_samples,
        evaluation_samples,
        baseline_family=baseline_family,
        granularity=granularity,
        max_greedy_steps=max_greedy_steps,
        minimum_improvement=minimum_improvement,
        auxiliary_tolerance=auxiliary_tolerance,
        target_regressor_mae=target_regressor_mae,
        target_classifier_mae=target_classifier_mae,
    )
    scenario = _SCENARIOS[baseline_family]
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
    definitions, all_groups = build_conditional_weight_groups(
        candidate,
        target_family=str(scenario["target_family"]),
        granularity=granularity,
    )
    groups = select_conditional_weight_groups(all_groups, requested_groups)
    if not groups:
        raise ValueError("Conditional weight sensitivity requires target groups.")

    baseline_definition = definitions[str(scenario["baseline_float_family"])]
    target_definition = definitions[str(scenario["target_family"])]
    full_parameter_count = sum(
        definition.parameter_element_count for definition in definitions.values()
    )

    with _ConditionalWeightEvaluator(
        float_model,
        candidate,
        evaluation_samples,
        baseline_float_paths=baseline_definition.site_paths,
        groups=groups,
        output_adapter=output_adapter,
    ) as evaluator:
        baseline = evaluator.evaluate(frozenset())
        endpoint = evaluator.evaluate(frozenset(group.name for group in groups))
        independent = _build_independent_report(
            groups,
            baseline,
            evaluator,
            auxiliary_tolerance=auxiliary_tolerance,
            target_regressor_mae=target_regressor_mae,
            target_classifier_mae=target_classifier_mae,
            full_parameter_count=full_parameter_count,
        )
        if run_greedy:
            steps, greedy_summary = _run_constrained_greedy(
                groups,
                baseline,
                evaluator,
                max_steps=max_greedy_steps,
                minimum_improvement=minimum_improvement,
                auxiliary_tolerance=auxiliary_tolerance,
                target_regressor_mae=target_regressor_mae,
                target_classifier_mae=target_classifier_mae,
                full_parameter_count=full_parameter_count,
            )
        else:
            steps = []
            greedy_summary = None
        evaluation_count = evaluator.evaluation_count

    target_selected_parameter_count = sum(
        group.parameter_element_count for group in groups
    )
    report: dict[str, Any] = {
        "analysis": "w8a16_conditional_weight_sensitivity",
        "metadata": {
            **p2_metadata,
            "baseline_family": baseline_family,
            "baseline_profile": scenario["baseline_profile"],
            "baseline_float_family": scenario["baseline_float_family"],
            "target_family": scenario["target_family"],
            "endpoint_profile": (
                "F4" if requested_groups is None else "selected_target_float"
            ),
            "granularity": granularity,
            "requested_groups": (
                list(requested_groups) if requested_groups is not None else None
            ),
            "group_count": len(groups),
            "baseline_float_parameter_site_count": baseline_definition.site_count,
            "baseline_float_parameter_element_count": (
                baseline_definition.parameter_element_count
            ),
            "target_family_parameter_site_count": target_definition.site_count,
            "target_family_parameter_element_count": (
                target_definition.parameter_element_count
            ),
            "selected_target_parameter_site_count": sum(
                group.site_count for group in groups
            ),
            "selected_target_parameter_element_count": (
                target_selected_parameter_count
            ),
            "full_parameter_element_count": full_parameter_count,
            "greedy_enabled": run_greedy,
            "max_greedy_steps": max_greedy_steps,
            "minimum_improvement": minimum_improvement,
            "auxiliary_tolerance": auxiliary_tolerance,
            "target_regressor_mae": target_regressor_mae,
            "target_classifier_mae": target_classifier_mae,
            "model_evaluation_count": evaluation_count,
        },
        "family_definitions": {
            name: definition.to_dict() for name, definition in definitions.items()
        },
        "group_definitions": [group.to_dict() for group in groups],
        "baseline": _copy_outputs(baseline),
        "all_selected_target_float_endpoint": {
            "outputs": _copy_outputs(endpoint),
            "target_reached": _target_reached(
                endpoint,
                target_regressor_mae,
                target_classifier_mae,
            ),
            "regressor_mae_improvement": (
                _mae(baseline, "regressors") - _mae(endpoint, "regressors")
            ),
            "classifier_mae_improvement": (
                _mae(baseline, "classifiers") - _mae(endpoint, "classifiers")
            ),
        },
        "independent": independent,
    }
    if run_greedy:
        report["greedy"] = {
            "steps": steps,
            "summary": greedy_summary,
        }
    report["recommendation"] = _recommendation(report)
    return report


def build_conditional_weight_groups(
    model: nn.Module,
    *,
    target_family: str,
    granularity: str = "semantic",
) -> tuple[dict[str, FamilyDefinition], tuple[ConditionalWeightGroup, ...]]:
    """Build family definitions and non-overlapping target-family groups."""
    if target_family not in {"regular", "depthwise"}:
        raise ValueError("target_family must be 'regular' or 'depthwise'.")
    if granularity not in _SUPPORTED_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {tuple(sorted(_SUPPORTED_GRANULARITIES))}."
        )

    site_groups = build_weight_sensitivity_groups(model, granularity="site")
    if not site_groups:
        raise ValueError("Conditional weight sensitivity requires parameter sites.")

    family_members: dict[str, list[WeightSensitivityGroup]] = defaultdict(list)
    for group in site_groups:
        if group.site_count != 1 or len(group.parameter_breakdown) != 1:
            raise RuntimeError(
                "Site-level weight groups must contain exactly one parameter site."
            )
        kind = group.parameter_breakdown[0].kind
        family = _KIND_TO_FAMILY.get(kind)
        if family is None:
            raise ValueError(f"Unsupported hand-detector parameter kind {kind!r}.")
        family_members[family].append(group)

    expected_families = ("regular", "depthwise", "prelu")
    missing = tuple(
        family for family in expected_families if not family_members.get(family)
    )
    if missing:
        raise RuntimeError(f"Missing expected parameter families: {missing}.")

    definitions = {
        family: FamilyDefinition(
            name=family,
            site_paths=tuple(
                path for group in family_members[family] for path in group.site_paths
            ),
            parameter_element_count=sum(
                group.parameter_element_count for group in family_members[family]
            ),
        )
        for family in expected_families
    }
    _validate_family_coverage(site_groups, definitions)

    target_sites = tuple(family_members[target_family])
    if granularity == "site":
        groups = tuple(
            _from_site_group(group, family=target_family) for group in target_sites
        )
    else:
        groups = _aggregate_semantic_groups(
            target_sites,
            family=target_family,
        )
    _validate_target_group_coverage(target_sites, groups)
    return definitions, groups


def select_conditional_weight_groups(
    groups: Sequence[ConditionalWeightGroup],
    names: Sequence[str] | None,
) -> tuple[ConditionalWeightGroup, ...]:
    """Resolve an optional ordered subset without silent misses."""
    available = {group.name: group for group in groups}
    if names is None:
        return tuple(groups)
    requested = tuple(names)
    if not requested:
        return ()
    if len(set(requested)) != len(requested):
        raise ValueError("Conditional weight group names must be unique.")
    missing = tuple(name for name in requested if name not in available)
    if missing:
        raise KeyError(
            f"Unknown conditional weight groups: {missing}; available groups: "
            f"{tuple(available)}."
        )
    return tuple(available[name] for name in requested)


class _ConditionalWeightEvaluator(
    AbstractContextManager["_ConditionalWeightEvaluator"]
):
    """Evaluate target-family FP subsets over one fixed family-FP baseline."""

    def __init__(
        self,
        reference_model: nn.Module,
        candidate_model: nn.Module,
        samples: Sequence[torch.Tensor],
        *,
        baseline_float_paths: Sequence[str],
        groups: Sequence[ConditionalWeightGroup],
        output_adapter: OutputAdapter,
    ) -> None:
        self.reference_model = reference_model
        self.candidate_model = candidate_model
        self.samples = samples
        self.baseline_float_paths = frozenset(baseline_float_paths)
        self.groups = tuple(groups)
        self.output_adapter = output_adapter
        self._group_paths = {
            group.name: frozenset(group.site_paths) for group in self.groups
        }
        self._context = FakeQuantState(candidate_model)
        self._state: FakeQuantState | None = None
        self._cache: dict[frozenset[str], dict[str, dict[str, float | int | None]]] = {}
        self.evaluation_count = 0

    def __enter__(self) -> "_ConditionalWeightEvaluator":
        self._state = self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._context.__exit__(exc_type, exc_value, traceback)
        self._state = None
        return None

    def evaluate(
        self,
        selected_float_groups: frozenset[str],
    ) -> dict[str, dict[str, float | int | None]]:
        """Evaluate the conditional baseline plus selected FP target groups."""
        cached = self._cache.get(selected_float_groups)
        if cached is not None:
            return _copy_outputs(cached)
        if self._state is None:
            raise RuntimeError("Conditional weight evaluator is not active.")
        unknown = tuple(
            sorted(set(selected_float_groups).difference(self._group_paths))
        )
        if unknown:
            raise KeyError(f"Unknown conditional weight groups: {unknown}.")

        floating_paths = set(self.baseline_float_paths)
        for name in selected_float_groups:
            floating_paths.update(self._group_paths[name])

        self._state.set_all(True)
        if floating_paths:
            paths = frozenset(floating_paths)
            self._state.set_where(
                SiteSelector(
                    lambda site, selected=paths: (  # type: ignore[misc]
                        site.path in selected
                    ),
                    "conditional_weight_float_paths",
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
        self._cache[selected_float_groups] = value
        self.evaluation_count += 1
        return _copy_outputs(value)


def _build_independent_report(
    groups: Sequence[ConditionalWeightGroup],
    baseline: MetricSummary,
    evaluator: _ConditionalWeightEvaluator,
    *,
    auxiliary_tolerance: float,
    target_regressor_mae: float,
    target_classifier_mae: float,
    full_parameter_count: int,
) -> list[dict[str, object]]:
    baseline_reg = _mae(baseline, "regressors")
    baseline_cls = _mae(baseline, "classifiers")
    rows: list[dict[str, Any]] = []
    for group in groups:
        outputs = evaluator.evaluate(frozenset({group.name}))
        reg_gain = baseline_reg - _mae(outputs, "regressors")
        cls_gain = baseline_cls - _mae(outputs, "classifiers")
        row = group.to_dict()
        row.update(
            {
                "outputs": _copy_outputs(outputs),
                "regressor_mae_improvement": reg_gain,
                "classifier_mae_improvement": cls_gain,
                "eligible": (
                    reg_gain > 0.0
                    and _mae(outputs, "classifiers")
                    <= baseline_cls + auxiliary_tolerance
                ),
                "target_reached": _target_reached(
                    outputs,
                    target_regressor_mae,
                    target_classifier_mae,
                ),
                "regressor_gain_per_million_parameters": (
                    reg_gain * 1_000_000.0 / max(group.parameter_element_count, 1)
                ),
                "float_parameter_ratio": (
                    group.parameter_element_count / max(full_parameter_count, 1)
                ),
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["regressor_mae_improvement"]),
            -float(row["classifier_mae_improvement"]),
            str(row["group"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _run_constrained_greedy(
    groups: Sequence[ConditionalWeightGroup],
    baseline: MetricSummary,
    evaluator: _ConditionalWeightEvaluator,
    *,
    max_steps: int,
    minimum_improvement: float,
    auxiliary_tolerance: float,
    target_regressor_mae: float,
    target_classifier_mae: float,
    full_parameter_count: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    group_by_name = {group.name: group for group in groups}
    selected: list[str] = []
    remaining = list(groups)
    current = _copy_outputs(baseline)
    baseline_reg = _mae(baseline, "regressors")
    baseline_cls = _mae(baseline, "classifiers")
    step_limit = len(groups) if max_steps == 0 else min(max_steps, len(groups))
    steps: list[dict[str, Any]] = []
    stop_reason = "all_groups_selected"

    if _target_reached(current, target_regressor_mae, target_classifier_mae):
        stop_reason = "baseline_target_reached"
    else:
        for step_index in range(1, step_limit + 1):
            current_reg = _mae(current, "regressors")
            current_cls = _mae(current, "classifiers")
            best: tuple[
                ConditionalWeightGroup,
                dict[str, dict[str, float | int | None]],
                float,
            ] | None = None
            best_improvement = float("-inf")
            selected_set = frozenset(selected)
            for group in remaining:
                candidate_names = selected_set | frozenset({group.name})
                outputs = evaluator.evaluate(candidate_names)
                reg_improvement = current_reg - _mae(outputs, "regressors")
                if reg_improvement <= minimum_improvement:
                    continue
                if _mae(outputs, "classifiers") > current_cls + auxiliary_tolerance:
                    continue
                if reg_improvement > best_improvement:
                    best = (group, outputs, reg_improvement)
                    best_improvement = reg_improvement

            if best is None:
                stop_reason = "no_eligible_improvement"
                break

            group, outputs, reg_improvement = best
            selected.append(group.name)
            remaining.remove(group)
            selected_parameter_count = sum(
                group_by_name[name].parameter_element_count for name in selected
            )
            step = {
                "step": step_index,
                "added_group": group.name,
                "selected_groups": list(selected),
                "outputs": _copy_outputs(outputs),
                "incremental_regressor_mae_improvement": reg_improvement,
                "cumulative_regressor_mae_improvement": (
                    baseline_reg - _mae(outputs, "regressors")
                ),
                "incremental_classifier_mae_improvement": (
                    current_cls - _mae(outputs, "classifiers")
                ),
                "cumulative_classifier_mae_improvement": (
                    baseline_cls - _mae(outputs, "classifiers")
                ),
                "selected_parameter_element_count": selected_parameter_count,
                "selected_parameter_ratio": (
                    selected_parameter_count / max(full_parameter_count, 1)
                ),
                "target_reached": _target_reached(
                    outputs,
                    target_regressor_mae,
                    target_classifier_mae,
                ),
            }
            steps.append(step)
            current = _copy_outputs(outputs)
            if bool(step["target_reached"]):
                stop_reason = "target_reached"
                break
        else:
            if step_limit < len(groups):
                stop_reason = "max_steps_reached"

    first_target_step = next(
        (int(step["step"]) for step in steps if bool(step["target_reached"])),
        None,
    )
    selected_set = frozenset(selected)
    selected_parameter_count = sum(
        group.parameter_element_count for group in groups if group.name in selected_set
    )
    summary = {
        "stop_reason": stop_reason,
        "target_reached": (
            first_target_step is not None
            or _target_reached(current, target_regressor_mae, target_classifier_mae)
        ),
        "first_target_step": first_target_step,
        "selected_group_count": len(selected),
        "selected_groups": list(selected),
        "remaining_w8_groups": [
            group.name for group in groups if group.name not in selected_set
        ],
        "selected_parameter_element_count": selected_parameter_count,
        "selected_parameter_ratio": (
            selected_parameter_count / max(full_parameter_count, 1)
        ),
        "final_outputs": _copy_outputs(current),
    }
    return steps, summary


def print_conditional_weight_sensitivity(
    report: Mapping[str, Any],
    *,
    top_k: int,
) -> None:
    """Print conditional baseline, independent ranking, and greedy path."""
    if top_k < 0:
        raise ValueError("top_k must be nonnegative.")
    metadata = report["metadata"]
    baseline = report["baseline"]
    endpoint = report["all_selected_target_float_endpoint"]["outputs"]
    print("\nW8/A16 conditional weight sensitivity")
    print(
        f"Baseline {metadata['baseline_profile']} "
        f"({metadata['baseline_family']}): "
        f"REG_MAE={_mae(baseline, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(baseline, 'classifiers'):.6e}"
    )
    print(
        f"Target family={metadata['target_family']}, "
        f"granularity={metadata['granularity']}, "
        f"groups={metadata['group_count']}"
    )
    print(
        "All selected target weights FP: "
        f"REG_MAE={_mae(endpoint, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(endpoint, 'classifiers'):.6e}"
    )

    independent = report["independent"]
    shown = independent if top_k == 0 else independent[:top_k]
    print("\nIndependent conditional leave-one-group-float ranking")
    print(
        f"{'rank':>4s} {'group':40s} {'REG_MAE':>13s} {'GAIN_REG':>13s} "
        f"{'CLS_MAE':>13s} {'GAIN_CLS':>13s} {'PARAMS':>11s} "
        f"{'SITES':>7s} {'OK':>4s}"
    )
    for row in shown:
        outputs = row["outputs"]
        print(
            f"{int(row['rank']):4d} "
            f"{str(row['group'])[:40]:40s} "
            f"{_mae(outputs, 'regressors'):13.6e} "
            f"{float(row['regressor_mae_improvement']):13.6e} "
            f"{_mae(outputs, 'classifiers'):13.6e} "
            f"{float(row['classifier_mae_improvement']):13.6e} "
            f"{int(row['parameter_element_count']):11d} "
            f"{int(row['site_count']):7d} "
            f"{'yes' if row['eligible'] else 'no':>4s}"
        )
    if top_k and len(independent) > top_k:
        print(f"Showing {top_k} of {len(independent)} groups; JSON contains all.")

    greedy = report.get("greedy")
    if isinstance(greedy, Mapping):
        print("\nClassifier-constrained conditional greedy path")
        print(
            f"{'step':>4s} {'added_group':40s} {'REG_MAE':>13s} "
            f"{'DELTA_REG':>13s} {'TOTAL_REG':>13s} {'CLS_MAE':>13s} "
            f"{'DELTA_CLS':>13s} {'FP_PARAMS':>11s}"
        )
        for step in greedy["steps"]:
            outputs = step["outputs"]
            print(
                f"{int(step['step']):4d} "
                f"{str(step['added_group'])[:40]:40s} "
                f"{_mae(outputs, 'regressors'):13.6e} "
                f"{float(step['incremental_regressor_mae_improvement']):13.6e} "
                f"{float(step['cumulative_regressor_mae_improvement']):13.6e} "
                f"{_mae(outputs, 'classifiers'):13.6e} "
                f"{float(step['incremental_classifier_mae_improvement']):13.6e} "
                f"{int(step['selected_parameter_element_count']):11d}"
            )
        summary = greedy["summary"]
        if bool(summary["target_reached"]):
            print(
                "Target reached at step "
                f"{summary['first_target_step']}: "
                f"REG_MAE<{metadata['target_regressor_mae']:g}, "
                f"CLS_MAE<{metadata['target_classifier_mae']:g}."
            )
        else:
            final = summary["final_outputs"]
            print(
                "Target not reached; "
                f"REG_MAE={_mae(final, 'regressors'):.6e}, "
                f"CLS_MAE={_mae(final, 'classifiers'):.6e}, "
                f"stop={summary['stop_reason']}."
            )
    print("Recommendation: " + str(report["recommendation"]))


def _aggregate_semantic_groups(
    members: Sequence[WeightSensitivityGroup],
    *,
    family: str,
) -> tuple[ConditionalWeightGroup, ...]:
    buckets: dict[str, list[WeightSensitivityGroup]] = {}
    for member in members:
        buckets.setdefault(member.semantic_group, []).append(member)

    results: list[ConditionalWeightGroup] = []
    for semantic_group, grouped in buckets.items():
        results.append(
            ConditionalWeightGroup(
                name=semantic_group,
                family=family,
                granularity="semantic",
                semantic_group=semantic_group,
                block_kind=grouped[0].block_kind,
                operation_positions=_unique_sorted(
                    value for group in grouped for value in group.operation_positions
                ),
                operation_indices=_unique_in_order(
                    value for group in grouped for value in group.operation_indices
                ),
                operation_names=_unique_in_order(
                    value for group in grouped for value in group.operation_names
                ),
                site_paths=tuple(
                    path for group in grouped for path in group.site_paths
                ),
                parameter_element_count=sum(
                    group.parameter_element_count for group in grouped
                ),
            )
        )
    return tuple(results)


def _from_site_group(
    group: WeightSensitivityGroup,
    *,
    family: str,
) -> ConditionalWeightGroup:
    return ConditionalWeightGroup(
        name=group.name,
        family=family,
        granularity="site",
        semantic_group=group.semantic_group,
        block_kind=group.block_kind,
        operation_positions=group.operation_positions,
        operation_indices=group.operation_indices,
        operation_names=group.operation_names,
        site_paths=group.site_paths,
        parameter_element_count=group.parameter_element_count,
    )


def _validate_family_coverage(
    site_groups: Sequence[WeightSensitivityGroup],
    definitions: Mapping[str, FamilyDefinition],
) -> None:
    expected = tuple(sorted(path for group in site_groups for path in group.site_paths))
    actual_paths = [
        path for definition in definitions.values() for path in definition.site_paths
    ]
    actual = tuple(sorted(actual_paths))
    if expected != actual or len(actual_paths) != len(set(actual_paths)):
        raise RuntimeError(
            "Conditional family definitions must cover every parameter site once."
        )


def _validate_target_group_coverage(
    target_sites: Sequence[WeightSensitivityGroup],
    groups: Sequence[ConditionalWeightGroup],
) -> None:
    expected = tuple(
        sorted(path for group in target_sites for path in group.site_paths)
    )
    actual_paths = [path for group in groups for path in group.site_paths]
    actual = tuple(sorted(actual_paths))
    if expected != actual or len(actual_paths) != len(set(actual_paths)):
        raise RuntimeError(
            "Conditional target groups must cover every target-family site once."
        )


def _recommendation(report: Mapping[str, Any]) -> str:
    metadata = report["metadata"]
    endpoint = report["all_selected_target_float_endpoint"]
    target_family = str(metadata["target_family"])
    if not bool(endpoint["target_reached"]):
        return (
            f"Even floating all selected {target_family} weights misses the "
            "targets; expand the group set or optimize both Conv families jointly."
        )
    greedy = report.get("greedy")
    if not isinstance(greedy, Mapping):
        return (
            f"Use the independent {target_family} ranking to choose joint scale "
            "and rounding reconstruction candidates."
        )
    summary = greedy["summary"]
    if bool(summary["target_reached"]):
        return (
            f"Conditional {target_family} greedy reaches the targets; prioritize "
            "its selected groups for joint W8 optimization."
        )
    return (
        f"The full {target_family} oracle reaches the targets but forward greedy "
        "does not, indicating strong within-family interaction; use broader joint "
        "reconstruction rather than independent per-group tuning."
    )


def _validate_run_arguments(
    calibration_samples: Sequence[torch.Tensor],
    evaluation_samples: Sequence[torch.Tensor],
    *,
    baseline_family: str,
    granularity: str,
    max_greedy_steps: int,
    minimum_improvement: float,
    auxiliary_tolerance: float,
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> None:
    if not calibration_samples or not evaluation_samples:
        raise ValueError(
            "Conditional weight sensitivity requires calibration and evaluation data."
        )
    if baseline_family not in _SCENARIOS:
        raise ValueError(f"baseline_family must be one of {tuple(sorted(_SCENARIOS))}.")
    if granularity not in _SUPPORTED_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {tuple(sorted(_SUPPORTED_GRANULARITIES))}."
        )
    if max_greedy_steps < 0:
        raise ValueError("max_greedy_steps must be nonnegative.")
    for name, value in (
        ("minimum_improvement", minimum_improvement),
        ("auxiliary_tolerance", auxiliary_tolerance),
        ("target_regressor_mae", target_regressor_mae),
        ("target_classifier_mae", target_classifier_mae),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if minimum_improvement < 0.0 or auxiliary_tolerance < 0.0:
        raise ValueError("Improvement and tolerance must be nonnegative.")
    if target_regressor_mae <= 0.0 or target_classifier_mae <= 0.0:
        raise ValueError("Target MAEs must be positive.")


def _target_reached(
    outputs: MetricSummary,
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> bool:
    return (
        _mae(outputs, "regressors") < target_regressor_mae
        and _mae(outputs, "classifiers") < target_classifier_mae
    )


def _mae(outputs: MetricSummary, output_name: str) -> float:
    value = outputs[output_name]["mae"]
    if value is None:
        raise ValueError(f"Missing MAE for output {output_name!r}.")
    return float(value)


def _copy_outputs(
    outputs: MetricSummary,
) -> dict[str, dict[str, float | int | None]]:
    return {name: dict(metrics) for name, metrics in outputs.items()}


def _unique_sorted(values) -> tuple[int, ...]:
    return tuple(sorted(set(int(value) for value in values)))


def _unique_in_order(values) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)
