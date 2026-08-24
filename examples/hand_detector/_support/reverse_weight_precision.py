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

"""Reverse FP-to-W8 parameter search under the P2/P3 hand-detector profiles."""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import torch
from tico.quantization.analysis import evaluate_models, OutputAdapter, SiteSelector
from tico.quantization.analysis.reverse_precision_search import (
    ReversePrecisionGroup,
    run_reverse_beam,
    run_reverse_greedy,
)
from tico.quantization.wrapq.control import FakeQuantState, SiteRole
from torch import nn

from examples.hand_detector._support.weight_precision_sensitivity import (
    build_w8a16_candidate,
    build_weight_sensitivity_groups,
    parameter_totals,
    select_weight_sensitivity_groups,
    WeightSensitivityGroup,
)


MetricSummary = Mapping[str, Mapping[str, float | int | None]]


def run_reverse_weight_precision_diagnostic(
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
    greedy_selection_objective: str,
    max_greedy_steps: int,
    beam_width: int,
    beam_exploration_slots: int,
    beam_candidate_count: int,
    max_beam_steps: int,
    target_regressor_mae: float,
    target_classifier_mae: float,
    search_regressor_ceiling: float | None,
    search_classifier_ceiling: float | None,
    output_adapter: OutputAdapter,
) -> dict[str, Any]:
    """Run one-group W8 costs, reverse greedy, and optional beam search."""
    _validate_arguments(
        evaluation_samples,
        max_greedy_steps=max_greedy_steps,
        beam_width=beam_width,
        beam_exploration_slots=beam_exploration_slots,
        beam_candidate_count=beam_candidate_count,
        max_beam_steps=max_beam_steps,
        target_regressor_mae=target_regressor_mae,
        target_classifier_mae=target_classifier_mae,
        search_regressor_ceiling=search_regressor_ceiling,
        search_classifier_ceiling=search_classifier_ceiling,
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
    if not groups:
        raise ValueError("Reverse weight search requires at least one selected group.")
    full_site_count, full_element_count = parameter_totals(all_groups)
    search_site_count, search_element_count = parameter_totals(groups)
    generic_groups = tuple(
        ReversePrecisionGroup(group.name, group.parameter_element_count)
        for group in groups
    )

    with _WeightSubsetEvaluator(
        float_model,
        candidate,
        evaluation_samples,
        all_groups,
        output_adapter=output_adapter,
    ) as evaluator:
        p3_outputs = evaluator.evaluate(frozenset())
        p2_outputs = evaluator.evaluate(frozenset(group.name for group in all_groups))
        _validate_entry_targets(
            p3_outputs,
            target_regressor_mae=target_regressor_mae,
            target_classifier_mae=target_classifier_mae,
        )
        independent = _build_independent_report(
            groups,
            p3_outputs,
            evaluator,
            target_regressor_mae=target_regressor_mae,
            target_classifier_mae=target_classifier_mae,
            full_parameter_element_count=full_element_count,
        )

        payload: dict[str, Any] = {
            "analysis": "w8a16_reverse_weight_precision_diagnostic",
            "metadata": {
                **p2_metadata,
                "granularity": granularity,
                "requested_groups": (
                    list(requested_groups) if requested_groups is not None else None
                ),
                "full_parameter_site_count": full_site_count,
                "full_parameter_element_count": full_element_count,
                "search_parameter_site_count": search_site_count,
                "search_parameter_element_count": search_element_count,
                "target_regressor_mae": target_regressor_mae,
                "target_classifier_mae": target_classifier_mae,
                "greedy_enabled": run_greedy,
                "greedy_selection_objective": greedy_selection_objective,
                "max_greedy_steps": max_greedy_steps,
                "beam_width": beam_width,
                "beam_exploration_slots": beam_exploration_slots,
                "beam_candidate_count": beam_candidate_count,
                "max_beam_steps": max_beam_steps,
            },
            "p3_entry": _copy_outputs(p3_outputs),
            "p2_endpoint": _copy_outputs(p2_outputs),
            "group_definitions": [group.to_dict() for group in groups],
            "independent": independent,
        }

        if run_greedy:
            greedy = run_reverse_greedy(
                generic_groups,
                p3_outputs,
                evaluator.evaluate,
                primary_output="regressors",
                auxiliary_output="classifiers",
                target_primary=target_regressor_mae,
                target_auxiliary=target_classifier_mae,
                max_steps=max_greedy_steps,
                selection_objective=greedy_selection_objective,
            )
            payload["greedy"] = _enrich_search_result(
                greedy.to_dict(),
                all_groups,
                full_parameter_element_count=full_element_count,
            )

        if beam_width > 0:
            beam_groups = _select_beam_groups(
                groups,
                independent,
                beam_candidate_count,
            )
            resolved_reg_ceiling = (
                _mae(p2_outputs, "regressors")
                if search_regressor_ceiling is None
                else search_regressor_ceiling
            )
            resolved_cls_ceiling = (
                _mae(p2_outputs, "classifiers")
                if search_classifier_ceiling is None
                else search_classifier_ceiling
            )
            resolved_reg_ceiling = max(
                resolved_reg_ceiling,
                target_regressor_mae,
            )
            resolved_cls_ceiling = max(
                resolved_cls_ceiling,
                target_classifier_mae,
            )
            beam = run_reverse_beam(
                tuple(
                    ReversePrecisionGroup(
                        group.name,
                        group.parameter_element_count,
                    )
                    for group in beam_groups
                ),
                p3_outputs,
                evaluator.evaluate,
                primary_output="regressors",
                auxiliary_output="classifiers",
                target_primary=target_regressor_mae,
                target_auxiliary=target_classifier_mae,
                search_primary_ceiling=resolved_reg_ceiling,
                search_auxiliary_ceiling=resolved_cls_ceiling,
                beam_width=beam_width,
                exploration_slots=beam_exploration_slots,
                max_steps=max_beam_steps,
            )
            beam_payload = _enrich_search_result(
                beam.to_dict(),
                all_groups,
                full_parameter_element_count=full_element_count,
            )
            beam_payload.update(
                {
                    "candidate_groups": [group.name for group in beam_groups],
                    "search_regressor_ceiling": resolved_reg_ceiling,
                    "search_classifier_ceiling": resolved_cls_ceiling,
                }
            )
            payload["beam"] = beam_payload

        payload["metadata"]["model_evaluation_count"] = evaluator.evaluation_count
    return payload


class _WeightSubsetEvaluator(AbstractContextManager["_WeightSubsetEvaluator"]):
    """Evaluate arbitrary W8 group subsets from one prepared P2 candidate."""

    def __init__(
        self,
        reference_model: nn.Module,
        candidate_model: nn.Module,
        samples: Sequence[torch.Tensor],
        groups: Sequence[WeightSensitivityGroup],
        *,
        output_adapter: OutputAdapter,
    ) -> None:
        self.reference_model = reference_model
        self.candidate_model = candidate_model
        self.samples = samples
        self.groups = tuple(groups)
        self.output_adapter = output_adapter
        self._group_paths = {
            group.name: frozenset(group.site_paths) for group in self.groups
        }
        self._cache: dict[frozenset[str], dict[str, dict[str, float | int | None]]] = {}
        self._context = FakeQuantState(candidate_model)
        self._state: FakeQuantState | None = None
        self.evaluation_count = 0

    def __enter__(self) -> "_WeightSubsetEvaluator":
        self._state = self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._context.__exit__(exc_type, exc_value, traceback)
        self._state = None
        return None

    def evaluate(
        self,
        selected_groups: frozenset[str],
    ) -> dict[str, dict[str, float | int | None]]:
        cached = self._cache.get(selected_groups)
        if cached is not None:
            return _copy_outputs(cached)
        if self._state is None:
            raise RuntimeError("Weight subset evaluator is not active.")
        unknown = tuple(sorted(set(selected_groups).difference(self._group_paths)))
        if unknown:
            raise KeyError(f"Unknown reverse weight groups: {unknown}.")
        self._state.set_all(True)
        self._state.set_where(
            SiteSelector.roles(SiteRole.PARAMETER),
            False,
        )
        selected_paths = frozenset(
            path for name in selected_groups for path in self._group_paths[name]
        )
        if selected_paths:
            self._state.set_where(
                SiteSelector(
                    lambda site, paths=selected_paths: (  # type: ignore[misc]
                        site.path in paths
                    ),
                    "reverse_weight_exact_paths",
                ),
                True,
            )
        outputs = evaluate_models(
            self.reference_model,
            self.candidate_model,
            self.samples,
            output_adapter=self.output_adapter,
        )
        value = _copy_outputs(outputs)
        self._cache[selected_groups] = value
        self.evaluation_count += 1
        return _copy_outputs(value)


def _build_independent_report(
    groups: Sequence[WeightSensitivityGroup],
    p3_outputs: MetricSummary,
    evaluator: _WeightSubsetEvaluator,
    *,
    target_regressor_mae: float,
    target_classifier_mae: float,
    full_parameter_element_count: int,
) -> list[dict[str, object]]:
    baseline_reg = _mae(p3_outputs, "regressors")
    baseline_cls = _mae(p3_outputs, "classifiers")
    rows: list[dict[str, Any]] = []
    for group in groups:
        outputs = evaluator.evaluate(frozenset({group.name}))
        reg = _mae(outputs, "regressors")
        cls = _mae(outputs, "classifiers")
        row = group.to_dict()
        row.update(
            {
                "outputs": _copy_outputs(outputs),
                "regressor_mae_cost": reg - baseline_reg,
                "classifier_mae_cost": cls - baseline_cls,
                "target_feasible": (
                    reg < target_regressor_mae and cls < target_classifier_mae
                ),
                "quantized_parameter_ratio": (
                    group.parameter_element_count / max(full_parameter_element_count, 1)
                ),
                "regressor_cost_per_million_parameters": (
                    (reg - baseline_reg)
                    * 1_000_000.0
                    / max(group.parameter_element_count, 1)
                ),
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["regressor_mae_cost"]),
            float(row["classifier_mae_cost"]),
            -int(row["parameter_element_count"]),
            str(row["group"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _select_beam_groups(
    groups: Sequence[WeightSensitivityGroup],
    independent: Sequence[Mapping[str, object]],
    candidate_count: int,
) -> tuple[WeightSensitivityGroup, ...]:
    if candidate_count == 0 or candidate_count >= len(groups):
        return tuple(groups)
    names = tuple(str(row["group"]) for row in independent[:candidate_count])
    return select_weight_sensitivity_groups(groups, names)


def _enrich_search_result(
    payload: dict[str, object],
    groups: Sequence[WeightSensitivityGroup],
    *,
    full_parameter_element_count: int,
) -> dict[str, object]:
    group_by_name = {group.name: group for group in groups}

    def enrich_state(state: dict[str, Any]) -> None:
        selected = tuple(str(name) for name in state["selected_groups"])
        selected_set = frozenset(selected)
        remaining = tuple(
            group.name for group in groups if group.name not in selected_set
        )
        quantized = sum(
            group_by_name[name].parameter_element_count for name in selected
        )
        state.update(
            {
                "remaining_float_groups": list(remaining),
                "remaining_float_group_count": len(remaining),
                "quantized_parameter_element_count": quantized,
                "remaining_float_parameter_element_count": (
                    full_parameter_element_count - quantized
                ),
                "quantized_parameter_ratio": (
                    quantized / max(full_parameter_element_count, 1)
                ),
            }
        )

    for key in ("entry", "final", "best"):
        value = payload.get(key)
        if isinstance(value, dict):
            enrich_state(value)
    steps = payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                enrich_state(step)
    frontier = payload.get("frontier")
    if isinstance(frontier, list):
        for state in frontier:
            if isinstance(state, dict):
                enrich_state(state)
    return payload


def print_reverse_weight_precision(report: Mapping[str, Any], *, top_k: int) -> None:
    """Print independent reverse costs and target-feasible search summaries."""
    p3 = report["p3_entry"]
    p2 = report["p2_endpoint"]
    metadata = report["metadata"]
    print("\nW8/A16 reverse weight precision diagnostic")
    print(
        "P3 entry, all weights FP32: "
        f"REG_MAE={_mae(p3, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(p3, 'classifiers'):.6e}"
    )
    print(
        "P2 endpoint, all weights W8: "
        f"REG_MAE={_mae(p2, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(p2, 'classifiers'):.6e}"
    )
    print(
        "Targets: "
        f"REG<{float(metadata['target_regressor_mae']):g}, "
        f"CLS<{float(metadata['target_classifier_mae']):g}"
    )
    rows = report["independent"]
    shown = rows if top_k == 0 else rows[:top_k]
    print("\nIndependent cost of quantizing one group from P3")
    print(
        f"{'rank':>4s} {'group':38s} {'REG_MAE':>13s} {'COST_REG':>13s} "
        f"{'CLS_MAE':>13s} {'COST_CLS':>13s} {'PARAMS':>11s} {'OK':>4s}"
    )
    for row in shown:
        outputs = row["outputs"]
        print(
            f"{int(row['rank']):4d} "
            f"{str(row['group'])[:38]:38s} "
            f"{_mae(outputs, 'regressors'):13.6e} "
            f"{float(row['regressor_mae_cost']):13.6e} "
            f"{_mae(outputs, 'classifiers'):13.6e} "
            f"{float(row['classifier_mae_cost']):13.6e} "
            f"{int(row['parameter_element_count']):11d} "
            f"{'yes' if row['target_feasible'] else 'no':>4s}"
        )
    if top_k and len(rows) > top_k:
        print(f"Showing {top_k} of {len(rows)} groups; JSON contains all rows.")

    greedy = report.get("greedy")
    if isinstance(greedy, Mapping):
        print("\nTarget-feasible reverse greedy path")
        print(
            f"{'step':>4s} {'added_group':38s} {'REG_MAE':>13s} "
            f"{'DELTA_REG':>13s} {'CLS_MAE':>13s} {'DELTA_CLS':>13s} "
            f"{'W8_PARAMS':>11s} {'W8(%)':>8s}"
        )
        for step in greedy["steps"]:
            outputs = step["outputs"]
            print(
                f"{int(step['step']):4d} "
                f"{str(step['added_group'])[:38]:38s} "
                f"{_mae(outputs, 'regressors'):13.6e} "
                f"{float(step['incremental_primary_cost']):13.6e} "
                f"{_mae(outputs, 'classifiers'):13.6e} "
                f"{float(step['incremental_auxiliary_cost']):13.6e} "
                f"{int(step['quantized_parameter_element_count']):11d} "
                f"{100.0 * float(step['quantized_parameter_ratio']):8.3f}"
            )
        final = greedy["final"]
        print(
            f"Greedy stopped: {greedy['stop_reason']}; "
            f"W8={100.0 * float(final['quantized_parameter_ratio']):.3f}%, "
            f"remaining FP groups={final['remaining_float_groups']}"
        )

    beam = report.get("beam")
    if isinstance(beam, Mapping):
        best = beam["best"]
        print("\nReverse beam-search best target-feasible state")
        print(
            f"REG_MAE={_mae(best['outputs'], 'regressors'):.6e}, "
            f"CLS_MAE={_mae(best['outputs'], 'classifiers'):.6e}, "
            f"W8={100.0 * float(best['quantized_parameter_ratio']):.3f}%"
        )
        print(f"W8 groups: {best['selected_groups']}")
        print(f"Must-optimize FP groups: {best['remaining_float_groups']}")
        print(
            f"Beam stopped: {beam['stop_reason']}; depth={beam['depth_reached']}, "
            f"evaluations={beam['evaluation_count']}, "
            f"expanded={beam['expanded_state_count']}"
        )


def _validate_arguments(
    evaluation_samples: Sequence[torch.Tensor],
    *,
    max_greedy_steps: int,
    beam_width: int,
    beam_exploration_slots: int,
    beam_candidate_count: int,
    max_beam_steps: int,
    target_regressor_mae: float,
    target_classifier_mae: float,
    search_regressor_ceiling: float | None,
    search_classifier_ceiling: float | None,
) -> None:
    if not evaluation_samples:
        raise ValueError("Reverse weight diagnostic requires evaluation samples.")
    for name, value in (
        ("max_greedy_steps", max_greedy_steps),
        ("beam_width", beam_width),
        ("beam_exploration_slots", beam_exploration_slots),
        ("beam_candidate_count", beam_candidate_count),
        ("max_beam_steps", max_beam_steps),
    ):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")
    if beam_width == 0 and beam_exploration_slots != 0:
        raise ValueError("beam_exploration_slots must be zero when beam is disabled.")
    if beam_width > 0 and beam_exploration_slots >= beam_width:
        raise ValueError("beam_exploration_slots must be smaller than beam_width.")
    for name, target_value in (
        ("target_regressor_mae", target_regressor_mae),
        ("target_classifier_mae", target_classifier_mae),
    ):
        if not math.isfinite(target_value) or target_value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, ceiling_value in (
        ("search_regressor_ceiling", search_regressor_ceiling),
        ("search_classifier_ceiling", search_classifier_ceiling),
    ):
        if ceiling_value is not None and (
            not math.isfinite(ceiling_value) or ceiling_value <= 0.0
        ):
            raise ValueError(f"{name} must be None or finite and positive.")


def _validate_entry_targets(
    outputs: MetricSummary,
    *,
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> None:
    reg = _mae(outputs, "regressors")
    cls = _mae(outputs, "classifiers")
    if reg >= target_regressor_mae or cls >= target_classifier_mae:
        raise ValueError(
            "P3 all-FP-weight entry does not satisfy the requested targets: "
            f"REG={reg:.6e}, CLS={cls:.6e}."
        )


def _mae(outputs: MetricSummary, name: str) -> float:
    value = outputs[name]["mae"]
    if not isinstance(value, (int, float)):
        raise KeyError(f"Output {name!r} has no numeric MAE.")
    return float(value)


def _copy_outputs(
    outputs: MetricSummary,
) -> dict[str, dict[str, float | int | None]]:
    return {name: dict(metrics) for name, metrics in outputs.items()}
