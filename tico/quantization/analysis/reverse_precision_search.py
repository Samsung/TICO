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

"""Reverse precision search under explicit numerical-error constraints."""

from __future__ import annotations

import math

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias


MetricSummary: TypeAlias = Mapping[str, Mapping[str, float | int | None]]
SubsetEvaluator: TypeAlias = Callable[[frozenset[str]], MetricSummary]


@dataclass(frozen=True)
class ReversePrecisionGroup:
    """Describe one independently switchable parameter group."""

    name: str
    element_count: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Reverse precision group names must be non-empty.")
        if self.element_count <= 0:
            raise ValueError(
                "Reverse precision groups require positive element counts."
            )


@dataclass(frozen=True)
class ReversePrecisionState:
    """Store one evaluated set of low-precision groups."""

    selected_groups: tuple[str, ...]
    outputs: Mapping[str, Mapping[str, float | int | None]]
    quantized_element_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_groups": list(self.selected_groups),
            "selected_group_count": len(self.selected_groups),
            "quantized_element_count": self.quantized_element_count,
            "outputs": _copy_outputs(self.outputs),
        }


@dataclass(frozen=True)
class ReversePrecisionStep:
    """Store one accepted reverse-greedy transition."""

    step: int
    added_group: str
    state: ReversePrecisionState
    incremental_primary_cost: float
    incremental_auxiliary_cost: float

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "added_group": self.added_group,
            "incremental_primary_cost": self.incremental_primary_cost,
            "incremental_auxiliary_cost": self.incremental_auxiliary_cost,
            **self.state.to_dict(),
        }


@dataclass(frozen=True)
class ReverseGreedyResult:
    """Store the complete target-feasible reverse-greedy path."""

    entry: ReversePrecisionState
    final: ReversePrecisionState
    steps: tuple[ReversePrecisionStep, ...]
    remaining_groups: tuple[str, ...]
    stop_reason: str
    evaluation_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "entry": self.entry.to_dict(),
            "final": self.final.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "remaining_groups": list(self.remaining_groups),
            "stop_reason": self.stop_reason,
            "evaluation_count": self.evaluation_count,
        }


@dataclass(frozen=True)
class ReverseBeamResult:
    """Store the best target-feasible state discovered by beam search."""

    entry: ReversePrecisionState
    best: ReversePrecisionState
    frontier: tuple[ReversePrecisionState, ...]
    stop_reason: str
    depth_reached: int
    evaluation_count: int
    expanded_state_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "entry": self.entry.to_dict(),
            "best": self.best.to_dict(),
            "frontier": [state.to_dict() for state in self.frontier],
            "stop_reason": self.stop_reason,
            "depth_reached": self.depth_reached,
            "evaluation_count": self.evaluation_count,
            "expanded_state_count": self.expanded_state_count,
        }


def run_reverse_greedy(
    groups: Sequence[ReversePrecisionGroup],
    entry_outputs: MetricSummary,
    evaluator: SubsetEvaluator,
    *,
    primary_output: str,
    auxiliary_output: str,
    target_primary: float,
    target_auxiliary: float,
    max_steps: int = 0,
    selection_objective: str = "primary-cost",
) -> ReverseGreedyResult:
    """Quantize one group at a time while preserving both output targets.

    ``primary-cost`` chooses the feasible transition with the smallest primary
    MAE increase. ``parameter-efficiency`` minimizes primary MAE increase per
    newly quantized element and therefore favors larger low-cost groups.
    """
    checked = _validate_search_inputs(
        groups,
        entry_outputs,
        primary_output=primary_output,
        auxiliary_output=auxiliary_output,
        target_primary=target_primary,
        target_auxiliary=target_auxiliary,
        max_steps=max_steps,
    )
    if selection_objective not in {"primary-cost", "parameter-efficiency"}:
        raise ValueError(
            "selection_objective must be 'primary-cost' or " "'parameter-efficiency'."
        )

    order = {group.name: index for index, group in enumerate(checked)}
    entry = ReversePrecisionState((), _copy_outputs(entry_outputs), 0)
    current = entry
    remaining = list(checked)
    steps: list[ReversePrecisionStep] = []
    evaluation_count = 0
    step_limit = len(checked) if max_steps == 0 else min(max_steps, len(checked))
    stop_reason = "max_steps"

    for step_index in range(1, step_limit + 1):
        candidates: list[
            tuple[
                tuple[float, ...],
                ReversePrecisionGroup,
                ReversePrecisionState,
                float,
                float,
            ]
        ] = []
        selected = frozenset(current.selected_groups)
        current_primary = _metric(current.outputs, primary_output)
        current_auxiliary = _metric(current.outputs, auxiliary_output)
        for group in remaining:
            candidate_names = frozenset((*selected, group.name))
            outputs = _copy_outputs(evaluator(candidate_names))
            evaluation_count += 1
            primary = _metric(outputs, primary_output)
            auxiliary = _metric(outputs, auxiliary_output)
            if not _within_targets(
                primary,
                auxiliary,
                target_primary=target_primary,
                target_auxiliary=target_auxiliary,
            ):
                continue
            primary_cost = primary - current_primary
            auxiliary_cost = auxiliary - current_auxiliary
            state = ReversePrecisionState(
                selected_groups=_ordered_names(candidate_names, checked),
                outputs=outputs,
                quantized_element_count=(
                    current.quantized_element_count + group.element_count
                ),
            )
            if selection_objective == "primary-cost":
                score = (
                    primary_cost,
                    auxiliary_cost,
                    -float(group.element_count),
                    float(order[group.name]),
                )
            else:
                score = (
                    primary_cost / group.element_count,
                    primary_cost,
                    auxiliary_cost,
                    -float(group.element_count),
                    float(order[group.name]),
                )
            candidates.append((score, group, state, primary_cost, auxiliary_cost))

        if not candidates:
            stop_reason = "no_target_feasible_transition"
            break
        _, group, state, primary_cost, auxiliary_cost = min(
            candidates,
            key=lambda value: value[0],
        )
        steps.append(
            ReversePrecisionStep(
                step=step_index,
                added_group=group.name,
                state=state,
                incremental_primary_cost=primary_cost,
                incremental_auxiliary_cost=auxiliary_cost,
            )
        )
        current = state
        remaining.remove(group)
        if not remaining:
            stop_reason = "all_groups_quantized"
            break
    else:
        if not remaining:
            stop_reason = "all_groups_quantized"

    return ReverseGreedyResult(
        entry=entry,
        final=current,
        steps=tuple(steps),
        remaining_groups=tuple(group.name for group in remaining),
        stop_reason=stop_reason,
        evaluation_count=evaluation_count,
    )


def run_reverse_beam(
    groups: Sequence[ReversePrecisionGroup],
    entry_outputs: MetricSummary,
    evaluator: SubsetEvaluator,
    *,
    primary_output: str,
    auxiliary_output: str,
    target_primary: float,
    target_auxiliary: float,
    search_primary_ceiling: float,
    search_auxiliary_ceiling: float,
    beam_width: int,
    exploration_slots: int = 1,
    max_steps: int = 0,
) -> ReverseBeamResult:
    """Search low-precision subsets with bounded temporary target violations.

    Target-feasible states are ranked by quantized parameter count. Reserved
    exploration slots keep high-coverage states that temporarily violate the
    final target but remain below the supplied search ceilings, allowing later
    groups to restore favorable error cancellation.
    """
    checked = _validate_search_inputs(
        groups,
        entry_outputs,
        primary_output=primary_output,
        auxiliary_output=auxiliary_output,
        target_primary=target_primary,
        target_auxiliary=target_auxiliary,
        max_steps=max_steps,
    )
    if beam_width <= 0:
        raise ValueError("beam_width must be positive.")
    if exploration_slots < 0 or exploration_slots >= beam_width:
        raise ValueError("exploration_slots must be in [0, beam_width).")
    for name, value, target in (
        ("search_primary_ceiling", search_primary_ceiling, target_primary),
        ("search_auxiliary_ceiling", search_auxiliary_ceiling, target_auxiliary),
    ):
        if not math.isfinite(value) or value < target:
            raise ValueError(f"{name} must be finite and no smaller than its target.")

    group_by_name = {group.name: group for group in checked}
    entry = ReversePrecisionState((), _copy_outputs(entry_outputs), 0)
    frontier = [entry]
    best = entry
    cache: dict[frozenset[str], Mapping[str, Mapping[str, float | int | None]]] = {
        frozenset(): entry.outputs
    }
    evaluation_count = 0
    expanded_state_count = 0
    depth_limit = len(checked) if max_steps == 0 else min(max_steps, len(checked))
    depth_reached = 0
    stop_reason = "max_steps"

    for depth in range(1, depth_limit + 1):
        expanded: dict[frozenset[str], ReversePrecisionState] = {}
        for state in frontier:
            selected = frozenset(state.selected_groups)
            for group in checked:
                if group.name in selected:
                    continue
                key = frozenset((*selected, group.name))
                if key in expanded:
                    continue
                outputs = cache.get(key)
                if outputs is None:
                    outputs = _copy_outputs(evaluator(key))
                    cache[key] = outputs
                    evaluation_count += 1
                primary = _metric(outputs, primary_output)
                auxiliary = _metric(outputs, auxiliary_output)
                if (
                    primary > search_primary_ceiling
                    or auxiliary > search_auxiliary_ceiling
                ):
                    continue
                expanded[key] = ReversePrecisionState(
                    selected_groups=_ordered_names(key, checked),
                    outputs=outputs,
                    quantized_element_count=sum(
                        group_by_name[name].element_count for name in key
                    ),
                )
        if not expanded:
            stop_reason = "no_search_admissible_transition"
            break
        expanded_state_count += len(expanded)
        depth_reached = depth
        states = list(expanded.values())
        for state in states:
            if _is_target_feasible(
                state,
                primary_output=primary_output,
                auxiliary_output=auxiliary_output,
                target_primary=target_primary,
                target_auxiliary=target_auxiliary,
            ) and _better_target_state(
                state,
                best,
                primary_output=primary_output,
                auxiliary_output=auxiliary_output,
            ):
                best = state

        exploit_count = beam_width - exploration_slots
        exploit = sorted(
            states,
            key=lambda state: _target_rank(
                state,
                primary_output=primary_output,
                auxiliary_output=auxiliary_output,
                target_primary=target_primary,
                target_auxiliary=target_auxiliary,
            ),
        )[:exploit_count]
        chosen = {frozenset(state.selected_groups) for state in exploit}
        explore_candidates = [
            state
            for state in sorted(
                states,
                key=lambda state: _exploration_rank(
                    state,
                    primary_output=primary_output,
                    auxiliary_output=auxiliary_output,
                    target_primary=target_primary,
                    target_auxiliary=target_auxiliary,
                ),
            )
            if frozenset(state.selected_groups) not in chosen
        ]
        frontier = [*exploit, *explore_candidates[:exploration_slots]]
        if all(len(state.selected_groups) == len(checked) for state in frontier):
            stop_reason = "all_groups_quantized"
            break
    else:
        if depth_limit == len(checked):
            stop_reason = "all_groups_quantized"

    return ReverseBeamResult(
        entry=entry,
        best=best,
        frontier=tuple(frontier),
        stop_reason=stop_reason,
        depth_reached=depth_reached,
        evaluation_count=evaluation_count,
        expanded_state_count=expanded_state_count,
    )


def _validate_search_inputs(
    groups: Sequence[ReversePrecisionGroup],
    entry_outputs: MetricSummary,
    *,
    primary_output: str,
    auxiliary_output: str,
    target_primary: float,
    target_auxiliary: float,
    max_steps: int,
) -> tuple[ReversePrecisionGroup, ...]:
    checked = tuple(groups)
    if not checked:
        raise ValueError("Reverse precision search requires at least one group.")
    names = tuple(group.name for group in checked)
    if len(set(names)) != len(names):
        raise ValueError("Reverse precision group names must be unique.")
    if primary_output == auxiliary_output:
        raise ValueError("Primary and auxiliary output names must differ.")
    for name, value in (
        ("target_primary", target_primary),
        ("target_auxiliary", target_auxiliary),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative.")
    entry_primary = _metric(entry_outputs, primary_output)
    entry_auxiliary = _metric(entry_outputs, auxiliary_output)
    if not _within_targets(
        entry_primary,
        entry_auxiliary,
        target_primary=target_primary,
        target_auxiliary=target_auxiliary,
    ):
        raise ValueError(
            "The all-high-precision entry state does not satisfy the requested "
            "primary and auxiliary targets."
        )
    return checked


def _ordered_names(
    selected: frozenset[str],
    groups: Sequence[ReversePrecisionGroup],
) -> tuple[str, ...]:
    return tuple(group.name for group in groups if group.name in selected)


def _within_targets(
    primary: float,
    auxiliary: float,
    *,
    target_primary: float,
    target_auxiliary: float,
) -> bool:
    return primary < target_primary and auxiliary < target_auxiliary


def _is_target_feasible(
    state: ReversePrecisionState,
    *,
    primary_output: str,
    auxiliary_output: str,
    target_primary: float,
    target_auxiliary: float,
) -> bool:
    return _within_targets(
        _metric(state.outputs, primary_output),
        _metric(state.outputs, auxiliary_output),
        target_primary=target_primary,
        target_auxiliary=target_auxiliary,
    )


def _better_target_state(
    candidate: ReversePrecisionState,
    incumbent: ReversePrecisionState,
    *,
    primary_output: str,
    auxiliary_output: str,
) -> bool:
    candidate_rank = (
        -candidate.quantized_element_count,
        _metric(candidate.outputs, primary_output),
        _metric(candidate.outputs, auxiliary_output),
        candidate.selected_groups,
    )
    incumbent_rank = (
        -incumbent.quantized_element_count,
        _metric(incumbent.outputs, primary_output),
        _metric(incumbent.outputs, auxiliary_output),
        incumbent.selected_groups,
    )
    return candidate_rank < incumbent_rank


def _target_rank(
    state: ReversePrecisionState,
    *,
    primary_output: str,
    auxiliary_output: str,
    target_primary: float,
    target_auxiliary: float,
) -> tuple[float, float, int, float, float, tuple[str, ...]]:
    primary = _metric(state.outputs, primary_output)
    auxiliary = _metric(state.outputs, auxiliary_output)
    primary_violation = max(primary - target_primary, 0.0) / target_primary
    auxiliary_violation = max(auxiliary - target_auxiliary, 0.0) / target_auxiliary
    return (
        float(primary_violation > 0.0 or auxiliary_violation > 0.0),
        primary_violation + auxiliary_violation,
        -state.quantized_element_count,
        primary,
        auxiliary,
        state.selected_groups,
    )


def _exploration_rank(
    state: ReversePrecisionState,
    *,
    primary_output: str,
    auxiliary_output: str,
    target_primary: float,
    target_auxiliary: float,
) -> tuple[int, float, float, float, tuple[str, ...]]:
    primary = _metric(state.outputs, primary_output)
    auxiliary = _metric(state.outputs, auxiliary_output)
    violation = (
        max(primary - target_primary, 0.0) / target_primary
        + max(auxiliary - target_auxiliary, 0.0) / target_auxiliary
    )
    return (
        -state.quantized_element_count,
        violation,
        primary,
        auxiliary,
        state.selected_groups,
    )


def _metric(outputs: MetricSummary, output_name: str) -> float:
    metrics = outputs.get(output_name)
    if metrics is None:
        raise KeyError(f"Missing output metrics for {output_name!r}.")
    value = metrics.get("mae")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Output {output_name!r} has no finite MAE metric.")
    return float(value)


def _copy_outputs(outputs: MetricSummary) -> dict[str, dict[str, float | int | None]]:
    return {name: dict(metrics) for name, metrics in outputs.items()}
