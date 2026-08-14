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

"""Group-wise fake-quantization sensitivity analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from torch import nn

from tico.quantization.analysis.inputs import ModelInput
from tico.quantization.analysis.metrics import evaluate_models
from tico.quantization.analysis.outputs import OutputAdapter
from tico.quantization.analysis.selector import SiteSelector
from tico.quantization.wrapq.control import (
    FakeQuantState,
    iter_quantization_sites,
    QuantizationSite,
)


class SensitivityMode(str, Enum):
    """Describe how one group differs from the selected baseline."""

    LEAVE_ONE_FLOAT = "leave_one_float"
    ENABLE_ONE = "enable_one"


@dataclass(frozen=True)
class QuantizationGroup:
    """Assign a stable name to one or more related quantization sites."""

    name: str
    selector: SiteSelector


@dataclass(frozen=True)
class SensitivityResult:
    """Store one independent group evaluation and its sensitivity score."""

    group: str
    outputs: Mapping[str, Mapping[str, float | int | None]]
    score: float
    sensitivity: float
    matched_sites: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible sensitivity result."""
        return {
            "group": self.group,
            "score": self.score,
            "sensitivity": self.sensitivity,
            "matched_site_count": len(self.matched_sites),
            "matched_sites": list(self.matched_sites),
            "outputs": {name: dict(metrics) for name, metrics in self.outputs.items()},
        }


@dataclass(frozen=True)
class SensitivityPathResult:
    """Store one cumulative or greedy sensitivity-path step."""

    step: int
    group: str
    selected_groups: tuple[str, ...]
    outputs: Mapping[str, Mapping[str, float | int | None]]
    score: float
    cumulative_sensitivity: float
    incremental_sensitivity: float
    matched_sites: tuple[str, ...]
    selected_sites: tuple[str, ...]

    @property
    def selected_group_count(self) -> int:
        """Return the number of groups accumulated through this step."""
        return len(self.selected_groups)

    @property
    def selected_site_count(self) -> int:
        """Return the number of quantization sites changed through this step."""
        return len(self.selected_sites)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible cumulative sensitivity step."""
        return {
            "step": self.step,
            "group": self.group,
            "selected_group_count": self.selected_group_count,
            "selected_groups": list(self.selected_groups),
            "score": self.score,
            "cumulative_sensitivity": self.cumulative_sensitivity,
            "incremental_sensitivity": self.incremental_sensitivity,
            "matched_site_count": len(self.matched_sites),
            "matched_sites": list(self.matched_sites),
            "selected_site_count": self.selected_site_count,
            "selected_sites": list(self.selected_sites),
            "outputs": {name: dict(metrics) for name, metrics in self.outputs.items()},
        }


@dataclass(frozen=True)
class _SensitivityContext:
    sites: tuple[QuantizationSite, ...]
    baseline_selector: SiteSelector
    effective_selectors: Mapping[str, SiteSelector]
    matched_sites: Mapping[str, tuple[str, ...]]


class QuantizationSensitivity:
    """Analyze model-defined groups relative to a fake-quantization baseline."""

    def __init__(
        self,
        reference_model: nn.Module,
        quantized_model: nn.Module,
        *,
        output_adapter: OutputAdapter | None = None,
    ) -> None:
        self.reference_model = reference_model
        self.quantized_model = quantized_model
        self.output_adapter = output_adapter

    def run(
        self,
        samples: Sequence[ModelInput],
        groups: Sequence[QuantizationGroup],
        *,
        mode: SensitivityMode = SensitivityMode.LEAVE_ONE_FLOAT,
        score_output: str,
        score_metric: str = "mae",
        baseline_selector: SiteSelector | None = None,
    ) -> tuple[dict[str, dict[str, float | int | None]], list[SensitivityResult]]:
        """Evaluate every group independently relative to a selected baseline.

        By default, ``LEAVE_ONE_FLOAT`` starts from every site enabled and
        ``ENABLE_ONE`` starts from every site disabled. ``baseline_selector``
        overrides that default and allows analysis relative to profiles such as
        E:internal-full, where final-output domains remain floating point.
        """
        context = self._prepare_context(
            samples,
            groups,
            mode=mode,
            baseline_selector=baseline_selector,
        )
        with FakeQuantState(self.quantized_model) as state:
            baseline_outputs, baseline_score = self._evaluate_baseline(
                state,
                context.baseline_selector,
                samples,
                score_output,
                score_metric,
            )
            results: list[SensitivityResult] = []
            for group in groups:
                outputs, score = self._evaluate_changed(
                    state,
                    context.baseline_selector,
                    context.effective_selectors[group.name],
                    samples,
                    mode,
                    score_output,
                    score_metric,
                )
                results.append(
                    SensitivityResult(
                        group=group.name,
                        outputs=outputs,
                        score=score,
                        sensitivity=_score_improvement(mode, baseline_score, score),
                        matched_sites=context.matched_sites[group.name],
                    )
                )

        results.sort(key=lambda result: result.sensitivity, reverse=True)
        return baseline_outputs, results

    def run_cumulative(
        self,
        samples: Sequence[ModelInput],
        groups: Sequence[QuantizationGroup],
        *,
        mode: SensitivityMode = SensitivityMode.LEAVE_ONE_FLOAT,
        score_output: str,
        score_metric: str = "mae",
        baseline_selector: SiteSelector | None = None,
    ) -> tuple[dict[str, dict[str, float | int | None]], list[SensitivityPathResult],]:
        """Apply groups cumulatively in the supplied order.

        Each step starts from the selected baseline and applies every group up
        to that point. For ``LEAVE_ONE_FLOAT``, selected groups are disabled;
        for ``ENABLE_ONE``, selected groups are enabled. A group that adds no
        mutable site after earlier steps is rejected as a cumulative no-op.
        """
        context = self._prepare_context(
            samples,
            groups,
            mode=mode,
            baseline_selector=baseline_selector,
        )
        with FakeQuantState(self.quantized_model) as state:
            baseline_outputs, baseline_score = self._evaluate_baseline(
                state,
                context.baseline_selector,
                samples,
                score_output,
                score_metric,
            )
            changed_selector = SiteSelector.none()
            selected_groups: list[str] = []
            previous_score = baseline_score
            results: list[SensitivityPathResult] = []
            for step, group in enumerate(groups, start=1):
                new_selector = (
                    context.effective_selectors[group.name] & ~changed_selector
                )
                new_sites = _matching_site_paths(context.sites, new_selector)
                if not new_sites:
                    raise ValueError(
                        f"Cumulative sensitivity group {group.name!r} adds no new "
                        "quantization sites after earlier groups."
                    )
                changed_selector = (
                    changed_selector | context.effective_selectors[group.name]
                )
                selected_groups.append(group.name)
                outputs, score = self._evaluate_changed(
                    state,
                    context.baseline_selector,
                    changed_selector,
                    samples,
                    mode,
                    score_output,
                    score_metric,
                )
                selected_sites = _matching_site_paths(
                    context.sites,
                    changed_selector,
                )
                results.append(
                    SensitivityPathResult(
                        step=step,
                        group=group.name,
                        selected_groups=tuple(selected_groups),
                        outputs=outputs,
                        score=score,
                        cumulative_sensitivity=_score_improvement(
                            mode,
                            baseline_score,
                            score,
                        ),
                        incremental_sensitivity=_score_improvement(
                            mode,
                            previous_score,
                            score,
                        ),
                        matched_sites=new_sites,
                        selected_sites=selected_sites,
                    )
                )
                previous_score = score
        return baseline_outputs, results

    def run_greedy(
        self,
        samples: Sequence[ModelInput],
        groups: Sequence[QuantizationGroup],
        *,
        mode: SensitivityMode = SensitivityMode.LEAVE_ONE_FLOAT,
        score_output: str,
        score_metric: str = "mae",
        baseline_selector: SiteSelector | None = None,
        max_steps: int | None = None,
        minimum_improvement: float = 0.0,
    ) -> tuple[dict[str, dict[str, float | int | None]], list[SensitivityPathResult],]:
        """Greedily select the best next group after every accumulated step.

        Remaining groups are re-evaluated after each selection. Search stops at
        ``max_steps`` or when the best incremental improvement is not greater
        than ``minimum_improvement``. The original group order provides a stable
        tie-breaker.
        """
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be nonnegative or None.")
        if not math.isfinite(minimum_improvement):
            raise ValueError("minimum_improvement must be finite.")

        context = self._prepare_context(
            samples,
            groups,
            mode=mode,
            baseline_selector=baseline_selector,
        )
        step_limit = len(groups) if max_steps is None else min(max_steps, len(groups))
        with FakeQuantState(self.quantized_model) as state:
            baseline_outputs, baseline_score = self._evaluate_baseline(
                state,
                context.baseline_selector,
                samples,
                score_output,
                score_metric,
            )
            changed_selector = SiteSelector.none()
            selected_groups: list[str] = []
            remaining = list(groups)
            current_score = baseline_score
            results: list[SensitivityPathResult] = []

            for step in range(1, step_limit + 1):
                best: tuple[
                    QuantizationGroup,
                    SiteSelector,
                    tuple[str, ...],
                    dict[str, dict[str, float | int | None]],
                    float,
                    float,
                ] | None = None
                best_improvement = float("-inf")
                for group in remaining:
                    new_selector = (
                        context.effective_selectors[group.name] & ~changed_selector
                    )
                    new_sites = _matching_site_paths(context.sites, new_selector)
                    if not new_sites:
                        continue
                    candidate_selector = (
                        changed_selector | context.effective_selectors[group.name]
                    )
                    outputs, score = self._evaluate_changed(
                        state,
                        context.baseline_selector,
                        candidate_selector,
                        samples,
                        mode,
                        score_output,
                        score_metric,
                    )
                    improvement = _score_improvement(mode, current_score, score)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best = (
                            group,
                            candidate_selector,
                            new_sites,
                            outputs,
                            score,
                            improvement,
                        )

                if best is None or best_improvement <= minimum_improvement:
                    break

                group, changed_selector, new_sites, outputs, score, improvement = best
                selected_groups.append(group.name)
                selected_sites = _matching_site_paths(
                    context.sites,
                    changed_selector,
                )
                results.append(
                    SensitivityPathResult(
                        step=step,
                        group=group.name,
                        selected_groups=tuple(selected_groups),
                        outputs=outputs,
                        score=score,
                        cumulative_sensitivity=_score_improvement(
                            mode,
                            baseline_score,
                            score,
                        ),
                        incremental_sensitivity=improvement,
                        matched_sites=new_sites,
                        selected_sites=selected_sites,
                    )
                )
                current_score = score
                remaining.remove(group)

        return baseline_outputs, results

    def _prepare_context(
        self,
        samples: Sequence[ModelInput],
        groups: Sequence[QuantizationGroup],
        *,
        mode: SensitivityMode,
        baseline_selector: SiteSelector | None,
    ) -> _SensitivityContext:
        if not samples:
            raise ValueError("Sensitivity analysis requires at least one sample.")
        if not groups:
            raise ValueError("Sensitivity analysis requires at least one group.")
        group_names = tuple(group.name for group in groups)
        if len(set(group_names)) != len(group_names):
            raise ValueError("Sensitivity group names must be unique.")

        sites = tuple(iter_quantization_sites(self.quantized_model))
        if not sites:
            raise ValueError("The candidate model does not contain WrapQ observers.")
        selected_baseline = (
            baseline_selector
            if baseline_selector is not None
            else _default_baseline_selector(mode)
        )
        effective_selectors = {
            group.name: (
                group.selector & selected_baseline
                if mode is SensitivityMode.LEAVE_ONE_FLOAT
                else group.selector & ~selected_baseline
            )
            for group in groups
        }
        matched_sites = {
            group.name: _matching_site_paths(
                sites,
                effective_selectors[group.name],
            )
            for group in groups
        }
        empty_groups = tuple(name for name, paths in matched_sites.items() if not paths)
        if empty_groups:
            raise ValueError(
                "Sensitivity selectors matched no quantization sites that can "
                f"change relative to the baseline for groups: {empty_groups}."
            )
        return _SensitivityContext(
            sites=sites,
            baseline_selector=selected_baseline,
            effective_selectors=effective_selectors,
            matched_sites=matched_sites,
        )

    def _evaluate_baseline(
        self,
        state: FakeQuantState,
        baseline_selector: SiteSelector,
        samples: Sequence[ModelInput],
        score_output: str,
        score_metric: str,
    ) -> tuple[dict[str, dict[str, float | int | None]], float]:
        _apply_baseline(state, baseline_selector)
        outputs = evaluate_models(
            self.reference_model,
            self.quantized_model,
            samples,
            output_adapter=self.output_adapter,
        )
        return outputs, _metric_value(outputs, score_output, score_metric)

    def _evaluate_changed(
        self,
        state: FakeQuantState,
        baseline_selector: SiteSelector,
        changed_selector: SiteSelector,
        samples: Sequence[ModelInput],
        mode: SensitivityMode,
        score_output: str,
        score_metric: str,
    ) -> tuple[dict[str, dict[str, float | int | None]], float]:
        _apply_baseline(state, baseline_selector)
        state.set_where(
            changed_selector,
            mode is SensitivityMode.ENABLE_ONE,
        )
        outputs = evaluate_models(
            self.reference_model,
            self.quantized_model,
            samples,
            output_adapter=self.output_adapter,
        )
        return outputs, _metric_value(outputs, score_output, score_metric)


def _default_baseline_selector(mode: SensitivityMode) -> SiteSelector:
    if mode is SensitivityMode.LEAVE_ONE_FLOAT:
        return SiteSelector.all()
    return SiteSelector.none()


def _apply_baseline(state: FakeQuantState, selector: SiteSelector) -> None:
    state.set_all(False)
    state.set_where(selector, True)


def _matching_site_paths(
    sites: Sequence[QuantizationSite],
    selector: SiteSelector,
) -> tuple[str, ...]:
    return tuple(site.path for site in sites if selector(site))


def _score_improvement(
    mode: SensitivityMode,
    previous_score: float,
    candidate_score: float,
) -> float:
    if mode is SensitivityMode.LEAVE_ONE_FLOAT:
        return previous_score - candidate_score
    return candidate_score - previous_score


def _metric_value(
    outputs: Mapping[str, Mapping[str, float | int | None]],
    output_name: str,
    metric_name: str,
) -> float:
    if output_name not in outputs:
        raise KeyError(
            f"Unknown output {output_name!r}; available outputs: {tuple(outputs)}."
        )
    value = outputs[output_name].get(metric_name)
    if not isinstance(value, (float, int)):
        raise KeyError(
            f"Metric {metric_name!r} for output {output_name!r} is not numeric."
        )
    return float(value)
