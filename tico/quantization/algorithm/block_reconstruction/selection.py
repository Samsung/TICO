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

"""Held-out end-to-end candidate selection for block reconstruction."""

from __future__ import annotations

import math

from collections.abc import Mapping
from dataclasses import dataclass, field


OutputMetrics = Mapping[str, Mapping[str, float | int | None]]


@dataclass(frozen=True)
class ValidationObjective:
    """Select a lower primary metric subject to output non-regression guards."""

    primary_output: str = "regressors"
    primary_metric: str = "mae"
    minimum_improvement: float = 0.0
    output_tolerances: Mapping[str, float] = field(
        default_factory=lambda: {"classifiers": 0.0}
    )

    def __post_init__(self) -> None:
        if not self.primary_output:
            raise ValueError("primary_output must be non-empty.")
        if not self.primary_metric:
            raise ValueError("primary_metric must be non-empty.")
        if (
            not math.isfinite(self.minimum_improvement)
            or self.minimum_improvement < 0.0
        ):
            raise ValueError("minimum_improvement must be finite and nonnegative.")
        for output_name, tolerance in self.output_tolerances.items():
            if not output_name:
                raise ValueError("Output-tolerance names must be non-empty.")
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError(
                    "Every output tolerance must be finite and nonnegative."
                )

    def score(self, outputs: OutputMetrics) -> float:
        """Return the primary score; lower is better."""
        return _metric(outputs, self.primary_output, self.primary_metric)

    def admissible(
        self,
        candidate: OutputMetrics,
        reference: OutputMetrics,
    ) -> tuple[bool, str]:
        """Check all auxiliary-output constraints against the window entry state."""
        for output_name, tolerance in self.output_tolerances.items():
            reference_value = _metric(
                reference,
                output_name,
                self.primary_metric,
            )
            candidate_value = _metric(
                candidate,
                output_name,
                self.primary_metric,
            )
            regression = candidate_value - reference_value
            if regression > tolerance:
                return (
                    False,
                    f"{output_name}.{self.primary_metric} regressed by "
                    f"{regression:.6e}, exceeding tolerance "
                    f"{tolerance:.6e}",
                )
        return True, "all output constraints satisfied"

    def better(
        self,
        candidate: OutputMetrics,
        incumbent: OutputMetrics,
        reference: OutputMetrics,
    ) -> tuple[bool, str]:
        """Return whether a checkpoint should replace the current best state."""
        admissible, reason = self.admissible(candidate, reference)
        if not admissible:
            return False, reason
        improvement = self.score(incumbent) - self.score(candidate)
        if improvement <= 0.0:
            return (
                False,
                f"primary improvement {improvement:.6e} was not positive",
            )
        return True, f"primary improvement {improvement:.6e}"

    def accepted(
        self,
        candidate: OutputMetrics,
        reference: OutputMetrics,
    ) -> tuple[bool, str]:
        """Return whether the selected state should replace the entry state."""
        admissible, reason = self.admissible(candidate, reference)
        if not admissible:
            return False, reason
        improvement = self.score(reference) - self.score(candidate)
        if improvement <= self.minimum_improvement:
            return (
                False,
                f"primary improvement {improvement:.6e} did not exceed "
                f"{self.minimum_improvement:.6e}",
            )
        return True, f"primary improvement {improvement:.6e}"


def copy_outputs(outputs: OutputMetrics) -> dict[str, dict[str, float | int | None]]:
    """Return a mutable JSON-compatible copy of output metrics."""
    return {name: dict(metrics) for name, metrics in outputs.items()}


def metric_value(
    outputs: OutputMetrics,
    output_name: str,
    metric_name: str,
) -> float:
    """Read one numeric output metric."""
    return _metric(outputs, output_name, metric_name)


def _metric(outputs: OutputMetrics, output_name: str, metric_name: str) -> float:
    if output_name not in outputs:
        raise KeyError(
            f"Unknown output {output_name!r}; available outputs: " f"{tuple(outputs)}."
        )
    value = outputs[output_name].get(metric_name)
    if not isinstance(value, (float, int)):
        raise KeyError(
            f"Metric {metric_name!r} for output {output_name!r} is not numeric."
        )
    return float(value)
