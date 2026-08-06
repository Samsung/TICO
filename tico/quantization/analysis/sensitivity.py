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

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from torch import nn

from tico.quantization.analysis.inputs import ModelInput
from tico.quantization.analysis.metrics import evaluate_models
from tico.quantization.analysis.outputs import OutputAdapter
from tico.quantization.analysis.selector import SiteSelector
from tico.quantization.wrapq.control import FakeQuantState, iter_quantization_sites


class SensitivityMode(str, Enum):
    """Describe how one group differs from the full-quantized baseline."""

    LEAVE_ONE_FLOAT = "leave_one_float"
    ENABLE_ONE = "enable_one"


@dataclass(frozen=True)
class QuantizationGroup:
    """Assign a stable name to one or more related quantization sites."""

    name: str
    selector: SiteSelector


@dataclass(frozen=True)
class SensitivityResult:
    """Store one group evaluation and its positive sensitivity score."""

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


class QuantizationSensitivity:
    """Rank model-defined groups by one output metric."""

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
    ) -> tuple[dict[str, Mapping[str, float | int | None]], list[SensitivityResult]]:
        """Evaluate and rank the supplied quantization groups."""
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
        matched_sites = {
            group.name: tuple(site.path for site in sites if group.selector(site))
            for group in groups
        }
        empty_groups = tuple(name for name, paths in matched_sites.items() if not paths)
        if empty_groups:
            raise ValueError(
                "Sensitivity selectors matched no quantization sites for groups: "
                f"{empty_groups}."
            )

        with FakeQuantState(self.quantized_model) as state:
            state.set_all(mode is SensitivityMode.LEAVE_ONE_FLOAT)
            baseline = evaluate_models(
                self.reference_model,
                self.quantized_model,
                samples,
                output_adapter=self.output_adapter,
            )
            baseline_score = _metric_value(baseline, score_output, score_metric)

            results: list[SensitivityResult] = []
            for group in groups:
                state.set_all(mode is SensitivityMode.LEAVE_ONE_FLOAT)
                state.set_where(
                    group.selector,
                    mode is SensitivityMode.ENABLE_ONE,
                )
                outputs = evaluate_models(
                    self.reference_model,
                    self.quantized_model,
                    samples,
                    output_adapter=self.output_adapter,
                )
                score = _metric_value(outputs, score_output, score_metric)
                sensitivity = (
                    baseline_score - score
                    if mode is SensitivityMode.LEAVE_ONE_FLOAT
                    else score - baseline_score
                )
                results.append(
                    SensitivityResult(
                        group=group.name,
                        outputs=outputs,
                        score=score,
                        sensitivity=sensitivity,
                        matched_sites=matched_sites[group.name],
                    )
                )

        results.sort(key=lambda result: result.sensitivity, reverse=True)
        return baseline, results


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
