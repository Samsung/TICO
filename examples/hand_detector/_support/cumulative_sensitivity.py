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

"""Cumulative activation-sensitivity reporting for the palm detector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from tico.quantization.analysis import SensitivityPathResult

from examples.hand_detector._support.sensitivity import ActivationSensitivityGroup


def select_activation_sensitivity_groups(
    groups: Sequence[ActivationSensitivityGroup],
    names: Sequence[str] | None,
) -> tuple[ActivationSensitivityGroup, ...]:
    """Select semantic groups by stable name in the requested order."""
    available = {group.name: group for group in groups}
    if names is None:
        return tuple(groups)

    requested = tuple(names)
    if not requested:
        raise ValueError("At least one activation sensitivity group is required.")
    duplicated = tuple(
        sorted(name for name in set(requested) if requested.count(name) > 1)
    )
    if duplicated:
        raise ValueError(f"Activation sensitivity group names repeat: {duplicated}.")
    unknown = tuple(name for name in requested if name not in available)
    if unknown:
        raise ValueError(
            f"Unknown activation sensitivity groups: {unknown}; "
            f"available groups: {tuple(available)}."
        )
    return tuple(available[name] for name in requested)


def build_activation_sensitivity_path_report(
    *,
    baseline: Mapping[str, Mapping[str, float | int | None]],
    results: Sequence[SensitivityPathResult],
    groups: Sequence[ActivationSensitivityGroup],
) -> list[dict[str, object]]:
    """Attach semantic metadata and output-specific gains to path steps."""
    metadata = {group.name: group for group in groups}
    previous_outputs = baseline
    report: list[dict[str, object]] = []
    for result in results:
        group = metadata[result.group]
        value = result.to_dict()
        value.update(group.to_dict())
        value["regressor_mae_improvement"] = _mae_improvement(
            baseline,
            result.outputs,
            "regressors",
        )
        value["classifier_mae_improvement"] = _mae_improvement(
            baseline,
            result.outputs,
            "classifiers",
        )
        value["incremental_regressor_mae_improvement"] = _mae_improvement(
            previous_outputs,
            result.outputs,
            "regressors",
        )
        value["incremental_classifier_mae_improvement"] = _mae_improvement(
            previous_outputs,
            result.outputs,
            "classifiers",
        )
        report.append(value)
        previous_outputs = result.outputs
    return report


def print_activation_sensitivity_path(
    *,
    strategy: str,
    dtype_name: str,
    percentile: float,
    baseline: Mapping[str, Mapping[str, float | int | None]],
    results: Sequence[SensitivityPathResult],
    baseline_site_count: int,
    score_output: str,
) -> None:
    """Print cumulative, ranked, or greedy leave-float path steps."""
    baseline_reg = float(cast(float, baseline["regressors"]["mae"]))
    baseline_cls = float(cast(float, baseline["classifiers"]["mae"]))
    print(
        f"\n{dtype_name.upper()} P{percentile:g} activation " f"{strategy} sensitivity"
    )
    print(
        "Baseline E:internal-full: "
        f"REG_MAE={baseline_reg:.6e}, "
        f"CLS_MAE={baseline_cls:.6e}, "
        f"SITES={baseline_site_count}"
    )
    if strategy == "greedy":
        print(
            f"Each step re-ranks remaining groups by incremental {score_output} "
            "MAE improvement."
        )
    elif strategy == "ranked":
        print(
            f"Groups follow the initial independent {score_output} MAE ranking; "
            "the ranking is not recomputed."
        )
    else:
        print("Groups are accumulated in the requested order.")

    print(
        f"{'step':>4s} {'added_group':34s} "
        f"{'REG_MAE':>13s} {'DELTA_REG':>13s} {'TOTAL_REG':>13s} "
        f"{'CLS_MAE':>13s} {'DELTA_CLS':>13s} {'TOTAL_CLS':>13s} "
        f"{'FLOAT_SITES':>11s}"
    )
    previous_reg = baseline_reg
    previous_cls = baseline_cls
    for result in results:
        regressor_mae = float(cast(float, result.outputs["regressors"]["mae"]))
        classifier_mae = float(cast(float, result.outputs["classifiers"]["mae"]))
        print(
            f"{result.step:4d} "
            f"{result.group[:34]:34s} "
            f"{regressor_mae:13.6e} "
            f"{previous_reg - regressor_mae:13.6e} "
            f"{baseline_reg - regressor_mae:13.6e} "
            f"{classifier_mae:13.6e} "
            f"{previous_cls - classifier_mae:13.6e} "
            f"{baseline_cls - classifier_mae:13.6e} "
            f"{result.selected_site_count:11d}"
        )
        previous_reg = regressor_mae
        previous_cls = classifier_mae
    if not results:
        print("No group satisfied the path selection criterion.")


def _mae_improvement(
    baseline: Mapping[str, Mapping[str, float | int | None]],
    outputs: Mapping[str, Mapping[str, float | int | None]],
    output_name: str,
) -> float:
    baseline_mae = cast(float, baseline[output_name]["mae"])
    output_mae = cast(float, outputs[output_name]["mae"])
    return float(baseline_mae) - float(output_mae)
