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

"""Block-wise activation sensitivity helpers for the palm detector."""

from __future__ import annotations

import re

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector.hand_detector import HandDetector, NHWCInputAdapter
from tico.quantization.analysis import (
    QuantizationBoundaries,
    QuantizationGroup,
    QuantizationProfile,
    SensitivityResult,
    SiteSelector,
)
from tico.quantization.wrapq.control import (
    iter_quantization_sites,
    QuantizationSite,
    SiteRole,
)

from torch import nn


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_PASSTHROUGH_HEAD_OPS = frozenset({"RESHAPE"})


@dataclass(frozen=True)
class ActivationSensitivityGroup:
    """Describe one semantic block and its internal activation domains."""

    group: QuantizationGroup
    kind: str
    operation_positions: tuple[int, ...]
    operation_indices: tuple[int, ...]
    operation_names: tuple[str, ...]
    tensor_ids: tuple[int, ...]
    site_paths: tuple[str, ...]

    @property
    def name(self) -> str:
        """Return the stable group name used by sensitivity reports."""
        return self.group.name

    @property
    def site_count(self) -> int:
        """Return the number of activation sites assigned to this block."""
        return len(self.site_paths)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible group metadata."""
        return {
            "group": self.name,
            "kind": self.kind,
            "operation_positions": list(self.operation_positions),
            "operation_indices": list(self.operation_indices),
            "operation_names": list(self.operation_names),
            "tensor_ids": list(self.tensor_ids),
            "site_count": self.site_count,
        }


@dataclass(frozen=True)
class _OperationGroup:
    name: str
    kind: str
    positions: tuple[int, ...]


def build_activation_sensitivity_groups(
    model: nn.Module,
    boundaries: QuantizationBoundaries,
) -> tuple[ActivationSensitivityGroup, ...]:
    """Group every internal activation site by its logical producer block.

    Producer ``act_out`` and consumer ``act_in`` sites for the same tensor are
    assigned to the producer block whenever the wrappers expose separate
    observers. Shared-domain wrappers such as MaxPool remain in the block that
    owns the operation because one observer covers both sides of the operator.
    """
    detector = _find_detector(model)
    operation_groups = _partition_operations(detector)
    operation_to_group = {
        position: group for group in operation_groups for position in group.positions
    }
    producer_by_tensor = _producer_positions(detector.operations)

    sites = tuple(iter_quantization_sites(model))
    activation_selector = boundaries.selector_for(QuantizationProfile.ACTIVATION_ONLY)
    target_sites = tuple(site for site in sites if activation_selector(site))
    if not target_sites:
        raise ValueError("No internal activation sites were selected for sensitivity.")

    assignments: dict[str, list[QuantizationSite]] = defaultdict(list)
    tensor_ids: dict[str, set[int]] = defaultdict(set)
    input_group = _OperationGroup("input_boundary", "input", ())
    for site in target_sites:
        group, connected_tensors = _site_group(
            site,
            detector,
            operation_to_group,
            producer_by_tensor,
            input_group,
        )
        assignments[group.name].append(site)
        tensor_ids[group.name].update(connected_tensors)

    ordered_groups = (input_group, *operation_groups)
    results: list[ActivationSensitivityGroup] = []
    assigned_paths: list[str] = []
    for group in ordered_groups:
        grouped_sites = assignments.get(group.name, ())
        if not grouped_sites:
            continue
        paths = tuple(sorted(site.path for site in grouped_sites))
        path_set = frozenset(paths)
        selector = SiteSelector(
            lambda site, selected=path_set: site.path in selected,
            f"exact_paths[{group.name}]",
        )
        operations = tuple(
            detector.operations[position] for position in group.positions
        )
        results.append(
            ActivationSensitivityGroup(
                group=QuantizationGroup(group.name, selector),
                kind=group.kind,
                operation_positions=group.positions,
                operation_indices=tuple(int(op["index"]) for op in operations),
                operation_names=tuple(str(op["name"]) for op in operations),
                tensor_ids=tuple(sorted(tensor_ids[group.name])),
                site_paths=paths,
            )
        )
        assigned_paths.extend(paths)

    expected_paths = tuple(sorted(site.path for site in target_sites))
    actual_paths = tuple(sorted(assigned_paths))
    if actual_paths != expected_paths:
        missing = tuple(sorted(set(expected_paths) - set(actual_paths)))
        duplicated = tuple(
            sorted(path for path in set(actual_paths) if actual_paths.count(path) > 1)
        )
        raise RuntimeError(
            "Activation sensitivity groups must cover every internal activation "
            f"site exactly once; missing={missing}, duplicated={duplicated}."
        )
    return tuple(results)


def build_activation_sensitivity_report(
    *,
    baseline: Mapping[str, Mapping[str, float | int | None]],
    results: Sequence[SensitivityResult],
    groups: Sequence[ActivationSensitivityGroup],
) -> list[dict[str, object]]:
    """Attach block metadata and both output-MAE improvements to results."""
    metadata = {group.name: group for group in groups}
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
        report.append(value)
    return report


def print_activation_sensitivity(
    *,
    dtype_name: str,
    percentile: float,
    baseline: Mapping[str, Mapping[str, float | int | None]],
    results: Sequence[SensitivityResult],
    top_k: int,
    baseline_site_count: int,
    score_output: str,
) -> None:
    """Print leave-one-float activation groups ranked by the selected output."""
    if top_k < 0:
        raise ValueError("top_k must be nonnegative.")
    shown = results if top_k == 0 else results[:top_k]
    baseline_reg = float(baseline["regressors"]["mae"])
    baseline_cls = float(baseline["classifiers"]["mae"])
    print(f"\n{dtype_name.upper()} P{percentile:g} activation block sensitivity")
    print(
        "Baseline E:internal-full: "
        f"REG_MAE={baseline_reg:.6e}, "
        f"CLS_MAE={baseline_cls:.6e}, "
        f"SITES={baseline_site_count}"
    )
    print(f"Groups are ranked by {score_output} MAE improvement when left float.")
    print(
        f"{'rank':>4s} {'group':34s} {'REG_MAE':>13s} {'GAIN_REG':>13s} "
        f"{'CLS_MAE':>13s} {'GAIN_CLS':>13s} {'SITES':>7s}"
    )
    for index, result in enumerate(shown, start=1):
        regressor_mae = float(result.outputs["regressors"]["mae"])
        classifier_mae = float(result.outputs["classifiers"]["mae"])
        print(
            f"{index:4d} "
            f"{result.group[:34]:34s} "
            f"{regressor_mae:13.6e} "
            f"{baseline_reg - regressor_mae:13.6e} "
            f"{classifier_mae:13.6e} "
            f"{baseline_cls - classifier_mae:13.6e} "
            f"{len(result.matched_sites):7d}"
        )
    if top_k and len(results) > top_k:
        print(f"Showing {top_k} of {len(results)} groups; JSON contains all groups.")


def _partition_operations(detector: HandDetector) -> tuple[_OperationGroup, ...]:
    head_groups, reserved_positions = _head_operation_groups(detector)
    feature_groups: list[_OperationGroup] = []
    current: list[int] = []
    feature_index = 0
    for position, operation in enumerate(detector.operations):
        if position in reserved_positions:
            continue
        current.append(position)
        if operation["name"] == "PRELU":
            if not feature_groups:
                name = "stem"
                kind = "stem"
            else:
                name = f"feature_block_{feature_index:02d}"
                kind = "feature"
                feature_index += 1
            feature_groups.append(_OperationGroup(name, kind, tuple(current)))
            current = []
    if current:
        feature_groups.append(
            _OperationGroup(
                f"feature_block_{feature_index:02d}",
                "feature",
                tuple(current),
            )
        )
    return tuple((*feature_groups, *head_groups))


def _head_operation_groups(
    detector: HandDetector,
) -> tuple[tuple[_OperationGroup, ...], frozenset[int]]:
    producers = _producer_positions(detector.operations)
    head_groups: list[_OperationGroup] = []
    reserved: set[int] = set()
    for output_tensor, output_name in zip(
        detector.output_tensors,
        OUTPUT_NAMES,
    ):
        final_position = producers.get(output_tensor)
        if final_position is None:
            raise RuntimeError(f"No operation produces output tensor {output_tensor}.")
        final_operation = detector.operations[final_position]
        if final_operation["name"] != "CONCATENATION":
            raise RuntimeError(
                f"Expected final output tensor {output_tensor} to be concatenated."
            )
        reserved.add(final_position)
        branches: list[tuple[int, tuple[int, ...]]] = []
        for input_tensor in final_operation["inputs"]:
            source_position, branch_positions = _trace_head_branch(
                int(input_tensor),
                detector.operations,
                producers,
            )
            branches.append((source_position, branch_positions))
            reserved.update(branch_positions)

        branches.sort(key=lambda item: item[0])
        for branch_index, (_, positions) in enumerate(branches):
            scale_name = _resolution_name(branch_index, len(branches))
            head_groups.append(
                _OperationGroup(
                    f"{output_name}_{scale_name}_head",
                    "head",
                    positions,
                )
            )
    head_groups.sort(key=lambda group: min(group.positions))
    return tuple(head_groups), frozenset(reserved)


def _trace_head_branch(
    tensor_id: int,
    operations: Sequence[Mapping[str, Any]],
    producers: Mapping[int, int],
) -> tuple[int, tuple[int, ...]]:
    positions: list[int] = []
    current_tensor = tensor_id
    while True:
        position = producers.get(current_tensor)
        if position is None:
            raise RuntimeError(f"No operation produces head tensor {current_tensor}.")
        operation = operations[position]
        positions.append(position)
        if operation["name"] not in _PASSTHROUGH_HEAD_OPS:
            break
        current_tensor = int(operation["inputs"][0])
    if operations[position]["name"] not in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
        raise RuntimeError(
            "Expected each final output branch to originate from a convolution, "
            f"but found {operations[position]['name']!r}."
        )
    return position, tuple(sorted(positions))


def _resolution_name(index: int, count: int) -> str:
    if count == 1:
        return "single_resolution"
    if count == 2:
        return "low_resolution" if index == 0 else "high_resolution"
    return f"resolution_{index:02d}"


def _site_group(
    site: QuantizationSite,
    detector: HandDetector,
    operation_to_group: Mapping[int, _OperationGroup],
    producer_by_tensor: Mapping[int, int],
    input_group: _OperationGroup,
) -> tuple[_OperationGroup, tuple[int, ...]]:
    module_name = getattr(site.module, "fp_name", None) or site.module_path
    if "input_quantizer" in module_name:
        return input_group, (detector.input_tensor,)

    match = _LAYER_PATTERN.search(module_name)
    if match is None:
        raise RuntimeError(
            f"Cannot map activation site {site.path!r} to a detector layer."
        )
    position = int(match.group(1))
    if position >= len(detector.operations):
        raise RuntimeError(f"Detector layer position {position} is out of range.")
    operation = detector.operations[position]

    if site.role is SiteRole.ACTIVATION_INPUT:
        input_tensor = int(operation["inputs"][0])
        # MaxPool exposes one observer on both sides, so keep the shared domain
        # in the block that owns the pooling operation.
        if operation["name"] == "MAX_POOL_2D":
            return _operation_group(operation_to_group, position), (
                input_tensor,
                *(int(value) for value in operation["outputs"]),
            )
        producer_position = producer_by_tensor.get(input_tensor)
        if producer_position is None:
            return input_group, (input_tensor,)
        return _operation_group(operation_to_group, producer_position), (input_tensor,)

    if site.role in {SiteRole.ACTIVATION_OUTPUT, SiteRole.ACTIVATION}:
        return _operation_group(operation_to_group, position), tuple(
            int(value) for value in operation["outputs"]
        )

    raise RuntimeError(
        f"Internal activation selector included unsupported site {site.path!r} "
        f"with role {site.role.value!r}."
    )


def _operation_group(
    operation_to_group: Mapping[int, _OperationGroup],
    position: int,
) -> _OperationGroup:
    group = operation_to_group.get(position)
    if group is None:
        raise RuntimeError(
            f"Detector operation position {position} is not assigned to a block."
        )
    return group


def _producer_positions(
    operations: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    producers: dict[int, int] = {}
    for position, operation in enumerate(operations):
        for output in operation["outputs"]:
            tensor_id = int(output)
            if tensor_id in producers:
                raise RuntimeError(f"Tensor {tensor_id} has multiple producers.")
            producers[tensor_id] = position
    return producers


def _mae_improvement(
    baseline: Mapping[str, Mapping[str, float | int | None]],
    outputs: Mapping[str, Mapping[str, float | int | None]],
    output_name: str,
) -> float:
    return float(baseline[output_name]["mae"]) - float(outputs[output_name]["mae"])


def _find_detector(model: nn.Module) -> HandDetector:
    if isinstance(model, NHWCInputAdapter):
        return model.detector
    if isinstance(model, HandDetector):
        return model
    detector = getattr(model, "detector", None)
    if isinstance(detector, HandDetector):
        return detector
    raise TypeError("Expected HandDetector or NHWCInputAdapter.")
