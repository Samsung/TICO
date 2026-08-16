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

"""Execution-order activation block reconstruction for the palm detector."""

from __future__ import annotations

import re
import weakref

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector._support.sensitivity import (
    ActivationSensitivityGroup,
    build_activation_sensitivity_groups,
)
from examples.hand_detector.hand_detector import HandDetector
from tico.quantization.algorithm.block_reconstruction import (
    AffineObserverGroup,
    BlockInvocation,
    BlockReconstructionConfig,
    BlockReconstructionResult,
    BlockReconstructor,
    ReconstructionCache,
    ReconstructionSample,
)
from tico.quantization.analysis import (
    evaluate_models,
    OutputAdapter,
    QuantizationBoundaries,
    QuantizationProfile,
)
from tico.quantization.analysis.inputs import ModelInput
from tico.quantization.wrapq.control import (
    FakeQuantState,
    iter_quantization_sites,
    QuantizationSite,
    SiteRole,
)

from torch import nn


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_ACTIVATION_ROLES = frozenset(
    {
        SiteRole.ACTIVATION_INPUT,
        SiteRole.ACTIVATION_OUTPUT,
        SiteRole.ACTIVATION,
    }
)


@dataclass(frozen=True)
class HandDetectorReconstructionBlock:
    """Describe one executable feature block and its logical qparam groups."""

    name: str
    kind: str
    operation_positions: tuple[int, ...]
    operation_indices: tuple[int, ...]
    operation_names: tuple[str, ...]
    input_tensor_ids: tuple[int, ...]
    output_tensor_ids: tuple[int, ...]
    observer_groups: tuple[AffineObserverGroup, ...]
    reference_module: nn.Module
    quantized_module: nn.Module

    @property
    def site_paths(self) -> tuple[str, ...]:
        """Return all tied observer sites optimized by this block."""
        return tuple(
            path for group in self.observer_groups for path in group.site_paths
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible block metadata."""
        return {
            "block": self.name,
            "kind": self.kind,
            "operation_positions": list(self.operation_positions),
            "operation_indices": list(self.operation_indices),
            "operation_names": list(self.operation_names),
            "input_tensor_ids": list(self.input_tensor_ids),
            "output_tensor_ids": list(self.output_tensor_ids),
            "qparam_group_count": len(self.observer_groups),
            "qparam_groups": [
                {
                    "name": group.name,
                    "site_paths": list(group.site_paths),
                }
                for group in self.observer_groups
            ],
        }


@dataclass(frozen=True)
class HandDetectorReconstructionStep:
    """Store one local reconstruction and the resulting full-model metrics."""

    block: HandDetectorReconstructionBlock
    local: BlockReconstructionResult
    outputs: Mapping[str, Mapping[str, float | int | None]]

    def to_dict(
        self,
        baseline: Mapping[str, Mapping[str, float | int | None]],
    ) -> dict[str, object]:
        """Return JSON-compatible local and cumulative model results."""
        value = self.block.to_dict()
        value["reconstruction"] = self.local.to_dict()
        value["outputs"] = {
            name: dict(metrics) for name, metrics in self.outputs.items()
        }
        value["regressor_mae_improvement"] = _mae(baseline, "regressors") - _mae(
            self.outputs,
            "regressors",
        )
        value["classifier_mae_improvement"] = _mae(baseline, "classifiers") - _mae(
            self.outputs,
            "classifiers",
        )
        return value


class _DetectorSubgraph(nn.Module):
    """Execute selected detector operations from explicit boundary tensors."""

    def __init__(
        self,
        detector: HandDetector,
        positions: Sequence[int],
        input_tensor_ids: Sequence[int],
        output_tensor_ids: Sequence[int],
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_detector_reference", weakref.ref(detector))
        self.positions = tuple(positions)
        self.input_tensor_ids = tuple(input_tensor_ids)
        self.output_tensor_ids = tuple(output_tensor_ids)

    @property
    def detector(self) -> HandDetector:
        """Return the detector referenced without registering it as a child."""
        detector = self._detector_reference()
        if detector is None:
            raise RuntimeError("The detector used by this block no longer exists.")
        return detector

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.input_tensor_ids):
            raise ValueError(
                f"Expected {len(self.input_tensor_ids)} block inputs, "
                f"but received {len(inputs)}."
            )
        values = self.detector.execute_segment(
            dict(zip(self.input_tensor_ids, inputs)),
            self.positions,
        )
        return tuple(values[tensor_id] for tensor_id in self.output_tensor_ids)


def build_hand_detector_reconstruction_blocks(
    reference_model: nn.Module,
    quantized_model: nn.Module,
    boundaries: QuantizationBoundaries,
    *,
    group_names: Sequence[str],
) -> tuple[HandDetectorReconstructionBlock, ...]:
    """Build selected stem/feature blocks in model execution order."""
    if not group_names:
        raise ValueError("At least one reconstruction group name is required.")
    reference_detector = _find_detector(reference_model)
    quantized_detector = _find_detector(quantized_model)
    all_groups = tuple(
        group
        for group in build_activation_sensitivity_groups(
            quantized_model,
            boundaries,
        )
        if group.kind in {"stem", "feature"}
    )
    by_name = {group.name: group for group in all_groups}
    requested = tuple(group_names)
    if len(set(requested)) != len(requested):
        raise ValueError("Reconstruction group names must be unique.")
    unknown = tuple(name for name in requested if name not in by_name)
    if unknown:
        raise KeyError(
            f"Unknown reconstruction groups {unknown}; available groups: "
            f"{tuple(by_name)}."
        )
    selected = tuple(
        sorted(
            (by_name[name] for name in requested),
            key=lambda group: min(group.operation_positions),
        )
    )

    sites = {site.path: site for site in iter_quantization_sites(quantized_model)}
    blocks = []
    for group in selected:
        positions = tuple(sorted(group.operation_positions))
        input_tensor_ids, output_tensor_ids = _boundary_tensor_ids(
            quantized_detector,
            positions,
        )
        observer_groups = _logical_observer_groups(
            group,
            sites,
            quantized_detector,
            positions,
        )
        operations = tuple(
            quantized_detector.operations[position] for position in positions
        )
        blocks.append(
            HandDetectorReconstructionBlock(
                name=group.name,
                kind=group.kind,
                operation_positions=positions,
                operation_indices=tuple(int(op["index"]) for op in operations),
                operation_names=tuple(str(op["name"]) for op in operations),
                input_tensor_ids=input_tensor_ids,
                output_tensor_ids=output_tensor_ids,
                observer_groups=observer_groups,
                reference_module=_DetectorSubgraph(
                    reference_detector,
                    positions,
                    input_tensor_ids,
                    output_tensor_ids,
                ),
                quantized_module=_DetectorSubgraph(
                    quantized_detector,
                    positions,
                    input_tensor_ids,
                    output_tensor_ids,
                ),
            )
        )
    return tuple(blocks)


def collect_hand_detector_reconstruction_cache(
    block: HandDetectorReconstructionBlock,
    reference_model: nn.Module,
    quantized_model: nn.Module,
    samples: Sequence[ModelInput],
) -> ReconstructionCache:
    """Collect float targets plus float and quantized-prefix boundary inputs."""
    if not samples:
        raise ValueError("Block reconstruction cache requires at least one sample.")
    reference_detector = _find_detector(reference_model)
    quantized_detector = _find_detector(quantized_model)
    cached: list[ReconstructionSample] = []
    with torch.inference_mode():
        for sample in samples:
            if not isinstance(sample, torch.Tensor):
                raise TypeError(
                    "The hand detector reconstruction example expects Tensor samples."
                )
            reference_values = reference_detector.forward_nhwc_values(sample)
            quantized_values = quantized_detector.forward_nhwc_values(sample)
            cached.append(
                ReconstructionSample(
                    float_input=BlockInvocation(
                        args=tuple(
                            reference_values[tensor_id]
                            for tensor_id in block.input_tensor_ids
                        )
                    ),
                    quantized_input=BlockInvocation(
                        args=tuple(
                            quantized_values[tensor_id]
                            for tensor_id in block.input_tensor_ids
                        )
                    ),
                    target=tuple(
                        reference_values[tensor_id]
                        for tensor_id in block.output_tensor_ids
                    ),
                )
            )
    return ReconstructionCache(cached)


def reconstruct_hand_detector_blocks(
    reference_model: nn.Module,
    quantized_model: nn.Module,
    calibration_samples: Sequence[ModelInput],
    evaluation_samples: Sequence[ModelInput],
    *,
    boundaries: QuantizationBoundaries,
    output_adapter: OutputAdapter | None,
    config: BlockReconstructionConfig,
    group_names: Sequence[str],
) -> tuple[
    dict[str, dict[str, float | int | None]],
    int,
    tuple[HandDetectorReconstructionStep, ...],
]:
    """Reconstruct selected blocks sequentially from an E baseline."""
    blocks = build_hand_detector_reconstruction_blocks(
        reference_model,
        quantized_model,
        boundaries,
        group_names=group_names,
    )
    selector = boundaries.selector_for(QuantizationProfile.INTERNAL_FULL)
    all_sites = tuple(iter_quantization_sites(quantized_model))
    baseline_site_count = sum(selector(site) for site in all_sites)
    if baseline_site_count == 0:
        raise ValueError("E:internal-full did not select any quantization sites.")

    reconstructor = BlockReconstructor(config)
    with FakeQuantState(quantized_model) as state:
        state.set_all(False)
        state.set_where(selector, True)
        baseline = evaluate_models(
            reference_model,
            quantized_model,
            evaluation_samples,
            output_adapter=output_adapter,
        )
        steps = []
        for block in blocks:
            cache = collect_hand_detector_reconstruction_cache(
                block,
                reference_model,
                quantized_model,
                calibration_samples,
            )
            result = reconstructor.reconstruct(
                block_name=block.name,
                observer_model=quantized_model,
                block=block.quantized_module,
                cache=cache,
                observer_groups=block.observer_groups,
            )
            outputs = evaluate_models(
                reference_model,
                quantized_model,
                evaluation_samples,
                output_adapter=output_adapter,
            )
            steps.append(
                HandDetectorReconstructionStep(
                    block=block,
                    local=result,
                    outputs=outputs,
                )
            )
    return baseline, baseline_site_count, tuple(steps)


def print_hand_detector_reconstruction(
    *,
    dtype_name: str,
    percentile: float,
    baseline: Mapping[str, Mapping[str, float | int | None]],
    baseline_site_count: int,
    steps: Sequence[HandDetectorReconstructionStep],
) -> None:
    """Print local losses and cumulative E metrics after every block."""
    baseline_reg = _mae(baseline, "regressors")
    baseline_cls = _mae(baseline, "classifiers")
    print(f"\n{dtype_name.upper()} P{percentile:g} block reconstruction")
    print(
        "Baseline E:internal-full: "
        f"REG_MAE={baseline_reg:.6e}, "
        f"CLS_MAE={baseline_cls:.6e}, "
        f"SITES={baseline_site_count}"
    )
    print(
        f"{'step':>4s} {'block':30s} {'LOCAL_IN':>11s} {'LOCAL_OUT':>11s} "
        f"{'REG_MAE':>13s} {'GAIN_REG':>13s} "
        f"{'CLS_MAE':>13s} {'GAIN_CLS':>13s} {'QGROUPS':>7s}"
    )
    for index, step in enumerate(steps, start=1):
        regressor_mae = _mae(step.outputs, "regressors")
        classifier_mae = _mae(step.outputs, "classifiers")
        print(
            f"{index:4d} "
            f"{step.block.name[:30]:30s} "
            f"{step.local.initial_loss:11.4e} "
            f"{step.local.final_loss:11.4e} "
            f"{regressor_mae:13.6e} "
            f"{baseline_reg - regressor_mae:13.6e} "
            f"{classifier_mae:13.6e} "
            f"{baseline_cls - classifier_mae:13.6e} "
            f"{len(step.block.observer_groups):7d}"
        )


def build_hand_detector_reconstruction_report(
    *,
    baseline: Mapping[str, Mapping[str, float | int | None]],
    steps: Sequence[HandDetectorReconstructionStep],
) -> list[dict[str, object]]:
    """Serialize reconstruction steps with cumulative model improvements."""
    return [step.to_dict(baseline) for step in steps]


def _logical_observer_groups(
    group: ActivationSensitivityGroup,
    sites: Mapping[str, QuantizationSite],
    detector: HandDetector,
    positions: Sequence[int],
) -> tuple[AffineObserverGroup, ...]:
    """Tie producer/consumer observers that represent the same tensor domain."""
    selected_positions = frozenset(positions)
    entries: list[tuple[str, frozenset[int], int | None]] = []
    for path in group.site_paths:
        site = sites.get(path)
        if site is None:
            raise KeyError(f"Unknown activation site {path!r} in group {group.name!r}.")
        if site.role not in _ACTIVATION_ROLES:
            raise ValueError(
                f"Group {group.name!r} includes non-activation site {path!r}."
            )
        tensor_ids = frozenset(_site_tensor_ids(site, detector))
        position = _site_operation_position(site)
        entries.append((path, tensor_ids, position))

    parents = list(range(len(entries)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            if entries[left][1] & entries[right][1]:
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(entries)):
        components.setdefault(find(index), []).append(index)

    result = []
    for indices in components.values():
        if not any(
            entries[index][2] is not None and entries[index][2] in selected_positions
            for index in indices
        ):
            continue
        tensor_ids = sorted(set().union(*(entries[index][1] for index in indices)))
        name = (
            f"tensor_{tensor_ids[0]}"
            if len(tensor_ids) == 1
            else "tensors_" + "_".join(str(value) for value in tensor_ids)
        )
        result.append(
            AffineObserverGroup(
                name=f"{group.name}.{name}",
                site_paths=tuple(entries[index][0] for index in indices),
            )
        )
    if not result:
        raise ValueError(
            f"Reconstruction block {group.name!r} has no in-block activation qparams."
        )
    return tuple(result)


def _site_tensor_ids(
    site: QuantizationSite,
    detector: HandDetector,
) -> tuple[int, ...]:
    fp_name = getattr(site.module, "fp_name", None) or site.module_path
    if "input_quantizer" in fp_name:
        return (detector.input_tensor,)
    match = _LAYER_PATTERN.search(fp_name)
    if match is None:
        raise RuntimeError(f"Cannot map activation site {site.path!r} to a layer.")
    position = int(match.group(1))
    operation = detector.operations[position]
    inputs = tuple(int(value) for value in operation["inputs"])
    outputs = tuple(int(value) for value in operation["outputs"])
    if site.role is SiteRole.ACTIVATION_INPUT:
        if operation["name"] == "MAX_POOL_2D":
            return (inputs[0], *outputs)
        return inputs[:1]
    if site.role in {SiteRole.ACTIVATION_OUTPUT, SiteRole.ACTIVATION}:
        return outputs
    raise RuntimeError(f"Unsupported activation site role: {site.role.value!r}.")


def _site_operation_position(site: QuantizationSite) -> int | None:
    fp_name = getattr(site.module, "fp_name", None) or site.module_path
    match = _LAYER_PATTERN.search(fp_name)
    return None if match is None else int(match.group(1))


def _boundary_tensor_ids(
    detector: HandDetector,
    positions: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected = frozenset(positions)
    produced = {
        int(output)
        for position in selected
        for output in detector.operations[position]["outputs"]
    }
    external_inputs: list[int] = []
    produced_order: list[int] = []
    for position in sorted(selected):
        operation = detector.operations[position]
        for tensor_id in _activation_inputs(operation):
            if tensor_id not in produced and tensor_id not in external_inputs:
                external_inputs.append(tensor_id)
        for output in operation["outputs"]:
            tensor_id = int(output)
            if tensor_id not in produced_order:
                produced_order.append(tensor_id)

    consumers: dict[int, set[int]] = {}
    for position, operation in enumerate(detector.operations):
        for tensor_id in _activation_inputs(operation):
            consumers.setdefault(tensor_id, set()).add(position)
    external_outputs = tuple(
        tensor_id
        for tensor_id in produced_order
        if tensor_id in detector.output_tensors
        or any(position not in selected for position in consumers.get(tensor_id, ()))
    )
    if not external_inputs or not external_outputs:
        raise RuntimeError(
            "A reconstruction block must have at least one external input and output."
        )
    return tuple(external_inputs), external_outputs


def _activation_inputs(operation: Mapping[str, Any]) -> tuple[int, ...]:
    name = str(operation["name"])
    inputs = tuple(int(value) for value in operation["inputs"])
    if name == "ADD":
        return inputs[:2]
    if name == "CONCATENATION":
        return inputs
    return inputs[:1]


def _find_detector(model: nn.Module) -> HandDetector:
    for module in model.modules():
        if isinstance(module, HandDetector):
            return module
    raise TypeError("Expected a HandDetector inside the supplied model.")


def _mae(
    outputs: Mapping[str, Mapping[str, float | int | None]],
    output_name: str,
) -> float:
    if output_name not in OUTPUT_NAMES:
        raise KeyError(f"Unknown detector output: {output_name!r}.")
    value = outputs[output_name].get("mae")
    if not isinstance(value, (float, int)):
        raise TypeError(f"Output {output_name!r} does not contain numeric MAE.")
    return float(value)
