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

"""Hand-detector caches, joint windows, and validated reconstruction flow."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from examples.hand_detector._support.analysis import output_boundaries
from examples.hand_detector._support.sensitivity import (
    build_activation_sensitivity_groups,
)
from examples.hand_detector.hand_detector import HandDetector, NHWCInputAdapter
from tico.quantization.algorithm.block_reconstruction import (
    AffineObserverGroup,
    BlockInvocation,
    BlockReconstructionConfig,
    BlockReconstructor,
    ReconstructionCache,
    ReconstructionSample,
    ValidationObjective,
)
from tico.quantization.analysis import (
    evaluate_models,
    OutputAdapter,
    QuantizationBoundaries,
    QuantizationProfile,
)
from tico.quantization.wrapq.control import (
    FakeQuantState,
    iter_quantization_sites,
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
class ReconstructionWindow:
    """Describe one single-group or joint semantic reconstruction window."""

    name: str
    group_names: tuple[str, ...]
    operation_positions: tuple[int, ...]
    input_tensor_ids: tuple[int, ...]
    output_tensor_ids: tuple[int, ...]
    site_paths: tuple[str, ...]

    @property
    def is_joint(self) -> bool:
        return len(self.group_names) > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "group_names": list(self.group_names),
            "is_joint": self.is_joint,
            "operation_positions": list(self.operation_positions),
            "input_tensor_ids": list(self.input_tensor_ids),
            "output_tensor_ids": list(self.output_tensor_ids),
            "site_count": len(self.site_paths),
            "site_paths": list(self.site_paths),
        }


class DetectorWindow(nn.Module):
    """Execute a selected static subgraph from cached live-in tensors."""

    def __init__(self, detector: HandDetector, window: ReconstructionWindow) -> None:
        super().__init__()
        # Keep the executable view from registering the full detector as a
        # second child-module tree. The candidate model remains the sole owner.
        object.__setattr__(self, "_detector", detector)
        self.window = window

    def forward(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.window.input_tensor_ids):
            raise ValueError(
                f"Window {self.window.name!r} expected "
                f"{len(self.window.input_tensor_ids)} inputs, got {len(inputs)}."
            )
        values = dict(zip(self.window.input_tensor_ids, inputs))
        for position in self.window.operation_positions:
            _execute_operation(
                self._detector.operations[position],
                self._detector.layers[position],
                values,
            )
        outputs = tuple(
            values[tensor_id] for tensor_id in self.window.output_tensor_ids
        )
        return outputs[0] if len(outputs) == 1 else outputs


def split_reconstruction_samples(
    calibration_samples: Sequence[torch.Tensor],
    selection_count: int,
    *,
    seed: int = 20260803,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Create deterministic disjoint optimization and held-out subsets."""
    if selection_count <= 0:
        raise ValueError("selection_count must be positive.")
    if selection_count >= len(calibration_samples):
        raise ValueError(
            "selection_count must be smaller than the calibration sample count."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(
        len(calibration_samples),
        generator=generator,
    ).tolist()
    selection_indices = frozenset(permutation[:selection_count])
    train = tuple(
        sample
        for index, sample in enumerate(calibration_samples)
        if index not in selection_indices
    )
    selection = tuple(
        sample
        for index, sample in enumerate(calibration_samples)
        if index in selection_indices
    )
    return train, selection


def build_reconstruction_windows(
    model: nn.Module,
    boundaries: QuantizationBoundaries,
    *,
    groups: Sequence[str] | None = None,
    windows: Sequence[str] | None = None,
) -> tuple[ReconstructionWindow, ...]:
    """Build single or joint windows from activation-sensitivity group names."""
    if groups and windows:
        raise ValueError("Specify either groups or windows, not both.")
    specifications = tuple(windows or groups or ())
    if not specifications:
        raise ValueError("At least one reconstruction group or window is required.")
    semantic_groups = build_activation_sensitivity_groups(model, boundaries)
    group_map = {group.name: group for group in semantic_groups}
    group_order = {group.name: index for index, group in enumerate(semantic_groups)}
    detector = _find_detector(model)
    seen_groups: set[str] = set()
    seen_sites: set[str] = set()
    result: list[ReconstructionWindow] = []
    for specification in specifications:
        names = tuple(part for part in specification.split("+") if part)
        if not names:
            raise ValueError(f"Invalid reconstruction window {specification!r}.")
        if len(set(names)) != len(names):
            raise ValueError(f"Window {specification!r} repeats a group.")
        missing = tuple(name for name in names if name not in group_map)
        if missing:
            raise KeyError(
                f"Unknown reconstruction groups {missing}; available groups: "
                f"{tuple(group_map)}."
            )
        overlap = seen_groups.intersection(names)
        if overlap:
            raise ValueError(
                "Reconstruction windows reuse groups: " f"{tuple(sorted(overlap))}."
            )
        _validate_consecutive_groups(names, group_order)
        selected = tuple(group_map[name] for name in names)
        if any(not group.operation_positions for group in selected):
            raise ValueError(
                f"Window {specification!r} contains a group without operations."
            )
        positions = tuple(
            sorted(
                {
                    position
                    for group in selected
                    for position in group.operation_positions
                }
            )
        )
        sites = tuple(sorted({path for group in selected for path in group.site_paths}))
        site_overlap = seen_sites.intersection(sites)
        if site_overlap:
            raise ValueError(
                f"Reconstruction windows overlap quantization sites: "
                f"{tuple(sorted(site_overlap))}."
            )
        inputs, outputs = _window_boundaries(detector, positions)
        name = "+".join(names)
        result.append(
            ReconstructionWindow(
                name=name,
                group_names=names,
                operation_positions=positions,
                input_tensor_ids=inputs,
                output_tensor_ids=outputs,
                site_paths=sites,
            )
        )
        seen_groups.update(names)
        seen_sites.update(sites)
    result.sort(key=lambda window: min(window.operation_positions))
    return tuple(result)


def _validate_consecutive_groups(
    names: tuple[str, ...],
    group_order: Mapping[str, int],
) -> None:
    """Require joint-window names to follow semantic execution order."""
    indices = tuple(group_order[name] for name in names)
    expected = tuple(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise ValueError(
            "Joint reconstruction windows must contain consecutive groups in "
            f"model execution order; names={names}, indices={indices}."
        )


def build_window_observer_groups(
    model: nn.Module,
    window: ReconstructionWindow,
) -> tuple[AffineObserverGroup, ...]:
    """Tie producer/consumer observer sites by static logical tensor domain."""
    detector = _find_detector(model)
    sites = {site.path: site for site in iter_quantization_sites(model)}
    entries: list[tuple[str, frozenset[int]]] = []
    for path in window.site_paths:
        site = sites.get(path)
        if site is None:
            raise KeyError(f"Window {window.name!r} references unknown site {path!r}.")
        if site.role not in _ACTIVATION_ROLES:
            raise ValueError(f"Window site {path!r} is not an activation site.")
        entries.append((path, frozenset(_site_tensor_domain(site, detector))))

    # A tensor-ID intersection describes graph connectivity, not qparam
    # identity. For example, QuantMaxPool2d owns one shared observer spanning
    # both its input and output tensors. That observer represents a distinct
    # requantization domain from an upstream producer observer, even though the
    # two domains both mention the MaxPool input tensor. Merging by transitive
    # intersection would force different calibrated qparams into one learnable
    # proxy and would also make step zero differ from the entry model state.
    #
    # Tie only observers that report the exact same static tensor-domain set.
    # Producer/consumer observers for one tensor still tie as ``(tensor_id,)``,
    # while a shared operator domain such as ``(input_id, output_id)`` remains
    # independent and preserves the original quantization boundary.
    components: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for path, tensor_ids in entries:
        components[tuple(sorted(tensor_ids))].append(path)

    definitions: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    for tensor_ids, paths in components.items():
        definitions.append((tensor_ids, tuple(sorted(paths))))
    definitions.sort(key=lambda item: (item[0], item[1]))
    groups = tuple(
        AffineObserverGroup(
            name="tensor_" + "_".join(str(value) for value in tensor_ids),
            site_paths=paths,
        )
        for tensor_ids, paths in definitions
    )
    assigned = tuple(sorted(path for group in groups for path in group.site_paths))
    if assigned != tuple(sorted(window.site_paths)):
        raise RuntimeError(
            f"Observer groups do not cover window {window.name!r} exactly once."
        )
    return groups


def collect_reconstruction_cache(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    samples: Sequence[torch.Tensor],
    window: ReconstructionWindow,
    *,
    boundaries: QuantizationBoundaries,
) -> ReconstructionCache:
    """Collect float inputs, quantized-prefix inputs, and all float live-outs."""
    if not samples:
        raise ValueError("Reconstruction cache collection requires samples.")
    reference_detector = _find_detector(reference_model)
    candidate_detector = _find_detector(candidate_model)
    selector = boundaries.selector_for(QuantizationProfile.INTERNAL_FULL)
    cached: list[ReconstructionSample] = []
    with torch.inference_mode(), FakeQuantState(candidate_model) as state:
        state.set_all(False)
        state.set_where(selector, True)
        for sample in samples:
            float_values = execute_detector_values(reference_detector, sample)
            quantized_values = execute_detector_values(candidate_detector, sample)
            float_input = BlockInvocation(
                args=_tensor_values(
                    float_values,
                    window.input_tensor_ids,
                    window=window,
                    context="float live-in",
                )
            )
            quantized_input = BlockInvocation(
                args=_tensor_values(
                    quantized_values,
                    window.input_tensor_ids,
                    window=window,
                    context="quantized live-in",
                )
            )
            targets = _tensor_values(
                float_values,
                window.output_tensor_ids,
                window=window,
                context="float live-out",
            )
            target = targets[0] if len(targets) == 1 else targets
            cached.append(
                ReconstructionSample(
                    float_input=float_input,
                    quantized_input=quantized_input,
                    target=target,
                )
            )
    return ReconstructionCache(cached)


def evaluate_internal_full(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    samples: Sequence[torch.Tensor],
    *,
    boundaries: QuantizationBoundaries,
    output_adapter: OutputAdapter,
) -> dict[str, dict[str, float | int | None]]:
    """Evaluate one candidate under E:internal-full."""
    selector = boundaries.selector_for(QuantizationProfile.INTERNAL_FULL)
    with FakeQuantState(candidate_model) as state:
        state.set_all(False)
        state.set_where(selector, True)
        return evaluate_models(
            reference_model,
            candidate_model,
            samples,
            output_adapter=output_adapter,
        )


def reconstruct_hand_detector_windows(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *,
    train_samples: Sequence[torch.Tensor],
    selection_samples: Sequence[torch.Tensor],
    evaluation_samples: Sequence[torch.Tensor],
    windows: Sequence[ReconstructionWindow],
    config: BlockReconstructionConfig,
    objective: ValidationObjective,
    output_adapter: OutputAdapter,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Reconstruct windows in execution order with validation-aware rollback."""
    boundaries = output_boundaries(candidate_model)
    selector = boundaries.selector_for(QuantizationProfile.INTERNAL_FULL)
    site_count = sum(
        selector(site) for site in iter_quantization_sites(candidate_model)
    )
    baseline_selection = evaluate_internal_full(
        reference_model,
        candidate_model,
        selection_samples,
        boundaries=boundaries,
        output_adapter=output_adapter,
    )
    baseline_evaluation = evaluate_internal_full(
        reference_model,
        candidate_model,
        evaluation_samples,
        boundaries=boundaries,
        output_adapter=output_adapter,
    )
    current_selection = baseline_selection
    current_evaluation = baseline_evaluation
    steps: list[dict[str, object]] = []

    for step, window in enumerate(windows, start=1):
        train_cache = collect_reconstruction_cache(
            reference_model,
            candidate_model,
            train_samples,
            window,
            boundaries=boundaries,
        )
        selection_cache = collect_reconstruction_cache(
            reference_model,
            candidate_model,
            selection_samples,
            window,
            boundaries=boundaries,
        )
        block = DetectorWindow(_find_detector(candidate_model), window)
        observer_groups = build_window_observer_groups(candidate_model, window)
        reconstructor = BlockReconstructor(config)

        def selection_evaluator():
            return evaluate_internal_full(
                reference_model,
                candidate_model,
                selection_samples,
                boundaries=boundaries,
                output_adapter=output_adapter,
            )

        result = reconstructor.reconstruct(
            block_name=window.name,
            observer_model=candidate_model,
            block=block,
            cache=train_cache,
            selection_cache=selection_cache,
            observer_groups=observer_groups,
            selection_evaluator=selection_evaluator,
            selection_objective=objective,
            device=device,
        )
        after_selection = evaluate_internal_full(
            reference_model,
            candidate_model,
            selection_samples,
            boundaries=boundaries,
            output_adapter=output_adapter,
        )
        after_evaluation = evaluate_internal_full(
            reference_model,
            candidate_model,
            evaluation_samples,
            boundaries=boundaries,
            output_adapter=output_adapter,
        )
        step_value = {
            "step": step,
            "window": window.to_dict(),
            "observer_group_count": len(observer_groups),
            "observer_groups": [
                {"name": group.name, "site_paths": list(group.site_paths)}
                for group in observer_groups
            ],
            "reconstruction": result.to_dict(),
            "selection_before": current_selection,
            "selection_after": after_selection,
            "evaluation_before": current_evaluation,
            "evaluation_after": after_evaluation,
        }
        steps.append(step_value)
        current_selection = after_selection
        current_evaluation = after_evaluation

    return {
        "baseline_profile": "E",
        "baseline_site_count": site_count,
        "train_sample_count": len(train_samples),
        "selection_sample_count": len(selection_samples),
        "evaluation_sample_count": len(evaluation_samples),
        "baseline_selection": baseline_selection,
        "baseline_evaluation": baseline_evaluation,
        "steps": steps,
        "final_selection": current_selection,
        "final_evaluation": current_evaluation,
    }


def execute_detector_values(
    detector: HandDetector,
    input_nhwc: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Execute the static detector and return every materialized tensor value."""
    quantized = detector.input_quantizer(input_nhwc)
    values: dict[int, torch.Tensor] = {
        detector.input_tensor: quantized.permute(0, 3, 1, 2)
    }
    for operation, layer in zip(detector.operations, detector.layers):
        _execute_operation(operation, layer, values)
    return values


def _execute_operation(
    operation: Mapping[str, Any],
    layer: nn.Module,
    values: dict[int, torch.Tensor],
) -> None:
    name = str(operation["name"])
    inputs = tuple(int(value) for value in operation["inputs"])
    output = int(operation["outputs"][0])
    config = operation["config"]
    if name in {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "PRELU",
        "MAX_POOL_2D",
        "PAD",
        "RESIZE_BILINEAR",
    }:
        values[output] = layer(values[inputs[0]])
    elif name == "ADD":
        values[output] = values[inputs[0]] + values[inputs[1]]
    elif name == "RESHAPE":
        source = values[inputs[0]]
        if bool(config["nhwc_memory_order"]):
            source = source.permute(0, 2, 3, 1)
        values[output] = source.reshape(tuple(config["shape"]))
    elif name == "CONCATENATION":
        values[output] = layer(tuple(values[index] for index in inputs))
    else:
        raise RuntimeError(f"Unsupported converted operation: {name}")


def _window_boundaries(
    detector: HandDetector,
    positions: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected = frozenset(positions)
    producers = _producer_positions(detector.operations)
    consumers = _consumer_positions(detector.operations)
    input_ids: list[int] = []
    output_ids: list[int] = []
    for position in positions:
        operation = detector.operations[position]
        for tensor_id in _activation_inputs(operation):
            if producers.get(tensor_id) not in selected and tensor_id not in input_ids:
                input_ids.append(tensor_id)
        for output_value in operation["outputs"]:
            tensor_id = int(output_value)
            external_consumer = any(
                consumer not in selected for consumer in consumers.get(tensor_id, ())
            )
            if (
                external_consumer
                or tensor_id in detector.output_tensors
                or not consumers.get(tensor_id)
            ) and tensor_id not in output_ids:
                output_ids.append(tensor_id)
    if not input_ids:
        raise RuntimeError(
            "A reconstruction window must expose at least one live-in tensor."
        )
    if not output_ids:
        last = detector.operations[positions[-1]]
        output_ids.extend(int(value) for value in last["outputs"])
    return tuple(input_ids), tuple(output_ids)


def _tensor_values(
    values: Mapping[int, torch.Tensor],
    tensor_ids: Sequence[int],
    *,
    window: ReconstructionWindow,
    context: str,
) -> tuple[torch.Tensor, ...]:
    """Return materialized tensor values with a diagnostic boundary error."""
    missing = tuple(tensor_id for tensor_id in tensor_ids if tensor_id not in values)
    if missing:
        raise RuntimeError(
            f"Window {window.name!r} requires unavailable {context} tensors "
            f"{missing}. This usually means a constant parameter tensor was "
            "mistaken for a runtime activation boundary."
        )
    return tuple(values[tensor_id] for tensor_id in tensor_ids)


def _activation_inputs(operation: Mapping[str, Any]) -> tuple[int, ...]:
    """Return runtime activation inputs, excluding embedded constant tensors.

    The converted detector specification retains original TFLite input tensor
    IDs, including Conv weights/biases, PReLU slopes, padding constants, and
    reshape shapes. Those constants are already embedded in the PyTorch layer
    or operation config and are not materialized by ``execute_detector_values``.
    """
    name = str(operation["name"])
    inputs = tuple(int(value) for value in operation["inputs"])
    if name == "ADD":
        return inputs[:2]
    if name == "CONCATENATION":
        return inputs
    return inputs[:1]


def _site_tensor_domain(site, detector: HandDetector) -> tuple[int, ...]:
    module_name = getattr(site.module, "fp_name", None) or site.module_path
    if "input_quantizer" in module_name:
        return (detector.input_tensor,)
    match = _LAYER_PATTERN.search(module_name)
    if match is None:
        raise RuntimeError(f"Cannot map site {site.path!r} to a detector layer.")
    position = int(match.group(1))
    operation = detector.operations[position]
    if site.role is SiteRole.ACTIVATION_INPUT:
        inputs = _activation_inputs(operation)
        if operation["name"] == "MAX_POOL_2D":
            return tuple((*inputs, *(int(value) for value in operation["outputs"])))
        return inputs
    if site.role in {SiteRole.ACTIVATION_OUTPUT, SiteRole.ACTIVATION}:
        return tuple(int(value) for value in operation["outputs"])
    raise RuntimeError(f"Unsupported activation site role: {site.role.value!r}.")


def _find_detector(model: nn.Module) -> HandDetector:
    if isinstance(model, HandDetector):
        return model
    if isinstance(model, NHWCInputAdapter):
        return model.detector
    detectors = tuple(
        module for module in model.modules() if isinstance(module, HandDetector)
    )
    if len(detectors) != 1:
        raise ValueError(f"Expected exactly one HandDetector, found {len(detectors)}.")
    return detectors[0]


def _producer_positions(operations: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for position, operation in enumerate(operations):
        for output in operation["outputs"]:
            tensor_id = int(output)
            if tensor_id in result:
                raise RuntimeError(f"Tensor {tensor_id} has multiple producers.")
            result[tensor_id] = position
    return result


def _consumer_positions(
    operations: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[int, ...]]:
    values: dict[int, list[int]] = defaultdict(list)
    for position, operation in enumerate(operations):
        for tensor_id in _activation_inputs(operation):
            values[tensor_id].append(position)
    return {tensor_id: tuple(positions) for tensor_id, positions in values.items()}
