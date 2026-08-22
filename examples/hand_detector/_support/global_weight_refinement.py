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

"""Hand-detector integration for global end-to-end Conv weight refinement."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tico.quantization.algorithm.adaround import (
    GlobalWeightRefinementCheckpoint,
    GlobalWeightRefinementConfig,
    GlobalWeightRefinementRunner,
    JointAdaRoundObjective,
    JointAdaRoundWeightGroup,
)
from tico.quantization.analysis import OutputAdapter
from torch import nn

from examples.hand_detector._support.joint_adaround import evaluate_full_quantized
from examples.hand_detector._support.multistart_reconstruction import (
    ReconstructionDataSplit,
)
from examples.hand_detector._support.reconstruction import _find_detector
from examples.hand_detector._support.weight_precision_sensitivity import (
    build_weight_sensitivity_groups,
)


_CONV_KINDS = frozenset(
    {
        "conv2d_weight",
        "depthwise_conv2d_weight",
    }
)
_KIND_TO_FAMILY = {
    "conv2d_weight": "regular_conv",
    "depthwise_conv2d_weight": "depthwise_conv",
}


@dataclass(frozen=True)
class GlobalConvWeightDefinition:
    """Describe one candidate Conv site and its original FP source weight."""

    group: JointAdaRoundWeightGroup
    semantic_group: str
    operation_position: int
    operation_index: int
    operation_name: str
    parameter_element_count: int
    source_weight: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group.name,
            "site_path": self.group.site_path,
            "family": self.group.family,
            "semantic_group": self.semantic_group,
            "operation_position": self.operation_position,
            "operation_index": self.operation_index,
            "operation_name": self.operation_name,
            "parameter_element_count": self.parameter_element_count,
        }


def build_global_conv_weight_definitions(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *,
    requested_groups: Sequence[str] | None = None,
) -> tuple[GlobalConvWeightDefinition, ...]:
    """Map every selected wrapped Conv site to the original FP32 weight."""
    reference_detector = _find_detector(reference_model)
    site_groups = build_weight_sensitivity_groups(
        candidate_model,
        granularity="site",
    )
    requested = None if requested_groups is None else tuple(requested_groups)
    if requested is not None:
        if not requested:
            raise ValueError("Global refinement --groups must not be empty.")
        if len(set(requested)) != len(requested):
            raise ValueError("Global refinement group names must be unique.")
        available = {group.semantic_group for group in site_groups}
        missing = tuple(name for name in requested if name not in available)
        if missing:
            raise KeyError(
                f"Unknown global refinement groups: {missing}; available groups: "
                f"{tuple(sorted(available))}."
            )
        selected_semantic = frozenset(requested)
    else:
        selected_semantic = None

    definitions: list[GlobalConvWeightDefinition] = []
    for site_group in site_groups:
        if site_group.site_count != 1 or len(site_group.parameter_breakdown) != 1:
            raise RuntimeError(
                "Global refinement expects one parameter site per site group."
            )
        kind = site_group.parameter_breakdown[0].kind
        if kind not in _CONV_KINDS:
            continue
        if (
            selected_semantic is not None
            and site_group.semantic_group not in selected_semantic
        ):
            continue
        if len(site_group.operation_positions) != 1:
            raise RuntimeError(
                f"Conv site {site_group.name!r} does not map to one operation."
            )
        position = int(site_group.operation_positions[0])
        layer = reference_detector.layers[position]
        source_module = getattr(layer, "conv", None)
        if not isinstance(source_module, nn.Conv2d):
            raise TypeError(
                f"Reference layer {position} for {site_group.name!r} is not a "
                "ConvNode with an nn.Conv2d weight."
            )
        site_path = site_group.site_paths[0]
        family = _KIND_TO_FAMILY[kind]
        definition = GlobalConvWeightDefinition(
            group=JointAdaRoundWeightGroup(
                name=site_group.name,
                site_path=site_path,
                family=family,
            ),
            semantic_group=site_group.semantic_group,
            operation_position=position,
            operation_index=int(site_group.operation_indices[0]),
            operation_name=str(site_group.operation_names[0]),
            parameter_element_count=site_group.parameter_element_count,
            source_weight=source_module.weight.detach().clone(),
        )
        definitions.append(definition)

    if not definitions:
        raise ValueError("Global refinement selected no Conv2d weight sites.")
    definitions.sort(
        key=lambda value: (
            value.operation_position,
            value.group.site_path,
        )
    )
    paths = tuple(value.group.site_path for value in definitions)
    names = tuple(value.group.name for value in definitions)
    if len(set(paths)) != len(paths) or len(set(names)) != len(names):
        raise RuntimeError("Global refinement Conv definitions are not unique.")
    if requested is not None:
        covered = {value.semantic_group for value in definitions}
        empty = tuple(name for name in requested if name not in covered)
        if empty:
            raise ValueError(f"Requested groups contain no Conv weights: {empty}.")
    return tuple(definitions)


def run_hand_detector_global_weight_refinement(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *,
    data_split: ReconstructionDataSplit,
    evaluation_samples: Sequence[torch.Tensor],
    config: GlobalWeightRefinementConfig,
    selection_objective: JointAdaRoundObjective,
    acceptance_objective: JointAdaRoundObjective,
    output_adapter: OutputAdapter,
    requested_groups: Sequence[str] | None = None,
    progress_callback: (
        Callable[[GlobalWeightRefinementCheckpoint], None] | None
    ) = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Refine selected Conv weights together using final teacher outputs."""
    definitions = build_global_conv_weight_definitions(
        reference_model,
        candidate_model,
        requested_groups=requested_groups,
    )
    groups = tuple(value.group for value in definitions)
    source_weights = {
        value.group.site_path: value.source_weight for value in definitions
    }

    def selection_evaluator():
        return evaluate_full_quantized(
            reference_model,
            candidate_model,
            data_split.selection,
            output_adapter=output_adapter,
        )

    def acceptance_evaluator():
        return evaluate_full_quantized(
            reference_model,
            candidate_model,
            data_split.acceptance,
            output_adapter=output_adapter,
        )

    def evaluation_evaluator():
        return evaluate_full_quantized(
            reference_model,
            candidate_model,
            evaluation_samples,
            output_adapter=output_adapter,
        )

    result = GlobalWeightRefinementRunner(config).refine(
        reference_model=reference_model,
        candidate_model=candidate_model,
        training_samples=data_split.train,
        weight_groups=groups,
        source_weights=source_weights,
        output_adapter=output_adapter,
        selection_evaluator=selection_evaluator,
        selection_objective=selection_objective,
        acceptance_evaluator=acceptance_evaluator,
        acceptance_objective=acceptance_objective,
        evaluation_evaluator=evaluation_evaluator,
        progress_callback=progress_callback,
        device=device,
    )
    family_counts: dict[str, int] = {}
    family_parameters: dict[str, int] = {}
    for value in definitions:
        family = value.group.family
        family_counts[family] = family_counts.get(family, 0) + 1
        family_parameters[family] = (
            family_parameters.get(family, 0) + value.parameter_element_count
        )
    return {
        "profile": "P2:W8/A16/reg-I16/cls-U8",
        "data_split": data_split.to_dict(),
        "weight_group_count": len(definitions),
        "weight_parameter_element_count": sum(
            value.parameter_element_count for value in definitions
        ),
        "family_site_counts": family_counts,
        "family_parameter_element_counts": family_parameters,
        "weight_definitions": [value.to_dict() for value in definitions],
        "global_refinement": result.to_dict(),
        "baseline_selection": copy_metric_outputs(result.entry_selection_outputs),
        "baseline_acceptance": copy_metric_outputs(result.entry_acceptance_outputs),
        "baseline_evaluation": copy_metric_outputs(result.entry_evaluation_outputs),
        "scale_only_selection": copy_metric_outputs(
            result.scale_only_selection_outputs
        ),
        "scale_only_acceptance": copy_metric_outputs(
            result.scale_only_acceptance_outputs
        ),
        "scale_only_evaluation": copy_metric_outputs(
            result.scale_only_evaluation_outputs
        ),
        "selected_selection": copy_metric_outputs(result.selected_outputs),
        "selected_acceptance": copy_metric_outputs(result.acceptance_outputs),
        "selected_evaluation": copy_metric_outputs(result.selected_evaluation_outputs),
        "final_evaluation": copy_metric_outputs(result.final_evaluation_outputs),
    }


def copy_metric_outputs(
    outputs: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    return {name: dict(metrics) for name, metrics in outputs.items()}
