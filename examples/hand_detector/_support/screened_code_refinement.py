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

"""Hand-detector integration for screened single-code refinement."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from tico.quantization.algorithm.adaround import (
    JointAdaRoundObjective,
    ScreenedCodeRefinementConfig,
    ScreenedCodeRefinementRunner,
    ScreenedCodeRoundResult,
)
from tico.quantization.analysis import OutputAdapter
from torch import nn

from examples.hand_detector._support.global_weight_refinement import (
    build_global_conv_weight_definitions,
    copy_metric_outputs,
)
from examples.hand_detector._support.joint_adaround import evaluate_full_quantized
from examples.hand_detector._support.multistart_reconstruction import (
    ReconstructionDataSplit,
)


def run_hand_detector_screened_code_refinement(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *,
    data_split: ReconstructionDataSplit,
    evaluation_samples: Sequence[torch.Tensor],
    config: ScreenedCodeRefinementConfig,
    selection_objective: JointAdaRoundObjective,
    acceptance_objective: JointAdaRoundObjective,
    output_adapter: OutputAdapter,
    requested_groups: Sequence[str] | None = None,
    progress_callback: Callable[[ScreenedCodeRoundResult], None] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Refine Conv codes with finite-difference screened single flips."""
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

    result = ScreenedCodeRefinementRunner(config).refine(
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
        "screened_code_refinement": result.to_dict(),
        "baseline_selection": copy_metric_outputs(result.entry_selection_outputs),
        "baseline_acceptance": copy_metric_outputs(result.entry_acceptance_outputs),
        "baseline_evaluation": copy_metric_outputs(result.entry_evaluation_outputs),
        "final_selection": copy_metric_outputs(result.final_selection_outputs),
        "final_acceptance": copy_metric_outputs(result.final_acceptance_outputs),
        "final_evaluation": copy_metric_outputs(result.final_evaluation_outputs),
    }
