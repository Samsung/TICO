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

"""Three-way data splitting and per-window QDrop multi-start reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tico.quantization.algorithm.block_reconstruction import (
    BlockReconstructionConfig,
    QDropCandidate,
    QDropMultiStartReconstructor,
    ValidationObjective,
)
from tico.quantization.analysis import OutputAdapter, QuantizationProfile
from tico.quantization.wrapq.control import iter_quantization_sites
from torch import nn

from examples.hand_detector._support.analysis import output_boundaries
from examples.hand_detector._support.reconstruction import (
    _find_detector,
    build_window_observer_groups,
    collect_reconstruction_cache,
    DetectorWindow,
    evaluate_internal_full,
    ReconstructionWindow,
)


@dataclass(frozen=True)
class ReconstructionDataSplit:
    """Hold disjoint optimization, checkpoint-selection, and acceptance subsets."""

    train: tuple[torch.Tensor, ...]
    selection: tuple[torch.Tensor, ...]
    acceptance: tuple[torch.Tensor, ...]

    def to_dict(self) -> dict[str, int]:
        """Return split cardinalities for reports."""
        return {
            "train_count": len(self.train),
            "selection_count": len(self.selection),
            "acceptance_count": len(self.acceptance),
        }


def split_reconstruction_samples_three_way(
    calibration_samples: Sequence[torch.Tensor],
    selection_count: int,
    acceptance_count: int,
    *,
    seed: int = 20260803,
) -> ReconstructionDataSplit:
    """Create deterministic disjoint train/selection/acceptance subsets."""
    if selection_count <= 0:
        raise ValueError("selection_count must be positive.")
    if acceptance_count <= 0:
        raise ValueError("acceptance_count must be positive.")
    held_out_count = selection_count + acceptance_count
    if held_out_count >= len(calibration_samples):
        raise ValueError(
            "selection_count + acceptance_count must be smaller than the "
            "calibration sample count."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(
        len(calibration_samples),
        generator=generator,
    ).tolist()
    selection_indices = frozenset(permutation[:selection_count])
    acceptance_indices = frozenset(permutation[selection_count:held_out_count])
    train = tuple(
        sample
        for index, sample in enumerate(calibration_samples)
        if index not in selection_indices and index not in acceptance_indices
    )
    selection = tuple(
        sample
        for index, sample in enumerate(calibration_samples)
        if index in selection_indices
    )
    acceptance = tuple(
        sample
        for index, sample in enumerate(calibration_samples)
        if index in acceptance_indices
    )
    return ReconstructionDataSplit(train, selection, acceptance)


def reconstruct_hand_detector_windows_multistart(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *,
    data_split: ReconstructionDataSplit,
    evaluation_samples: Sequence[torch.Tensor],
    windows: Sequence[ReconstructionWindow],
    config: BlockReconstructionConfig,
    candidates: Sequence[QDropCandidate],
    selection_objective: ValidationObjective,
    acceptance_objective: ValidationObjective,
    output_adapter: OutputAdapter,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Compete QDrop candidates per window and commit one acceptance winner."""
    boundaries = output_boundaries(candidate_model)
    selector = boundaries.selector_for(QuantizationProfile.INTERNAL_FULL)
    site_count = sum(
        selector(site) for site in iter_quantization_sites(candidate_model)
    )
    baseline_selection = evaluate_internal_full(
        reference_model,
        candidate_model,
        data_split.selection,
        boundaries=boundaries,
        output_adapter=output_adapter,
    )
    baseline_acceptance = evaluate_internal_full(
        reference_model,
        candidate_model,
        data_split.acceptance,
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
    steps: list[dict[str, object]] = []

    for step, window in enumerate(windows, start=1):
        train_cache = collect_reconstruction_cache(
            reference_model,
            candidate_model,
            data_split.train,
            window,
            boundaries=boundaries,
        )
        selection_cache = collect_reconstruction_cache(
            reference_model,
            candidate_model,
            data_split.selection,
            window,
            boundaries=boundaries,
        )
        block = DetectorWindow(_find_detector(candidate_model), window)
        observer_groups = build_window_observer_groups(candidate_model, window)

        def selection_evaluator():
            return evaluate_internal_full(
                reference_model,
                candidate_model,
                data_split.selection,
                boundaries=boundaries,
                output_adapter=output_adapter,
            )

        def acceptance_evaluator():
            return evaluate_internal_full(
                reference_model,
                candidate_model,
                data_split.acceptance,
                boundaries=boundaries,
                output_adapter=output_adapter,
            )

        result = QDropMultiStartReconstructor(
            config,
            candidates,
            acceptance_objective=acceptance_objective,
        ).reconstruct(
            block_name=window.name,
            observer_model=candidate_model,
            block=block,
            cache=train_cache,
            selection_cache=selection_cache,
            observer_groups=observer_groups,
            selection_evaluator=selection_evaluator,
            selection_objective=selection_objective,
            acceptance_evaluator=acceptance_evaluator,
            device=device,
        )
        after_selection = selection_evaluator()
        after_acceptance = acceptance_evaluator()
        after_evaluation = evaluate_internal_full(
            reference_model,
            candidate_model,
            evaluation_samples,
            boundaries=boundaries,
            output_adapter=output_adapter,
        )
        winner_record = next(
            (
                candidate
                for candidate in result.candidates
                if candidate.selected_as_winner
            ),
            None,
        )
        if winner_record is None:
            compatibility_reconstruction = result.candidates[0].reconstruction.to_dict()
            compatibility_reconstruction["accepted"] = False
            compatibility_reconstruction["best_step"] = 0
            compatibility_reconstruction["acceptance_reason"] = result.acceptance_reason
        else:
            compatibility_reconstruction = winner_record.reconstruction.to_dict()
            compatibility_reconstruction["accepted"] = result.accepted
            compatibility_reconstruction["acceptance_reason"] = result.acceptance_reason
        steps.append(
            {
                "step": step,
                "window": window.to_dict(),
                "observer_group_count": len(observer_groups),
                "observer_groups": [
                    {
                        "name": group.name,
                        "site_paths": list(group.site_paths),
                    }
                    for group in observer_groups
                ],
                "reconstruction": compatibility_reconstruction,
                "multistart": result.to_dict(),
                "selection_after": after_selection,
                "acceptance_after": after_acceptance,
                "evaluation_after": after_evaluation,
            }
        )

    return {
        "profile": "E:internal-full",
        "enabled_site_count": site_count,
        "data_split": data_split.to_dict(),
        "baseline_selection": baseline_selection,
        "baseline_acceptance": baseline_acceptance,
        "baseline_evaluation": baseline_evaluation,
        "steps": steps,
        "final_selection": (
            steps[-1]["selection_after"] if steps else baseline_selection
        ),
        "final_acceptance": (
            steps[-1]["acceptance_after"] if steps else baseline_acceptance
        ),
        "final_evaluation": (
            steps[-1]["evaluation_after"] if steps else baseline_evaluation
        ),
    }
