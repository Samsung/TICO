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

"""Finite-difference screened single-coordinate W8 code refinement."""

from __future__ import annotations

import math

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import nn

from tico.quantization.algorithm.adaround.discrete_refinement import (
    _gradient_sample_indices,
    DiscreteCodeCandidate,
    DiscreteCodeFinalChange,
    DiscreteCodeGradientStatistics,
    DiscreteCodeTransitionSummary,
    DiscreteCodeWeightSet,
    DiscreteCodeWeightStatistics,
)
from tico.quantization.algorithm.adaround.global_refinement import (
    _module_device,
    _output_loss,
    _RequiresGradState,
    _TeacherOutputCache,
    _validate_batch_one,
)
from tico.quantization.algorithm.adaround.joint import JointAdaRoundWeightGroup
from tico.quantization.algorithm.adaround.joint_runner import JointAdaRoundObjective
from tico.quantization.algorithm.block_reconstruction.selection import (
    copy_outputs,
    metric_value,
    OutputMetrics,
)
from tico.quantization.analysis import OutputAdapter
from tico.quantization.wrapq.control import FakeQuantState


MetricsEvaluator = Callable[[], OutputMetrics]
ProgressCallback = Callable[["ScreenedCodeRoundResult"], None]


@dataclass(frozen=True)
class ScreenedCodeCandidateEvaluation:
    """Record one single-code finite-difference screening evaluation."""

    candidate: DiscreteCodeCandidate
    shortlist_sources: tuple[str, ...]
    screening_rank: int
    screening_primary_score: float
    screening_auxiliary_score: float
    screening_total_score: float
    screening_improvement: float
    selection_attempted: bool = False
    selection_outputs: OutputMetrics | None = None
    selection_score: float | None = None
    selection_improvement: float | None = None
    selection_eligible: bool | None = None
    selection_reason: str | None = None
    selection_rank: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "shortlist_sources": list(self.shortlist_sources),
            "screening_rank": self.screening_rank,
            "screening_primary_score": self.screening_primary_score,
            "screening_auxiliary_score": self.screening_auxiliary_score,
            "screening_total_score": self.screening_total_score,
            "screening_improvement": self.screening_improvement,
            "selection_attempted": self.selection_attempted,
            "selection_outputs": (
                copy_outputs(self.selection_outputs)
                if self.selection_outputs is not None
                else None
            ),
            "selection_score": self.selection_score,
            "selection_improvement": self.selection_improvement,
            "selection_eligible": self.selection_eligible,
            "selection_reason": self.selection_reason,
            "selection_rank": self.selection_rank,
        }


@dataclass(frozen=True)
class ScreenedCodeRoundResult:
    """Record one screened single-coordinate refinement transaction."""

    round_index: int
    entry_selection_outputs: OutputMetrics
    entry_acceptance_outputs: OutputMetrics
    entry_evaluation_outputs: OutputMetrics
    gradient_statistics: DiscreteCodeGradientStatistics
    screening_sample_indices: tuple[int, ...]
    entry_screening_primary_score: float
    entry_screening_auxiliary_score: float
    entry_screening_total_score: float
    candidate_evaluations: tuple[ScreenedCodeCandidateEvaluation, ...]
    selected_candidate: DiscreteCodeCandidate | None
    selected_selection_outputs: OutputMetrics
    selected_acceptance_outputs: OutputMetrics
    selected_evaluation_outputs: OutputMetrics
    accepted: bool
    acceptance_improvement: float | None
    acceptance_reason: str
    transition_summary: DiscreteCodeTransitionSummary
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "entry_selection_outputs": copy_outputs(self.entry_selection_outputs),
            "entry_acceptance_outputs": copy_outputs(self.entry_acceptance_outputs),
            "entry_evaluation_outputs": copy_outputs(self.entry_evaluation_outputs),
            "gradient_statistics": self.gradient_statistics.to_dict(),
            "screening_sample_count": len(self.screening_sample_indices),
            "screening_sample_indices": list(self.screening_sample_indices),
            "entry_screening_primary_score": (self.entry_screening_primary_score),
            "entry_screening_auxiliary_score": (self.entry_screening_auxiliary_score),
            "entry_screening_total_score": self.entry_screening_total_score,
            "candidate_evaluations": [
                value.to_dict() for value in self.candidate_evaluations
            ],
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "selected_selection_outputs": copy_outputs(self.selected_selection_outputs),
            "selected_acceptance_outputs": copy_outputs(
                self.selected_acceptance_outputs
            ),
            "selected_evaluation_outputs": copy_outputs(
                self.selected_evaluation_outputs
            ),
            "accepted": self.accepted,
            "acceptance_improvement": self.acceptance_improvement,
            "acceptance_reason": self.acceptance_reason,
            "transition_summary": self.transition_summary.to_dict(),
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class ScreenedCodeRefinementConfig:
    """Configure finite-difference screened single-coordinate refinement."""

    max_rounds: int = 16
    gradient_sample_count: int = 0
    gradient_seed: int = 20260901
    screening_sample_count: int = 16
    screening_seed: int = 20260911
    global_shortlist_count: int = 32
    per_site_shortlist_count: int = 1
    per_channel_shortlist_count: int = 1
    maximum_channel_candidates: int = 64
    maximum_shortlist_count: int = 256
    selection_candidate_count: int = 8
    primary_output: str = "regressors"
    auxiliary_output: str = "classifiers"
    auxiliary_gradient_weight: float = 0.0
    screening_auxiliary_weight: float = 0.0
    training_loss: str = "raw_mae"
    loss_epsilon: float = 1.0e-8
    minimum_predicted_improvement: float = 0.0
    minimum_screening_improvement: float = 0.0
    target_primary_score: float | None = 0.1
    initialization_metric_tolerance: float = 1.0e-4
    initialization_metric_relative_tolerance: float = 1.0e-3

    def validate(self) -> None:
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive.")
        for name, value in (
            ("gradient_sample_count", self.gradient_sample_count),
            ("screening_sample_count", self.screening_sample_count),
            ("global_shortlist_count", self.global_shortlist_count),
            ("per_site_shortlist_count", self.per_site_shortlist_count),
            ("per_channel_shortlist_count", self.per_channel_shortlist_count),
            ("maximum_channel_candidates", self.maximum_channel_candidates),
            ("maximum_shortlist_count", self.maximum_shortlist_count),
            ("selection_candidate_count", self.selection_candidate_count),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.screening_sample_count == 0:
            raise ValueError("screening_sample_count must be positive.")
        if self.maximum_shortlist_count == 0:
            raise ValueError("maximum_shortlist_count must be positive.")
        if self.selection_candidate_count == 0:
            raise ValueError("selection_candidate_count must be positive.")
        channel_source_enabled = (
            self.per_channel_shortlist_count > 0 and self.maximum_channel_candidates > 0
        )
        if (
            self.global_shortlist_count == 0
            and self.per_site_shortlist_count == 0
            and not channel_source_enabled
        ):
            raise ValueError("At least one shortlist source must be enabled.")
        if not isinstance(self.gradient_seed, int):
            raise TypeError("gradient_seed must be an integer.")
        if not isinstance(self.screening_seed, int):
            raise TypeError("screening_seed must be an integer.")
        if not self.primary_output or not self.auxiliary_output:
            raise ValueError("Output names must be non-empty.")
        if self.primary_output == self.auxiliary_output:
            raise ValueError("Primary and auxiliary outputs must differ.")
        if self.training_loss not in {"raw_mae", "normalized_l1"}:
            raise ValueError("training_loss must be 'raw_mae' or 'normalized_l1'.")
        for name, float_value in (
            ("auxiliary_gradient_weight", self.auxiliary_gradient_weight),
            ("screening_auxiliary_weight", self.screening_auxiliary_weight),
            ("minimum_predicted_improvement", self.minimum_predicted_improvement),
            ("minimum_screening_improvement", self.minimum_screening_improvement),
            ("initialization_metric_tolerance", self.initialization_metric_tolerance),
            (
                "initialization_metric_relative_tolerance",
                self.initialization_metric_relative_tolerance,
            ),
        ):
            if not math.isfinite(float_value) or float_value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not math.isfinite(self.loss_epsilon) or self.loss_epsilon <= 0.0:
            raise ValueError("loss_epsilon must be finite and positive.")
        if self.target_primary_score is not None and (
            not math.isfinite(self.target_primary_score)
            or self.target_primary_score <= 0.0
        ):
            raise ValueError("target_primary_score must be positive or None.")


@dataclass(frozen=True)
class ScreenedCodeRefinementResult:
    """Summarize screened single-coordinate hard-code refinement."""

    requested_rounds: int
    completed_rounds: int
    accepted_rounds: int
    stop_reason: str
    weight_groups: tuple[str, ...]
    weight_families: tuple[str, ...]
    entry_selection_outputs: OutputMetrics
    entry_acceptance_outputs: OutputMetrics
    entry_evaluation_outputs: OutputMetrics
    final_selection_outputs: OutputMetrics
    final_acceptance_outputs: OutputMetrics
    final_evaluation_outputs: OutputMetrics
    rounds: tuple[ScreenedCodeRoundResult, ...]
    weight_statistics: tuple[DiscreteCodeWeightStatistics, ...]
    final_code_changes: tuple[DiscreteCodeFinalChange, ...]

    @property
    def accepted(self) -> bool:
        return self.accepted_rounds > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_rounds": self.requested_rounds,
            "completed_rounds": self.completed_rounds,
            "accepted_rounds": self.accepted_rounds,
            "accepted": self.accepted,
            "stop_reason": self.stop_reason,
            "weight_group_count": len(self.weight_groups),
            "weight_groups": list(self.weight_groups),
            "weight_families": list(self.weight_families),
            "entry_selection_outputs": copy_outputs(self.entry_selection_outputs),
            "entry_acceptance_outputs": copy_outputs(self.entry_acceptance_outputs),
            "entry_evaluation_outputs": copy_outputs(self.entry_evaluation_outputs),
            "final_selection_outputs": copy_outputs(self.final_selection_outputs),
            "final_acceptance_outputs": copy_outputs(self.final_acceptance_outputs),
            "final_evaluation_outputs": copy_outputs(self.final_evaluation_outputs),
            "rounds": [value.to_dict() for value in self.rounds],
            "weight_statistics": [value.to_dict() for value in self.weight_statistics],
            "final_code_change_count": len(self.final_code_changes),
            "final_code_changes": [
                value.to_dict() for value in self.final_code_changes
            ],
        }


class ScreenedCodeRefinementRunner:
    """Screen gradient-diversified single-code candidates transactionally."""

    def __init__(
        self,
        config: ScreenedCodeRefinementConfig | None = None,
    ) -> None:
        self.config = config or ScreenedCodeRefinementConfig()
        self.config.validate()

    def refine(
        self,
        *,
        reference_model: nn.Module,
        candidate_model: nn.Module,
        training_samples: Sequence[torch.Tensor],
        weight_groups: Sequence[JointAdaRoundWeightGroup],
        source_weights: Mapping[str, torch.Tensor],
        output_adapter: OutputAdapter,
        selection_evaluator: MetricsEvaluator,
        selection_objective: JointAdaRoundObjective,
        acceptance_evaluator: MetricsEvaluator,
        acceptance_objective: JointAdaRoundObjective,
        evaluation_evaluator: MetricsEvaluator,
        progress_callback: ProgressCallback | None = None,
        device: torch.device | str | None = None,
    ) -> ScreenedCodeRefinementResult:
        groups = tuple(weight_groups)
        if not groups:
            raise ValueError("Screened refinement requires Conv weight groups.")
        if not training_samples:
            raise ValueError("Screened refinement requires training samples.")
        optimization_device = torch.device(device or _module_device(candidate_model))
        _validate_batch_one(training_samples)
        teacher_cache = _TeacherOutputCache(
            reference_model,
            training_samples,
            output_adapter=output_adapter,
            device=optimization_device,
        )
        entry_selection = copy_outputs(selection_evaluator())
        entry_acceptance = copy_outputs(acceptance_evaluator())
        entry_evaluation = copy_outputs(evaluation_evaluator())
        current_selection: OutputMetrics = entry_selection
        current_acceptance = entry_acceptance
        current_evaluation = entry_evaluation
        round_results: list[ScreenedCodeRoundResult] = []
        accepted_rounds = 0
        stop_reason = "maximum rounds reached"

        with (
            _RequiresGradState(candidate_model),
            FakeQuantState(candidate_model) as fake_quant_state,
        ):
            fake_quant_state.set_all(True)
            weights = DiscreteCodeWeightSet(
                candidate_model,
                groups,
                source_weights,
            )
            try:
                initialized_selection = copy_outputs(selection_evaluator())
                self._validate_initialization(
                    entry_selection,
                    initialized_selection,
                )
                current_selection = initialized_selection

                for round_index in range(1, self.config.max_rounds + 1):
                    round_entry_selection = current_selection
                    round_entry_acceptance = current_acceptance
                    round_entry_evaluation = current_evaluation
                    round_state = weights.state_snapshot()
                    gradient_statistics, shortlist = self._collect_shortlist(
                        round_index,
                        teacher_cache,
                        candidate_model,
                        weights,
                        output_adapter,
                        optimization_device,
                    )
                    if not shortlist:
                        stop_reason = "no predicted-improving shortlist candidates"
                        result = self._stopped_round(
                            round_index,
                            round_entry_selection,
                            round_entry_acceptance,
                            round_entry_evaluation,
                            gradient_statistics,
                            weights,
                            stop_reason,
                        )
                        round_results.append(result)
                        if progress_callback is not None:
                            progress_callback(result)
                        break

                    screening_indices = _gradient_sample_indices(
                        len(teacher_cache),
                        self.config.screening_sample_count,
                        seed=self.config.screening_seed + round_index - 1,
                    )
                    entry_screening = self._evaluate_teacher_loss(
                        teacher_cache,
                        candidate_model,
                        output_adapter,
                        screening_indices,
                        optimization_device,
                    )
                    evaluations = self._screen_candidates(
                        shortlist,
                        round_state,
                        weights,
                        teacher_cache,
                        candidate_model,
                        output_adapter,
                        screening_indices,
                        entry_screening,
                        optimization_device,
                    )
                    screening_successes = [
                        index
                        for index, value in enumerate(evaluations)
                        if value.screening_improvement
                        > self.config.minimum_screening_improvement
                    ]
                    if not screening_successes:
                        weights.load_state_snapshot(round_state)
                        stop_reason = (
                            "no finite-difference candidate improved screening loss"
                        )
                        result = self._stopped_round(
                            round_index,
                            round_entry_selection,
                            round_entry_acceptance,
                            round_entry_evaluation,
                            gradient_statistics,
                            weights,
                            stop_reason,
                            screening_sample_indices=screening_indices,
                            entry_screening=entry_screening,
                            candidate_evaluations=tuple(evaluations),
                        )
                        round_results.append(result)
                        if progress_callback is not None:
                            progress_callback(result)
                        break

                    screening_successes.sort(
                        key=lambda index: evaluations[index].screening_total_score
                    )
                    selection_indices = screening_successes[
                        : self.config.selection_candidate_count
                    ]
                    entry_selection_score = selection_objective.score(
                        round_entry_selection
                    )
                    for evaluation_index in selection_indices:
                        weights.load_state_snapshot(round_state)
                        evaluation = evaluations[evaluation_index]
                        weights.apply_candidates((evaluation.candidate,))
                        outputs = copy_outputs(selection_evaluator())
                        eligible, reason = selection_objective.better(
                            outputs,
                            round_entry_selection,
                            round_entry_selection,
                        )
                        score = selection_objective.score(outputs)
                        evaluations[evaluation_index] = replace(
                            evaluation,
                            selection_attempted=True,
                            selection_outputs=outputs,
                            selection_score=score,
                            selection_improvement=(entry_selection_score - score),
                            selection_eligible=eligible,
                            selection_reason=reason,
                        )
                    attempted = sorted(
                        selection_indices,
                        key=lambda index: (
                            evaluations[index].selection_score,
                            evaluations[index].screening_rank,
                        ),
                    )
                    for selection_rank, evaluation_index in enumerate(
                        attempted,
                        start=1,
                    ):
                        evaluations[evaluation_index] = replace(
                            evaluations[evaluation_index],
                            selection_rank=selection_rank,
                        )
                    weights.load_state_snapshot(round_state)
                    selection_successes = [
                        index
                        for index in selection_indices
                        if evaluations[index].selection_eligible
                    ]
                    if not selection_successes:
                        stop_reason = "no screened candidate improved selection metrics"
                        result = self._stopped_round(
                            round_index,
                            round_entry_selection,
                            round_entry_acceptance,
                            round_entry_evaluation,
                            gradient_statistics,
                            weights,
                            stop_reason,
                            screening_sample_indices=screening_indices,
                            entry_screening=entry_screening,
                            candidate_evaluations=tuple(evaluations),
                        )
                        round_results.append(result)
                        if progress_callback is not None:
                            progress_callback(result)
                        break

                    def _selection_score(index: int) -> float:
                        score = evaluations[index].selection_score
                        assert score is not None
                        return score

                    winner_index = min(
                        selection_successes,
                        key=_selection_score,
                    )
                    winner = evaluations[winner_index]
                    assert winner.selection_outputs is not None
                    weights.load_state_snapshot(round_state)
                    weights.apply_candidates((winner.candidate,))
                    acceptance = copy_outputs(acceptance_evaluator())
                    accepted, reason = acceptance_objective.accepted(
                        acceptance,
                        round_entry_acceptance,
                    )
                    entry_acceptance_score = acceptance_objective.score(
                        round_entry_acceptance
                    )
                    acceptance_improvement = (
                        entry_acceptance_score - acceptance_objective.score(acceptance)
                    )
                    if not accepted:
                        weights.load_state_snapshot(round_state)
                        stop_reason = "selection winner failed acceptance"
                        result = ScreenedCodeRoundResult(
                            round_index=round_index,
                            entry_selection_outputs=round_entry_selection,
                            entry_acceptance_outputs=round_entry_acceptance,
                            entry_evaluation_outputs=round_entry_evaluation,
                            gradient_statistics=gradient_statistics,
                            screening_sample_indices=screening_indices,
                            entry_screening_primary_score=entry_screening[0],
                            entry_screening_auxiliary_score=entry_screening[1],
                            entry_screening_total_score=entry_screening[2],
                            candidate_evaluations=tuple(evaluations),
                            selected_candidate=winner.candidate,
                            selected_selection_outputs=winner.selection_outputs,
                            selected_acceptance_outputs=acceptance,
                            selected_evaluation_outputs=round_entry_evaluation,
                            accepted=False,
                            acceptance_improvement=acceptance_improvement,
                            acceptance_reason=reason,
                            transition_summary=weights.transition_summary(()),
                            stop_reason=stop_reason,
                        )
                        round_results.append(result)
                        if progress_callback is not None:
                            progress_callback(result)
                        break

                    evaluation_outputs = copy_outputs(evaluation_evaluator())
                    current_selection = winner.selection_outputs
                    current_acceptance = acceptance
                    current_evaluation = evaluation_outputs
                    accepted_rounds += 1
                    transition = weights.transition_summary((winner.candidate,))
                    result = ScreenedCodeRoundResult(
                        round_index=round_index,
                        entry_selection_outputs=round_entry_selection,
                        entry_acceptance_outputs=round_entry_acceptance,
                        entry_evaluation_outputs=round_entry_evaluation,
                        gradient_statistics=gradient_statistics,
                        screening_sample_indices=screening_indices,
                        entry_screening_primary_score=entry_screening[0],
                        entry_screening_auxiliary_score=entry_screening[1],
                        entry_screening_total_score=entry_screening[2],
                        candidate_evaluations=tuple(evaluations),
                        selected_candidate=winner.candidate,
                        selected_selection_outputs=current_selection,
                        selected_acceptance_outputs=current_acceptance,
                        selected_evaluation_outputs=current_evaluation,
                        accepted=True,
                        acceptance_improvement=acceptance_improvement,
                        acceptance_reason=reason,
                        transition_summary=transition,
                    )
                    round_results.append(result)
                    if progress_callback is not None:
                        progress_callback(result)
                    if (
                        self.config.target_primary_score is not None
                        and acceptance_objective.score(current_acceptance)
                        < self.config.target_primary_score
                    ):
                        stop_reason = "target primary score reached on acceptance set"
                        break
                else:
                    stop_reason = "maximum rounds reached"

                final_changes = weights.final_code_changes()
                selected_statistics = weights.statistics()
                if accepted_rounds > 0:
                    weight_statistics = weights.finalize()
                else:
                    weights.restore()
                    weight_statistics = selected_statistics
            except Exception:
                weights.restore()
                raise

        final_evaluation = copy_outputs(evaluation_evaluator())
        return ScreenedCodeRefinementResult(
            requested_rounds=self.config.max_rounds,
            completed_rounds=len(round_results),
            accepted_rounds=accepted_rounds,
            stop_reason=stop_reason,
            weight_groups=tuple(group.name for group in groups),
            weight_families=tuple(group.family for group in groups),
            entry_selection_outputs=entry_selection,
            entry_acceptance_outputs=entry_acceptance,
            entry_evaluation_outputs=entry_evaluation,
            final_selection_outputs=current_selection,
            final_acceptance_outputs=current_acceptance,
            final_evaluation_outputs=final_evaluation,
            rounds=tuple(round_results),
            weight_statistics=weight_statistics,
            final_code_changes=final_changes,
        )

    def _collect_shortlist(
        self,
        round_index: int,
        teacher_cache: _TeacherOutputCache,
        candidate_model: nn.Module,
        weights: DiscreteCodeWeightSet,
        output_adapter: OutputAdapter,
        device: torch.device,
    ) -> tuple[
        DiscreteCodeGradientStatistics,
        tuple[tuple[DiscreteCodeCandidate, tuple[str, ...]], ...],
    ]:
        sample_indices, primary, auxiliary, total = self._collect_gradient(
            round_index,
            teacher_cache,
            candidate_model,
            weights,
            output_adapter,
            device,
        )
        global_ranked, reachable, improving, histogram = weights.rank_candidates(
            maximum_count=self.config.global_shortlist_count,
            minimum_predicted_improvement=(self.config.minimum_predicted_improvement),
        )
        sources: dict[tuple[str, int], set[str]] = defaultdict(set)
        scores: dict[tuple[str, int], float] = {}
        binding_indices = {
            binding.group.site_path: index
            for index, binding in enumerate(weights.bindings)
        }
        for candidate in global_ranked:
            key = (candidate.site_path, candidate.flat_index)
            sources[key].add("global")
            scores[key] = candidate.predicted_loss_delta

        channel_winners: list[tuple[float, int, int]] = []
        for binding_index, binding in enumerate(weights.bindings):
            proxy = binding.proxy
            gradient = proxy.effective_weight.grad
            if gradient is None:
                raise RuntimeError(
                    "No hard-weight gradient was collected for "
                    f"{binding.group.site_path!r}."
                )
            alternative_codes, valid = proxy.alternative_codes()
            alternative_weight = proxy.alternative_weight(alternative_codes)
            score = gradient.detach() * (
                alternative_weight - proxy.effective_weight.detach()
            )
            improving_mask = (
                valid
                & torch.isfinite(score)
                & (score < -self.config.minimum_predicted_improvement)
            )
            if self.config.per_site_shortlist_count > 0:
                for flat_index in _topk_masked_indices(
                    score.flatten(),
                    improving_mask.flatten(),
                    self.config.per_site_shortlist_count,
                ):
                    key = (binding.group.site_path, flat_index)
                    sources[key].add("site")
                    scores[key] = float(score.flatten()[flat_index].item())
            if self.config.per_channel_shortlist_count > 0:
                channel_scores = score.reshape(score.shape[0], -1)
                channel_mask = improving_mask.reshape(score.shape[0], -1)
                for value, flat_index in _channel_topk_entries(
                    channel_scores,
                    channel_mask,
                    self.config.per_channel_shortlist_count,
                ):
                    channel_winners.append((value, binding_index, flat_index))
        channel_winners.sort(key=lambda value: value[0])
        for channel_score, binding_index, flat_index in channel_winners[
            : self.config.maximum_channel_candidates
        ]:
            binding = weights.bindings[binding_index]
            key = (binding.group.site_path, flat_index)
            sources[key].add("channel")
            scores[key] = channel_score

        ordered = sorted(scores, key=lambda key: (scores[key], key[0], key[1]))
        ordered = ordered[: self.config.maximum_shortlist_count]
        shortlist: list[tuple[DiscreteCodeCandidate, tuple[str, ...]]] = []
        for rank, key in enumerate(ordered, start=1):
            site_path, flat_index = key
            binding_index = binding_indices[site_path]
            candidate = weights._candidate_record(
                rank,
                binding_index,
                flat_index,
                scores[key],
            )
            shortlist.append((candidate, tuple(sorted(sources[key]))))
        statistics = DiscreteCodeGradientStatistics(
            sample_indices=sample_indices,
            primary_loss=primary,
            auxiliary_loss=auxiliary,
            total_loss=total,
            gradient_histogram=histogram,
            reachable_candidate_count=reachable,
            predicted_improving_candidate_count=improving,
            recorded_candidate_count=len(shortlist),
        )
        return statistics, tuple(shortlist)

    def _collect_gradient(
        self,
        round_index: int,
        teacher_cache: _TeacherOutputCache,
        candidate_model: nn.Module,
        weights: DiscreteCodeWeightSet,
        output_adapter: OutputAdapter,
        device: torch.device,
    ) -> tuple[tuple[int, ...], float, float, float]:
        sample_indices = _gradient_sample_indices(
            len(teacher_cache),
            self.config.gradient_sample_count,
            seed=self.config.gradient_seed + round_index - 1,
        )
        weights.zero_grad()
        primary_value = 0.0
        auxiliary_value = 0.0
        count = len(sample_indices)
        for index in sample_indices:
            sample, target = teacher_cache.get(index, device=device)
            outputs = output_adapter(candidate_model(sample))
            primary = _output_loss(
                outputs[self.config.primary_output],
                target[self.config.primary_output],
                kind=self.config.training_loss,
                epsilon=self.config.loss_epsilon,
            )
            auxiliary = _output_loss(
                outputs[self.config.auxiliary_output],
                target[self.config.auxiliary_output],
                kind=self.config.training_loss,
                epsilon=self.config.loss_epsilon,
            )
            loss = (primary + self.config.auxiliary_gradient_weight * auxiliary) / count
            loss.backward()
            primary_value += float(primary.detach().cpu().item())
            auxiliary_value += float(auxiliary.detach().cpu().item())
        primary_value /= count
        auxiliary_value /= count
        total_value = (
            primary_value + self.config.auxiliary_gradient_weight * auxiliary_value
        )
        return sample_indices, primary_value, auxiliary_value, total_value

    def _screen_candidates(
        self,
        shortlist: Sequence[tuple[DiscreteCodeCandidate, tuple[str, ...]]],
        round_state: Mapping[str, torch.Tensor],
        weights: DiscreteCodeWeightSet,
        teacher_cache: _TeacherOutputCache,
        candidate_model: nn.Module,
        output_adapter: OutputAdapter,
        sample_indices: Sequence[int],
        entry_screening: tuple[float, float, float],
        device: torch.device,
    ) -> list[ScreenedCodeCandidateEvaluation]:
        raw: list[
            tuple[
                DiscreteCodeCandidate,
                tuple[str, ...],
                float,
                float,
                float,
                float,
            ]
        ] = []
        for candidate, sources in shortlist:
            weights.load_state_snapshot(round_state)
            weights.apply_candidates((candidate,))
            primary, auxiliary, total = self._evaluate_teacher_loss(
                teacher_cache,
                candidate_model,
                output_adapter,
                sample_indices,
                device,
            )
            raw.append(
                (
                    candidate,
                    sources,
                    primary,
                    auxiliary,
                    total,
                    entry_screening[2] - total,
                )
            )
        weights.load_state_snapshot(round_state)
        ordered = sorted(
            raw,
            key=lambda value: (
                value[4],
                value[0].predicted_loss_delta,
                value[0].site_path,
                value[0].flat_index,
            ),
        )
        ranks = {
            (value[0].site_path, value[0].flat_index): rank
            for rank, value in enumerate(ordered, start=1)
        }
        return [
            ScreenedCodeCandidateEvaluation(
                candidate=candidate,
                shortlist_sources=sources,
                screening_rank=ranks[(candidate.site_path, candidate.flat_index)],
                screening_primary_score=primary,
                screening_auxiliary_score=auxiliary,
                screening_total_score=total,
                screening_improvement=improvement,
            )
            for candidate, sources, primary, auxiliary, total, improvement in raw
        ]

    def _evaluate_teacher_loss(
        self,
        teacher_cache: _TeacherOutputCache,
        candidate_model: nn.Module,
        output_adapter: OutputAdapter,
        sample_indices: Sequence[int],
        device: torch.device,
    ) -> tuple[float, float, float]:
        primary_value = 0.0
        auxiliary_value = 0.0
        with torch.no_grad():
            for index in sample_indices:
                sample, target = teacher_cache.get(index, device=device)
                outputs = output_adapter(candidate_model(sample))
                primary = _output_loss(
                    outputs[self.config.primary_output],
                    target[self.config.primary_output],
                    kind=self.config.training_loss,
                    epsilon=self.config.loss_epsilon,
                )
                auxiliary = _output_loss(
                    outputs[self.config.auxiliary_output],
                    target[self.config.auxiliary_output],
                    kind=self.config.training_loss,
                    epsilon=self.config.loss_epsilon,
                )
                primary_value += float(primary.cpu().item())
                auxiliary_value += float(auxiliary.cpu().item())
        count = len(sample_indices)
        primary_value /= count
        auxiliary_value /= count
        total_value = (
            primary_value + self.config.screening_auxiliary_weight * auxiliary_value
        )
        return primary_value, auxiliary_value, total_value

    def _validate_initialization(
        self,
        entry: OutputMetrics,
        initialized: OutputMetrics,
    ) -> None:
        for output_name in (
            self.config.primary_output,
            self.config.auxiliary_output,
        ):
            left = metric_value(entry, output_name, "mae")
            right = metric_value(initialized, output_name, "mae")
            difference = abs(left - right)
            allowed = (
                self.config.initialization_metric_tolerance
                + self.config.initialization_metric_relative_tolerance
                * max(abs(left), abs(right))
            )
            if difference > allowed:
                raise RuntimeError(
                    "Screened refinement did not reproduce the loaded checkpoint "
                    f"for {output_name}.mae: {left:.9e} != {right:.9e}; "
                    f"difference={difference:.3e}, allowed={allowed:.3e}."
                )

    @staticmethod
    def _stopped_round(
        round_index: int,
        selection: OutputMetrics,
        acceptance: OutputMetrics,
        evaluation: OutputMetrics,
        gradient_statistics: DiscreteCodeGradientStatistics,
        weights: DiscreteCodeWeightSet,
        reason: str,
        *,
        screening_sample_indices: tuple[int, ...] = (),
        entry_screening: tuple[float, float, float] = (0.0, 0.0, 0.0),
        candidate_evaluations: tuple[ScreenedCodeCandidateEvaluation, ...] = (),
    ) -> ScreenedCodeRoundResult:
        return ScreenedCodeRoundResult(
            round_index=round_index,
            entry_selection_outputs=selection,
            entry_acceptance_outputs=acceptance,
            entry_evaluation_outputs=evaluation,
            gradient_statistics=gradient_statistics,
            screening_sample_indices=screening_sample_indices,
            entry_screening_primary_score=entry_screening[0],
            entry_screening_auxiliary_score=entry_screening[1],
            entry_screening_total_score=entry_screening[2],
            candidate_evaluations=candidate_evaluations,
            selected_candidate=None,
            selected_selection_outputs=selection,
            selected_acceptance_outputs=acceptance,
            selected_evaluation_outputs=evaluation,
            accepted=False,
            acceptance_improvement=None,
            acceptance_reason=reason,
            transition_summary=weights.transition_summary(()),
            stop_reason=reason,
        )


def _topk_masked_indices(
    values: torch.Tensor,
    mask: torch.Tensor,
    count: int,
) -> tuple[int, ...]:
    if count <= 0:
        return ()
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return ()
    selected_count = min(count, int(indices.numel()))
    selected_values = values[indices]
    _, positions = torch.topk(
        selected_values,
        k=selected_count,
        largest=False,
        sorted=True,
    )
    selected = indices[positions].detach().cpu().tolist()
    return tuple(int(value) for value in selected)


def _channel_topk_entries(
    values: torch.Tensor,
    mask: torch.Tensor,
    count: int,
) -> tuple[tuple[float, int], ...]:
    if count <= 0 or values.numel() == 0:
        return ()
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("Channel top-K expects matching rank-two tensors.")
    selected_count = min(count, values.shape[1])
    masked = torch.where(
        mask,
        values,
        torch.full_like(values, float("inf")),
    )
    selected_values, selected_indices = torch.topk(
        masked,
        k=selected_count,
        dim=1,
        largest=False,
        sorted=True,
    )
    finite = torch.isfinite(selected_values)
    channels, positions = torch.nonzero(finite, as_tuple=True)
    if channels.numel() == 0:
        return ()
    local_indices = selected_indices[channels, positions]
    flat_indices = channels * values.shape[1] + local_indices
    result_values = selected_values[channels, positions].detach().cpu().tolist()
    result_indices = flat_indices.detach().cpu().tolist()
    return tuple(
        (float(value), int(flat_index))
        for value, flat_index in zip(result_values, result_indices)
    )
