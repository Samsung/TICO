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

"""Gradient-ranked transactional refinement of fixed-scale affine W8 codes."""

from __future__ import annotations

import math

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import nn

from tico.quantization.algorithm.adaround.global_refinement import (
    _absolute_tensor_histogram,
    _broadcast_qparams,
    _effective_hard_codes,
    _module_device,
    _observer_attribute_names,
    _output_loss,
    _replace_observer_attributes,
    _RequiresGradState,
    _TeacherOutputCache,
    _validate_batch_one,
    _validate_groups,
    GlobalRefinementTensorHistogram,
)
from tico.quantization.algorithm.adaround.joint import JointAdaRoundWeightGroup
from tico.quantization.algorithm.adaround.joint_runner import JointAdaRoundObjective
from tico.quantization.algorithm.block_reconstruction.selection import (
    copy_outputs,
    metric_value,
    OutputMetrics,
)
from tico.quantization.analysis import OutputAdapter
from tico.quantization.wrapq.control import (
    FakeQuantState,
    iter_quantization_sites,
    QuantizationSite,
    SiteRole,
)
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase


MetricsEvaluator = Callable[[], OutputMetrics]
ProgressCallback = Callable[["DiscreteCodeRoundResult"], None]

_WEIGHT_GRADIENT_HISTOGRAM_EDGES = (
    1.0e-12,
    1.0e-10,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
)


@dataclass(frozen=True)
class DiscreteCodeCandidate:
    """Describe one exact floor/ceil hard-code alternative."""

    rank: int
    group_name: str
    site_path: str
    family: str
    flat_index: int
    tensor_index: tuple[int, ...]
    old_code: int
    new_code: int
    source_weight: float
    current_weight: float
    alternative_weight: float
    scale: float
    zero_point: int
    gradient: float
    predicted_loss_delta: float
    transition_kind: str

    @property
    def predicted_improvement(self) -> float:
        return -self.predicted_loss_delta

    @property
    def direction(self) -> int:
        return self.new_code - self.old_code

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "group_name": self.group_name,
            "site_path": self.site_path,
            "family": self.family,
            "flat_index": self.flat_index,
            "tensor_index": list(self.tensor_index),
            "old_code": self.old_code,
            "new_code": self.new_code,
            "direction": self.direction,
            "source_weight": self.source_weight,
            "current_weight": self.current_weight,
            "alternative_weight": self.alternative_weight,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "gradient": self.gradient,
            "predicted_loss_delta": self.predicted_loss_delta,
            "predicted_improvement": self.predicted_improvement,
            "transition_kind": self.transition_kind,
        }


@dataclass(frozen=True)
class DiscreteCodeFinalChange:
    """Describe one final code difference relative to the command entry."""

    group_name: str
    site_path: str
    family: str
    flat_index: int
    tensor_index: tuple[int, ...]
    entry_code: int
    final_code: int
    scale: float
    zero_point: int
    entry_weight: float
    final_weight: float

    def to_dict(self) -> dict[str, object]:
        return {
            "group_name": self.group_name,
            "site_path": self.site_path,
            "family": self.family,
            "flat_index": self.flat_index,
            "tensor_index": list(self.tensor_index),
            "entry_code": self.entry_code,
            "final_code": self.final_code,
            "direction": self.final_code - self.entry_code,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "entry_weight": self.entry_weight,
            "final_weight": self.final_weight,
        }


@dataclass(frozen=True)
class DiscreteCodeGradientStatistics:
    """Summarize one round's exact hard-weight gradient aggregation."""

    sample_indices: tuple[int, ...]
    primary_loss: float
    auxiliary_loss: float
    total_loss: float
    gradient_histogram: GlobalRefinementTensorHistogram
    reachable_candidate_count: int
    predicted_improving_candidate_count: int
    recorded_candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": len(self.sample_indices),
            "sample_indices": list(self.sample_indices),
            "primary_loss": self.primary_loss,
            "auxiliary_loss": self.auxiliary_loss,
            "total_loss": self.total_loss,
            "gradient_histogram": self.gradient_histogram.to_dict(),
            "reachable_candidate_count": self.reachable_candidate_count,
            "predicted_improving_candidate_count": (
                self.predicted_improving_candidate_count
            ),
            "recorded_candidate_count": self.recorded_candidate_count,
        }


@dataclass(frozen=True)
class DiscreteCodeProposalEvaluation:
    """Record one nested top-K hard proposal and its validation outcome."""

    requested_size: int
    applied_size: int
    predicted_loss_delta_sum: float
    selection_outputs: OutputMetrics
    selection_score: float
    selection_improvement: float
    selection_eligible: bool
    selection_reason: str
    acceptance_attempted: bool = False
    acceptance_outputs: OutputMetrics | None = None
    acceptance_improvement: float | None = None
    acceptance_eligible: bool | None = None
    acceptance_reason: str | None = None

    @property
    def predicted_improvement(self) -> float:
        return -self.predicted_loss_delta_sum

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_size": self.requested_size,
            "applied_size": self.applied_size,
            "predicted_loss_delta_sum": self.predicted_loss_delta_sum,
            "predicted_improvement": self.predicted_improvement,
            "selection_outputs": copy_outputs(self.selection_outputs),
            "selection_score": self.selection_score,
            "selection_improvement": self.selection_improvement,
            "selection_eligible": self.selection_eligible,
            "selection_reason": self.selection_reason,
            "acceptance_attempted": self.acceptance_attempted,
            "acceptance_outputs": (
                copy_outputs(self.acceptance_outputs)
                if self.acceptance_outputs is not None
                else None
            ),
            "acceptance_improvement": self.acceptance_improvement,
            "acceptance_eligible": self.acceptance_eligible,
            "acceptance_reason": self.acceptance_reason,
        }


@dataclass(frozen=True)
class DiscreteCodeTransitionSummary:
    """Summarize changes relative to the command-entry checkpoint."""

    newly_changed_count: int
    reverted_count: int
    retained_count: int
    net_changed_count: int
    site_change_counts: Mapping[str, Mapping[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "newly_changed_count": self.newly_changed_count,
            "reverted_count": self.reverted_count,
            "retained_count": self.retained_count,
            "net_changed_count": self.net_changed_count,
            "site_change_counts": {
                path: dict(values) for path, values in self.site_change_counts.items()
            },
        }


@dataclass(frozen=True)
class DiscreteCodeRoundResult:
    """Record one transactional gradient-ranked proposal round."""

    round_index: int
    entry_selection_outputs: OutputMetrics
    entry_acceptance_outputs: OutputMetrics
    entry_evaluation_outputs: OutputMetrics
    gradient_statistics: DiscreteCodeGradientStatistics
    ranked_candidates: tuple[DiscreteCodeCandidate, ...]
    proposal_evaluations: tuple[DiscreteCodeProposalEvaluation, ...]
    selected_size: int | None
    selected_candidates: tuple[DiscreteCodeCandidate, ...]
    selected_selection_outputs: OutputMetrics
    selected_acceptance_outputs: OutputMetrics
    selected_evaluation_outputs: OutputMetrics
    accepted: bool
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
            "ranked_candidates": [value.to_dict() for value in self.ranked_candidates],
            "proposal_evaluations": [
                value.to_dict() for value in self.proposal_evaluations
            ],
            "selected_size": self.selected_size,
            "selected_candidates": [
                value.to_dict() for value in self.selected_candidates
            ],
            "selected_selection_outputs": copy_outputs(self.selected_selection_outputs),
            "selected_acceptance_outputs": copy_outputs(
                self.selected_acceptance_outputs
            ),
            "selected_evaluation_outputs": copy_outputs(
                self.selected_evaluation_outputs
            ),
            "accepted": self.accepted,
            "acceptance_reason": self.acceptance_reason,
            "transition_summary": self.transition_summary.to_dict(),
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class DiscreteCodeWeightStatistics:
    """Summarize one fixed-scale Conv code tensor after refinement."""

    group_name: str
    site_path: str
    family: str
    element_count: int
    reachable_decision_count: int
    changed_from_entry_count: int

    @property
    def changed_from_entry_ratio(self) -> float:
        return self.changed_from_entry_count / max(self.element_count, 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_name": self.group_name,
            "site_path": self.site_path,
            "family": self.family,
            "element_count": self.element_count,
            "reachable_decision_count": self.reachable_decision_count,
            "changed_from_entry_count": self.changed_from_entry_count,
            "changed_from_entry_ratio": self.changed_from_entry_ratio,
        }


@dataclass(frozen=True)
class DiscreteCodeRefinementConfig:
    """Configure explicit fixed-scale hard-code coordinate refinement."""

    max_rounds: int = 4
    proposal_sizes: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)
    gradient_sample_count: int = 0
    gradient_seed: int = 20260901
    primary_output: str = "regressors"
    auxiliary_output: str = "classifiers"
    auxiliary_gradient_weight: float = 0.0
    training_loss: str = "raw_mae"
    loss_epsilon: float = 1.0e-8
    minimum_predicted_improvement: float = 0.0
    target_primary_score: float | None = 0.1
    initialization_metric_tolerance: float = 1.0e-4
    initialization_metric_relative_tolerance: float = 1.0e-3

    def validate(self) -> None:
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be nonnegative.")
        if not self.proposal_sizes:
            raise ValueError("proposal_sizes must not be empty.")
        invalid_sizes = (
            not isinstance(value, int) or value <= 0 for value in self.proposal_sizes
        )
        if any(invalid_sizes):
            raise ValueError("Every proposal size must be a positive integer.")
        if len(set(self.proposal_sizes)) != len(self.proposal_sizes):
            raise ValueError("proposal_sizes must be unique.")
        if self.gradient_sample_count < 0:
            raise ValueError("gradient_sample_count must be nonnegative.")
        if not isinstance(self.gradient_seed, int):
            raise TypeError("gradient_seed must be an integer.")
        if not self.primary_output or not self.auxiliary_output:
            raise ValueError("Output names must be non-empty.")
        if self.primary_output == self.auxiliary_output:
            raise ValueError("Primary and auxiliary outputs must differ.")
        if self.training_loss not in {"raw_mae", "normalized_l1"}:
            raise ValueError("training_loss must be 'raw_mae' or 'normalized_l1'.")
        for name, value in (
            ("auxiliary_gradient_weight", self.auxiliary_gradient_weight),
            ("minimum_predicted_improvement", self.minimum_predicted_improvement),
            ("initialization_metric_tolerance", self.initialization_metric_tolerance),
            (
                "initialization_metric_relative_tolerance",
                self.initialization_metric_relative_tolerance,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not math.isfinite(self.loss_epsilon) or self.loss_epsilon <= 0.0:
            raise ValueError("loss_epsilon must be finite and positive.")
        if self.target_primary_score is not None and (
            not math.isfinite(self.target_primary_score)
            or self.target_primary_score <= 0.0
        ):
            raise ValueError("target_primary_score must be positive or None.")


@dataclass(frozen=True)
class DiscreteCodeRefinementResult:
    """Summarize transactional explicit code refinement."""

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
    rounds: tuple[DiscreteCodeRoundResult, ...]
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


class FixedScaleCodeObserver(ObserverBase):
    """Expose one exact fixed-scale code tensor as a differentiable weight."""

    _fixed_scale: torch.Tensor
    _fixed_zero_point: torch.Tensor
    _reference_weight: torch.Tensor
    _entry_codes: torch.Tensor
    _current_codes: torch.Tensor
    _entry_effective_weight: torch.Tensor
    _floor_codes: torch.Tensor
    _ceil_codes: torch.Tensor
    effective_weight: nn.Parameter

    def __init__(
        self,
        original: AffineObserverBase,
        reference_weight: torch.Tensor,
        checkpoint_effective_weight: torch.Tensor,
    ) -> None:
        super().__init__(
            name=original.name,
            dtype=original.dtype,
            qscheme=original.qscheme,
            channel_axis=original.channel_axis,
        )
        if original.channel_axis != 0:
            raise ValueError("Discrete Conv refinement expects channel_axis=0.")
        self.enabled = bool(original.enabled)
        self.fake_quant_enabled = bool(original.fake_quant_enabled)
        scale, zero_point = original.compute_qparams()
        source = reference_weight.detach().to(
            device=checkpoint_effective_weight.device,
            dtype=checkpoint_effective_weight.dtype,
        )
        effective = checkpoint_effective_weight.detach().to(
            device=source.device,
            dtype=source.dtype,
        )
        if source.shape != effective.shape:
            raise ValueError("Reference and checkpoint weight shapes differ.")
        entry_codes = _effective_hard_codes(
            effective,
            scale,
            zero_point,
            channel_axis=original.channel_axis,
            qmin=original.dtype.qmin,
            qmax=original.dtype.qmax,
        ).to(dtype=torch.int64)
        scale_broadcast, zero_point_broadcast = _broadcast_qparams(
            source,
            scale,
            zero_point,
            channel_axis=original.channel_axis,
        )
        normalized = source / scale_broadcast + zero_point_broadcast
        floor = torch.floor(normalized).clamp(
            original.dtype.qmin,
            original.dtype.qmax,
        )
        ceil = (torch.floor(normalized) + 1.0).clamp(
            original.dtype.qmin,
            original.dtype.qmax,
        )
        floor = floor.to(dtype=torch.int64)
        ceil = ceil.to(dtype=torch.int64)
        reachable = (entry_codes == floor) | (entry_codes == ceil)
        if not bool(reachable.all()):
            count = int((~reachable).sum().cpu().item())
            raise ValueError(
                "Checkpoint codes are not reachable by fixed-scale floor/ceil "
                f"decisions; unreachable elements={count}."
            )
        self.register_buffer("_fixed_scale", scale.detach().clone())
        self.register_buffer("_fixed_zero_point", zero_point.detach().clone())
        self.register_buffer("_reference_weight", source.clone())
        self.register_buffer("_entry_codes", entry_codes.clone())
        self.register_buffer("_current_codes", entry_codes.clone())
        self.register_buffer("_entry_effective_weight", effective.clone())
        self.register_buffer("_floor_codes", floor.clone())
        self.register_buffer("_ceil_codes", ceil.clone())
        self.effective_weight = nn.Parameter(effective.clone())

    def reset(self) -> None:
        return None

    def _update_stats(self, x: torch.Tensor) -> None:
        del x

    def compute_qparams(self):
        return self._fixed_scale, self._fixed_zero_point

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return x
        if x.shape != self.effective_weight.shape:
            raise ValueError(
                "Discrete refinement weight shape changed: "
                f"{tuple(x.shape)} != {tuple(self.effective_weight.shape)}."
            )
        return self.effective_weight

    def entry_codes(self) -> torch.Tensor:
        return self._entry_codes.detach().clone()

    def current_codes(self) -> torch.Tensor:
        return self._current_codes.detach().clone()

    def reachable_decision_mask(self) -> torch.Tensor:
        return self._floor_codes != self._ceil_codes

    def alternative_codes(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = self._current_codes
        valid = self.reachable_decision_mask()
        valid = valid & ((current == self._floor_codes) | (current == self._ceil_codes))
        alternative = torch.where(
            current == self._floor_codes,
            self._ceil_codes,
            self._floor_codes,
        )
        return alternative, valid

    def alternative_weight(self, alternative_codes: torch.Tensor) -> torch.Tensor:
        return self._weight_from_codes(alternative_codes)

    def set_codes(self, codes: torch.Tensor) -> None:
        values = codes.detach().to(
            device=self._current_codes.device,
            dtype=self._current_codes.dtype,
        )
        if values.shape != self._current_codes.shape:
            raise ValueError("Discrete code tensor shape changed.")
        if not bool(((values >= self.dtype.qmin) & (values <= self.dtype.qmax)).all()):
            raise ValueError("Discrete codes exceed dtype bounds.")
        reachable = (values == self._floor_codes) | (values == self._ceil_codes)
        if not bool(reachable.all()):
            count = int((~reachable).sum().cpu().item())
            raise ValueError(
                "Discrete code state contains unreachable floor/ceil values; "
                f"count={count}."
            )
        weight = self._weight_from_codes(values)
        with torch.no_grad():
            self._current_codes.copy_(values)
            self.effective_weight.copy_(weight)

    def _weight_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        scale, zero_point = _broadcast_qparams(
            self._reference_weight,
            self._fixed_scale,
            self._fixed_zero_point,
            channel_axis=self.channel_axis,
        )
        dequantized = (
            codes.to(dtype=self._reference_weight.dtype) - zero_point
        ) * scale
        return torch.where(
            codes == self._entry_codes,
            self._entry_effective_weight,
            dequantized,
        )


@dataclass
class _DiscreteBinding:
    group: JointAdaRoundWeightGroup
    owner: nn.Module
    attribute_names: tuple[str, ...]
    weight_module: nn.Conv2d
    original_observer: AffineObserverBase
    entry_weight: torch.Tensor
    entry_scale: torch.Tensor
    entry_zero_point: torch.Tensor
    entry_qparams_locked: bool
    entry_enabled: bool
    entry_fake_quant_enabled: bool
    proxy: FixedScaleCodeObserver


class DiscreteCodeWeightSet:
    """Install fixed-scale exact-code proxies and manage them transactionally."""

    def __init__(
        self,
        model: nn.Module,
        groups: Sequence[JointAdaRoundWeightGroup],
        source_weights: Mapping[str, torch.Tensor],
    ) -> None:
        definitions = tuple(groups)
        _validate_groups(definitions, source_weights)
        sites = {site.path: site for site in iter_quantization_sites(model)}
        bindings: list[_DiscreteBinding] = []
        try:
            for group in definitions:
                site = sites.get(group.site_path)
                if site is None:
                    raise KeyError(
                        f"Unknown discrete-refinement site {group.site_path!r}."
                    )
                binding = _build_binding(
                    group,
                    site,
                    source_weights[group.site_path],
                )
                _replace_observer_attributes(
                    binding.owner,
                    binding.attribute_names,
                    expected=binding.original_observer,
                    replacement=binding.proxy,
                    site_path=group.site_path,
                )
                bindings.append(binding)
        except Exception:
            for binding in reversed(bindings):
                _replace_observer_attributes(
                    binding.owner,
                    binding.attribute_names,
                    expected=binding.proxy,
                    replacement=binding.original_observer,
                    site_path=binding.group.site_path,
                )
            raise
        self.bindings = tuple(bindings)
        self._by_path = {binding.group.site_path: binding for binding in self.bindings}
        self._closed = False

    def gradient_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(binding.proxy.effective_weight for binding in self.bindings)

    def zero_grad(self) -> None:
        for parameter in self.gradient_parameters():
            parameter.grad = None

    def state_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            binding.group.name: binding.proxy.current_codes().cpu()
            for binding in self.bindings
        }

    def load_state_snapshot(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = tuple(binding.group.name for binding in self.bindings)
        if tuple(state) != expected:
            raise ValueError(
                f"Discrete state keys differ: {tuple(state)} != {expected}."
            )
        for binding in self.bindings:
            value = state[binding.group.name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("Every discrete state value must be a Tensor.")
            binding.proxy.set_codes(value)

    def rank_candidates(
        self,
        *,
        maximum_count: int,
        minimum_predicted_improvement: float,
    ) -> tuple[
        tuple[DiscreteCodeCandidate, ...],
        int,
        int,
        GlobalRefinementTensorHistogram,
    ]:
        score_parts: list[torch.Tensor] = []
        binding_parts: list[torch.Tensor] = []
        index_parts: list[torch.Tensor] = []
        reachable_count = 0
        improving_count = 0
        gradients: list[torch.Tensor] = []
        for binding_index, binding in enumerate(self.bindings):
            proxy = binding.proxy
            gradient = proxy.effective_weight.grad
            if gradient is None:
                raise RuntimeError(
                    "No hard-weight gradient was collected for "
                    f"{binding.group.site_path!r}."
                )
            gradients.append(gradient)
            alternative_codes, valid = proxy.alternative_codes()
            alternative_weight = proxy.alternative_weight(alternative_codes)
            delta_weight = alternative_weight - proxy.effective_weight.detach()
            score = gradient.detach() * delta_weight
            finite = torch.isfinite(score)
            reachable_count += int(valid.sum().cpu().item())
            improving = valid & finite & (score < -minimum_predicted_improvement)
            indices = torch.nonzero(improving.flatten(), as_tuple=False).flatten()
            improving_count += int(indices.numel())
            if indices.numel() == 0:
                continue
            flat_score = score.flatten()[indices]
            score_parts.append(flat_score)
            binding_parts.append(
                torch.full_like(indices, binding_index, dtype=torch.int64)
            )
            index_parts.append(indices.to(dtype=torch.int64))

        histogram = _absolute_tensor_histogram(
            gradients,
            edges=_WEIGHT_GRADIENT_HISTOGRAM_EDGES,
        )
        if not score_parts or maximum_count <= 0:
            return (), reachable_count, improving_count, histogram
        scores = torch.cat(score_parts)
        bindings = torch.cat(binding_parts)
        indices = torch.cat(index_parts)
        count = min(maximum_count, int(scores.numel()))
        selected_scores, selected_positions = torch.topk(
            scores,
            k=count,
            largest=False,
            sorted=True,
        )
        selected_bindings = bindings[selected_positions]
        selected_indices = indices[selected_positions]
        records: list[DiscreteCodeCandidate] = []
        for rank, (score, binding_index, flat_index) in enumerate(
            zip(
                selected_scores.detach().cpu().tolist(),
                selected_bindings.detach().cpu().tolist(),
                selected_indices.detach().cpu().tolist(),
            ),
            start=1,
        ):
            records.append(
                self._candidate_record(
                    rank,
                    int(binding_index),
                    int(flat_index),
                    float(score),
                )
            )
        return tuple(records), reachable_count, improving_count, histogram

    def apply_candidates(
        self,
        candidates: Sequence[DiscreteCodeCandidate],
    ) -> None:
        grouped: dict[str, list[DiscreteCodeCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.site_path].append(candidate)
        for site_path, values in grouped.items():
            binding = self._by_path.get(site_path)
            if binding is None:
                raise KeyError(f"Unknown discrete candidate site {site_path!r}.")
            codes = binding.proxy.current_codes()
            flat = codes.flatten()
            for candidate in values:
                if int(flat[candidate.flat_index].item()) != candidate.old_code:
                    raise RuntimeError(
                        "Discrete candidate entry code changed before proposal "
                        f"application at {site_path}[{candidate.flat_index}]."
                    )
                flat[candidate.flat_index] = candidate.new_code
            binding.proxy.set_codes(codes)

    def transition_summary(
        self,
        selected_candidates: Sequence[DiscreteCodeCandidate],
    ) -> DiscreteCodeTransitionSummary:
        kinds = Counter(value.transition_kind for value in selected_candidates)
        current_changed = 0
        site_counts: dict[str, dict[str, int]] = {}
        selected_by_site: dict[str, Counter[str]] = defaultdict(Counter)
        for value in selected_candidates:
            selected_by_site[value.site_path][value.transition_kind] += 1
        for binding in self.bindings:
            changed = int(
                (binding.proxy.current_codes() != binding.proxy.entry_codes())
                .sum()
                .cpu()
                .item()
            )
            current_changed += changed
            counts = selected_by_site.get(binding.group.site_path, Counter())
            site_counts[binding.group.site_path] = {
                "new": int(counts.get("new", 0)),
                "reverted": int(counts.get("reverted", 0)),
                "retained": max(
                    changed - int(counts.get("new", 0)),
                    0,
                ),
                "net_changed": changed,
            }
        return DiscreteCodeTransitionSummary(
            newly_changed_count=int(kinds.get("new", 0)),
            reverted_count=int(kinds.get("reverted", 0)),
            retained_count=max(
                current_changed - int(kinds.get("new", 0)),
                0,
            ),
            net_changed_count=current_changed,
            site_change_counts=site_counts,
        )

    def statistics(self) -> tuple[DiscreteCodeWeightStatistics, ...]:
        values: list[DiscreteCodeWeightStatistics] = []
        for binding in self.bindings:
            proxy = binding.proxy
            values.append(
                DiscreteCodeWeightStatistics(
                    group_name=binding.group.name,
                    site_path=binding.group.site_path,
                    family=binding.group.family,
                    element_count=proxy.effective_weight.numel(),
                    reachable_decision_count=int(
                        proxy.reachable_decision_mask().sum().cpu().item()
                    ),
                    changed_from_entry_count=int(
                        (proxy.current_codes() != proxy.entry_codes())
                        .sum()
                        .cpu()
                        .item()
                    ),
                )
            )
        return tuple(values)

    def final_code_changes(self) -> tuple[DiscreteCodeFinalChange, ...]:
        changes: list[DiscreteCodeFinalChange] = []
        for binding in self.bindings:
            proxy = binding.proxy
            entry = proxy.entry_codes().flatten()
            final = proxy.current_codes().flatten()
            indices = torch.nonzero(entry != final, as_tuple=False).flatten()
            for flat_index in indices.detach().cpu().tolist():
                index = _unravel_index(int(flat_index), proxy.effective_weight.shape)
                channel = index[0]
                scale = float(proxy._fixed_scale[channel].detach().cpu().item())
                zero_point = int(proxy._fixed_zero_point[channel].detach().cpu().item())
                entry_weight = float(
                    proxy._entry_effective_weight.flatten()[flat_index]
                    .detach()
                    .cpu()
                    .item()
                )
                final_weight = float(
                    proxy.effective_weight.flatten()[flat_index].detach().cpu().item()
                )
                changes.append(
                    DiscreteCodeFinalChange(
                        group_name=binding.group.name,
                        site_path=binding.group.site_path,
                        family=binding.group.family,
                        flat_index=int(flat_index),
                        tensor_index=index,
                        entry_code=int(entry[flat_index].cpu().item()),
                        final_code=int(final[flat_index].cpu().item()),
                        scale=scale,
                        zero_point=zero_point,
                        entry_weight=entry_weight,
                        final_weight=final_weight,
                    )
                )
        changes.sort(key=lambda value: (value.site_path, value.flat_index))
        return tuple(changes)

    def finalize(self) -> tuple[DiscreteCodeWeightStatistics, ...]:
        if self._closed:
            raise RuntimeError("Discrete refinement weight set is already closed.")
        statistics = self.statistics()
        try:
            with torch.no_grad():
                for binding in self.bindings:
                    binding.weight_module.weight.copy_(
                        binding.proxy.effective_weight.detach()
                    )
            for binding in self.bindings:
                device = binding.original_observer.min_val.device
                binding.original_observer.load_qparams(
                    binding.entry_scale.to(device=device),
                    binding.entry_zero_point.to(device=device),
                    lock=True,
                )
                binding.original_observer.enabled = binding.entry_enabled
                binding.original_observer.fake_quant_enabled = (
                    binding.entry_fake_quant_enabled
                )
            self._restore_observers()
        except Exception:
            self._restore_entry_state()
            raise
        self._closed = True
        return statistics

    def restore(self) -> None:
        if self._closed:
            return
        self._restore_entry_state()
        self._closed = True

    def _candidate_record(
        self,
        rank: int,
        binding_index: int,
        flat_index: int,
        score: float,
    ) -> DiscreteCodeCandidate:
        binding = self.bindings[binding_index]
        proxy = binding.proxy
        current_codes = proxy.current_codes().flatten()
        alternative_codes, _ = proxy.alternative_codes()
        alternative_codes = alternative_codes.flatten()
        gradient = proxy.effective_weight.grad
        assert gradient is not None
        index = _unravel_index(flat_index, proxy.effective_weight.shape)
        channel = index[0]
        old_code = int(current_codes[flat_index].cpu().item())
        new_code = int(alternative_codes[flat_index].cpu().item())
        entry_code = int(proxy.entry_codes().flatten()[flat_index].cpu().item())
        scale = float(proxy._fixed_scale[channel].detach().cpu().item())
        zero_point = int(proxy._fixed_zero_point[channel].detach().cpu().item())
        if new_code == entry_code:
            alternative_weight = float(
                proxy._entry_effective_weight.flatten()[flat_index]
                .detach()
                .cpu()
                .item()
            )
        else:
            alternative_weight = (new_code - zero_point) * scale
        if old_code == entry_code and new_code != entry_code:
            transition = "new"
        elif old_code != entry_code and new_code == entry_code:
            transition = "reverted"
        else:
            transition = "switched"
        return DiscreteCodeCandidate(
            rank=rank,
            group_name=binding.group.name,
            site_path=binding.group.site_path,
            family=binding.group.family,
            flat_index=flat_index,
            tensor_index=index,
            old_code=old_code,
            new_code=new_code,
            source_weight=float(
                proxy._reference_weight.flatten()[flat_index].detach().cpu().item()
            ),
            current_weight=float(
                proxy.effective_weight.flatten()[flat_index].detach().cpu().item()
            ),
            alternative_weight=alternative_weight,
            scale=scale,
            zero_point=zero_point,
            gradient=float(gradient.flatten()[flat_index].detach().cpu().item()),
            predicted_loss_delta=score,
            transition_kind=transition,
        )

    def _restore_entry_state(self) -> None:
        with torch.no_grad():
            for binding in self.bindings:
                binding.weight_module.weight.copy_(binding.entry_weight)
        for binding in self.bindings:
            device = binding.original_observer.min_val.device
            binding.original_observer.load_qparams(
                binding.entry_scale.to(device=device),
                binding.entry_zero_point.to(device=device),
                lock=binding.entry_qparams_locked,
            )
            binding.original_observer.enabled = binding.entry_enabled
            binding.original_observer.fake_quant_enabled = (
                binding.entry_fake_quant_enabled
            )
        self._restore_observers()

    def _restore_observers(self) -> None:
        for binding in self.bindings:
            current = tuple(
                getattr(binding.owner, name, None) for name in binding.attribute_names
            )
            if all(value is binding.original_observer for value in current):
                continue
            if not all(value is binding.proxy for value in current):
                raise RuntimeError(
                    f"Observer aliases for {binding.group.site_path!r} are mixed."
                )
            _replace_observer_attributes(
                binding.owner,
                binding.attribute_names,
                expected=binding.proxy,
                replacement=binding.original_observer,
                site_path=binding.group.site_path,
            )


class DiscreteCodeRefinementRunner:
    """Run explicit gradient-ranked top-K code proposals transactionally."""

    def __init__(
        self,
        config: DiscreteCodeRefinementConfig | None = None,
    ) -> None:
        self.config = config or DiscreteCodeRefinementConfig()
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
    ) -> DiscreteCodeRefinementResult:
        groups = tuple(weight_groups)
        if not groups:
            raise ValueError("Discrete refinement requires Conv weight groups.")
        if not training_samples:
            raise ValueError("Discrete refinement requires training samples.")
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
        round_results: list[DiscreteCodeRoundResult] = []
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
                self._validate_initialization(entry_selection, initialized_selection)
                current_selection: OutputMetrics = initialized_selection
                current_acceptance: OutputMetrics = entry_acceptance
                current_evaluation: OutputMetrics = entry_evaluation

                for round_index in range(1, self.config.max_rounds + 1):
                    round_entry_selection = current_selection
                    round_entry_acceptance = current_acceptance
                    round_entry_evaluation = current_evaluation
                    round_state = weights.state_snapshot()
                    sample_indices, primary, auxiliary, total = self._collect_gradient(
                        round_index,
                        teacher_cache,
                        candidate_model,
                        weights,
                        output_adapter,
                        optimization_device,
                    )
                    maximum_count = max(self.config.proposal_sizes)
                    (
                        ranked,
                        reachable_count,
                        improving_count,
                        gradient_histogram,
                    ) = weights.rank_candidates(
                        maximum_count=maximum_count,
                        minimum_predicted_improvement=(
                            self.config.minimum_predicted_improvement
                        ),
                    )
                    gradient_statistics = DiscreteCodeGradientStatistics(
                        sample_indices=sample_indices,
                        primary_loss=primary,
                        auxiliary_loss=auxiliary,
                        total_loss=total,
                        gradient_histogram=gradient_histogram,
                        reachable_candidate_count=reachable_count,
                        predicted_improving_candidate_count=improving_count,
                        recorded_candidate_count=len(ranked),
                    )
                    sizes = _proposal_sizes(
                        self.config.proposal_sizes,
                        len(ranked),
                    )
                    if not sizes:
                        stop_reason = "no predicted-improving code candidates"
                        result = self._stopped_round(
                            round_index,
                            round_entry_selection,
                            round_entry_acceptance,
                            round_entry_evaluation,
                            gradient_statistics,
                            ranked,
                            stop_reason,
                            weights,
                        )
                        round_results.append(result)
                        if progress_callback is not None:
                            progress_callback(result)
                        break

                    evaluations: list[DiscreteCodeProposalEvaluation] = []
                    selection_successes: list[int] = []
                    entry_selection_score = selection_objective.score(
                        round_entry_selection
                    )
                    for size in sizes:
                        weights.load_state_snapshot(round_state)
                        proposal = ranked[:size]
                        weights.apply_candidates(proposal)
                        outputs = copy_outputs(selection_evaluator())
                        eligible, reason = selection_objective.better(
                            outputs,
                            round_entry_selection,
                            round_entry_selection,
                        )
                        evaluation = DiscreteCodeProposalEvaluation(
                            requested_size=size,
                            applied_size=len(proposal),
                            predicted_loss_delta_sum=sum(
                                value.predicted_loss_delta for value in proposal
                            ),
                            selection_outputs=outputs,
                            selection_score=selection_objective.score(outputs),
                            selection_improvement=(
                                entry_selection_score
                                - selection_objective.score(outputs)
                            ),
                            selection_eligible=eligible,
                            selection_reason=reason,
                        )
                        evaluations.append(evaluation)
                        if eligible:
                            selection_successes.append(len(evaluations) - 1)

                    weights.load_state_snapshot(round_state)
                    if not selection_successes:
                        stop_reason = "no nested proposal improved selection metrics"
                        result = self._stopped_round(
                            round_index,
                            round_entry_selection,
                            round_entry_acceptance,
                            round_entry_evaluation,
                            gradient_statistics,
                            ranked,
                            stop_reason,
                            weights,
                            proposal_evaluations=tuple(evaluations),
                        )
                        round_results.append(result)
                        if progress_callback is not None:
                            progress_callback(result)
                        break

                    selection_successes.sort(
                        key=lambda index: evaluations[index].selection_score
                    )
                    selected_index: int | None = None
                    selected_acceptance: OutputMetrics | None = None
                    selected_evaluation: OutputMetrics | None = None
                    entry_acceptance_score = acceptance_objective.score(
                        round_entry_acceptance
                    )
                    for evaluation_index in selection_successes:
                        evaluation = evaluations[evaluation_index]
                        weights.load_state_snapshot(round_state)
                        proposal = ranked[: evaluation.applied_size]
                        weights.apply_candidates(proposal)
                        acceptance = copy_outputs(acceptance_evaluator())
                        accepted, reason = acceptance_objective.accepted(
                            acceptance,
                            round_entry_acceptance,
                        )
                        evaluation = replace(
                            evaluation,
                            acceptance_attempted=True,
                            acceptance_outputs=acceptance,
                            acceptance_improvement=(
                                entry_acceptance_score
                                - acceptance_objective.score(acceptance)
                            ),
                            acceptance_eligible=accepted,
                            acceptance_reason=reason,
                        )
                        evaluations[evaluation_index] = evaluation
                        if not accepted:
                            continue
                        selected_index = evaluation_index
                        selected_acceptance = acceptance
                        selected_evaluation = copy_outputs(evaluation_evaluator())
                        break

                    if selected_index is None:
                        weights.load_state_snapshot(round_state)
                        stop_reason = (
                            "no selection-improving proposal passed acceptance"
                        )
                        result = self._stopped_round(
                            round_index,
                            round_entry_selection,
                            round_entry_acceptance,
                            round_entry_evaluation,
                            gradient_statistics,
                            ranked,
                            stop_reason,
                            weights,
                            proposal_evaluations=tuple(evaluations),
                        )
                        round_results.append(result)
                        if progress_callback is not None:
                            progress_callback(result)
                        break

                    selected = evaluations[selected_index]
                    selected_candidates = ranked[: selected.applied_size]
                    assert selected_acceptance is not None
                    assert selected_evaluation is not None
                    current_selection = selected.selection_outputs
                    current_acceptance = selected_acceptance
                    current_evaluation = selected_evaluation
                    accepted_rounds += 1
                    transition = weights.transition_summary(selected_candidates)
                    round_result = DiscreteCodeRoundResult(
                        round_index=round_index,
                        entry_selection_outputs=round_entry_selection,
                        entry_acceptance_outputs=round_entry_acceptance,
                        entry_evaluation_outputs=round_entry_evaluation,
                        gradient_statistics=gradient_statistics,
                        ranked_candidates=ranked,
                        proposal_evaluations=tuple(evaluations),
                        selected_size=selected.applied_size,
                        selected_candidates=selected_candidates,
                        selected_selection_outputs=current_selection,
                        selected_acceptance_outputs=current_acceptance,
                        selected_evaluation_outputs=current_evaluation,
                        accepted=True,
                        acceptance_reason=(selected.acceptance_reason or "accepted"),
                        transition_summary=transition,
                    )
                    round_results.append(round_result)
                    if progress_callback is not None:
                        progress_callback(round_result)
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
        return DiscreteCodeRefinementResult(
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
                    "Discrete refinement did not reproduce the loaded checkpoint "
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
        ranked: tuple[DiscreteCodeCandidate, ...],
        reason: str,
        weights: DiscreteCodeWeightSet,
        *,
        proposal_evaluations: tuple[DiscreteCodeProposalEvaluation, ...] = (),
    ) -> DiscreteCodeRoundResult:
        return DiscreteCodeRoundResult(
            round_index=round_index,
            entry_selection_outputs=selection,
            entry_acceptance_outputs=acceptance,
            entry_evaluation_outputs=evaluation,
            gradient_statistics=gradient_statistics,
            ranked_candidates=ranked,
            proposal_evaluations=proposal_evaluations,
            selected_size=None,
            selected_candidates=(),
            selected_selection_outputs=selection,
            selected_acceptance_outputs=acceptance,
            selected_evaluation_outputs=evaluation,
            accepted=False,
            acceptance_reason=reason,
            transition_summary=weights.transition_summary(()),
            stop_reason=reason,
        )


def _build_binding(
    group: JointAdaRoundWeightGroup,
    site: QuantizationSite,
    source_weight: torch.Tensor,
) -> _DiscreteBinding:
    if site.role is not SiteRole.PARAMETER or site.observer_name != "weight":
        raise ValueError(f"Discrete site {site.path!r} must be a weight site.")
    if not isinstance(site.observer, AffineObserverBase):
        raise TypeError(f"Discrete site {site.path!r} is not affine.")
    weight_module = getattr(site.module, "module", None)
    if not isinstance(weight_module, nn.Conv2d):
        raise TypeError(
            "Discrete refinement supports Conv2d, got "
            f"{type(weight_module).__name__} at {site.path!r}."
        )
    source = source_weight.detach().to(
        device=weight_module.weight.device,
        dtype=weight_module.weight.dtype,
    )
    if source.shape != weight_module.weight.shape:
        raise ValueError(
            f"FP source shape mismatch at {site.path!r}: "
            f"{tuple(source.shape)} != {tuple(weight_module.weight.shape)}."
        )
    scale, zero_point = site.observer.compute_qparams()
    entry_weight = weight_module.weight.detach().clone()
    with torch.no_grad():
        checkpoint_effective_weight = site.observer.fake_quant(entry_weight)
        checkpoint_effective_weight = checkpoint_effective_weight.detach().clone()
    attributes = _observer_attribute_names(site.module, site.observer)
    if not attributes:
        raise RuntimeError(f"No observer alias exists for {site.path!r}.")
    proxy = FixedScaleCodeObserver(
        site.observer,
        source,
        checkpoint_effective_weight,
    )
    return _DiscreteBinding(
        group=group,
        owner=site.module,
        attribute_names=attributes,
        weight_module=weight_module,
        original_observer=site.observer,
        entry_weight=entry_weight,
        entry_scale=scale.detach().clone(),
        entry_zero_point=zero_point.detach().clone(),
        entry_qparams_locked=bool(site.observer._qparams_locked),
        entry_enabled=bool(site.observer.enabled),
        entry_fake_quant_enabled=bool(site.observer.fake_quant_enabled),
        proxy=proxy,
    )


def _gradient_sample_indices(
    sample_count: int,
    requested_count: int,
    *,
    seed: int,
) -> tuple[int, ...]:
    if sample_count <= 0:
        raise ValueError("Gradient sample count must be positive.")
    if requested_count == 0 or requested_count >= sample_count:
        return tuple(range(sample_count))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator)
    return tuple(int(value) for value in permutation[:requested_count].tolist())


def _proposal_sizes(
    requested: Sequence[int],
    candidate_count: int,
) -> tuple[int, ...]:
    if candidate_count <= 0:
        return ()
    values = {min(int(value), candidate_count) for value in requested if value > 0}
    return tuple(sorted(values, reverse=True))


def _unravel_index(flat_index: int, shape: Sequence[int]) -> tuple[int, ...]:
    if flat_index < 0:
        raise ValueError("flat_index must be nonnegative.")
    remaining = flat_index
    values: list[int] = []
    for dimension in reversed(tuple(int(value) for value in shape)):
        if dimension <= 0:
            raise ValueError("Tensor dimensions must be positive.")
        values.append(remaining % dimension)
        remaining //= dimension
    if remaining != 0:
        raise ValueError("flat_index exceeds tensor shape.")
    return tuple(reversed(values))
