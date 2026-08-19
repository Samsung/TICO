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

"""Per-window QDrop probability/seed competition on a held-out acceptance set."""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import nn

from tico.quantization.algorithm.block_reconstruction.cache import ReconstructionCache
from tico.quantization.algorithm.block_reconstruction.observer import (
    AffineObserverGroup,
)
from tico.quantization.algorithm.block_reconstruction.runner import (
    BlockReconstructionConfig,
    BlockReconstructionResult,
    BlockReconstructor,
    SelectionEvaluator,
)
from tico.quantization.algorithm.block_reconstruction.selection import (
    copy_outputs,
    OutputMetrics,
    ValidationObjective,
)
from tico.quantization.wrapq.control import iter_quantization_sites
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase


@dataclass(frozen=True)
class QDropCandidate:
    """Describe one QDrop probability and random-mask seed."""

    probability: float
    seed: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("QDrop candidate probability must be in [0, 1].")
        if not isinstance(self.seed, int):
            raise TypeError("QDrop candidate seed must be an integer.")

    @property
    def name(self) -> str:
        """Return a stable report key."""
        probability = f"{self.probability:g}".replace(".", "_")
        if self.probability == 0.0:
            return "qdrop_0"
        return f"qdrop_{probability}_seed_{self.seed}"

    def to_dict(self) -> dict[str, float | int | str]:
        """Return JSON-compatible candidate metadata."""
        return {
            "name": self.name,
            "probability": self.probability,
            "seed": self.seed,
        }


def build_qdrop_candidates(
    probabilities: Sequence[float],
    seeds: Sequence[int],
) -> tuple[QDropCandidate, ...]:
    """Build a stable candidate product and deduplicate the deterministic control."""
    if not probabilities:
        raise ValueError("At least one QDrop probability is required.")
    if not seeds:
        raise ValueError("At least one QDrop seed is required.")
    unique_probabilities = tuple(dict.fromkeys(float(value) for value in probabilities))
    unique_seeds = tuple(dict.fromkeys(int(value) for value in seeds))
    candidates: list[QDropCandidate] = []
    for probability in unique_probabilities:
        if probability == 0.0:
            candidates.append(QDropCandidate(0.0, unique_seeds[0]))
            continue
        candidates.extend(QDropCandidate(probability, seed) for seed in unique_seeds)
    return tuple(candidates)


@dataclass(frozen=True)
class _AffineQParamSiteState:
    site_path: str
    minimum: torch.Tensor
    maximum: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    qparams_locked: bool
    enabled: bool
    fake_quant_enabled: bool


@dataclass(frozen=True)
class AffineQParamSnapshot:
    """Capture and restore the exact original-observer state for one window."""

    sites: tuple[_AffineQParamSiteState, ...]

    @classmethod
    def capture(
        cls,
        model: nn.Module,
        observer_groups: Sequence[AffineObserverGroup],
    ) -> "AffineQParamSnapshot":
        """Capture every unique site used by the supplied observer groups."""
        paths = tuple(
            dict.fromkeys(
                path for group in observer_groups for path in group.site_paths
            )
        )
        if not paths:
            raise ValueError("A qparam snapshot requires at least one observer site.")
        sites = {site.path: site for site in iter_quantization_sites(model)}
        missing = tuple(path for path in paths if path not in sites)
        if missing:
            raise KeyError(f"Unknown qparam snapshot sites: {missing}.")
        states: list[_AffineQParamSiteState] = []
        for path in paths:
            observer = sites[path].observer
            if not isinstance(observer, AffineObserverBase):
                raise TypeError(f"Quantization site {path!r} is not affine.")
            scale, zero_point = observer.compute_qparams()
            states.append(
                _AffineQParamSiteState(
                    site_path=path,
                    minimum=observer.min_val.detach().cpu().clone(),
                    maximum=observer.max_val.detach().cpu().clone(),
                    scale=scale.detach().cpu().clone(),
                    zero_point=zero_point.detach().cpu().clone(),
                    qparams_locked=bool(getattr(observer, "_qparams_locked", True)),
                    enabled=bool(observer.enabled),
                    fake_quant_enabled=bool(observer.fake_quant_enabled),
                )
            )
        return cls(tuple(states))

    def restore(self, model: nn.Module) -> None:
        """Restore all observer statistics, qparams, and runtime switches."""
        sites = {site.path: site for site in iter_quantization_sites(model)}
        for state in self.sites:
            site = sites.get(state.site_path)
            if site is None:
                raise KeyError(f"Unknown qparam restore site {state.site_path!r}.")
            observer = site.observer
            if not isinstance(observer, AffineObserverBase):
                raise TypeError(f"Quantization site {state.site_path!r} is not affine.")
            device = observer.min_val.device
            observer.min_val.copy_(state.minimum.to(device=device))
            observer.max_val.copy_(state.maximum.to(device=device))
            observer.load_qparams(
                state.scale.to(device=device),
                state.zero_point.to(device=device),
                lock=state.qparams_locked,
            )
            observer.enabled = state.enabled
            observer.fake_quant_enabled = state.fake_quant_enabled

    def to_dict(self) -> dict[str, dict[str, float | int | bool]]:
        """Return scalar qparams for audit-friendly reports."""
        return {
            state.site_path: {
                "scale": float(state.scale.reshape(-1)[0].item()),
                "zero_point": int(state.zero_point.reshape(-1)[0].item()),
                "qparams_locked": state.qparams_locked,
                "enabled": state.enabled,
                "fake_quant_enabled": state.fake_quant_enabled,
            }
            for state in self.sites
        }


@dataclass(frozen=True)
class QDropCandidateResult:
    """Store one selected checkpoint and its independent acceptance result."""

    candidate: QDropCandidate
    reconstruction: BlockReconstructionResult
    acceptance_outputs: OutputMetrics
    acceptance_score: float
    acceptance_eligible: bool
    acceptance_reason: str
    selected_as_winner: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible candidate diagnostics."""
        return {
            "candidate": self.candidate.to_dict(),
            "selection_accepted": self.reconstruction.accepted,
            "acceptance_eligible": self.acceptance_eligible,
            "acceptance_reason": self.acceptance_reason,
            "acceptance_score": self.acceptance_score,
            "selected_as_winner": self.selected_as_winner,
            "acceptance_outputs": copy_outputs(self.acceptance_outputs),
            "reconstruction": self.reconstruction.to_dict(),
        }


@dataclass(frozen=True)
class QDropMultiStartResult:
    """Summarize per-window candidate competition on an acceptance set."""

    block: str
    accepted: bool
    winner: QDropCandidate | None
    acceptance_reason: str
    entry_acceptance_outputs: OutputMetrics
    selected_acceptance_outputs: OutputMetrics
    candidates: tuple[QDropCandidateResult, ...]
    committed_qparams: Mapping[str, Mapping[str, float | int | bool]]

    def to_dict(self) -> dict[str, object]:
        """Return a complete JSON-compatible competition report."""
        return {
            "block": self.block,
            "accepted": self.accepted,
            "winner": self.winner.to_dict() if self.winner is not None else None,
            "acceptance_reason": self.acceptance_reason,
            "entry_acceptance_outputs": copy_outputs(self.entry_acceptance_outputs),
            "selected_acceptance_outputs": copy_outputs(
                self.selected_acceptance_outputs
            ),
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "committed_qparams": {
                name: dict(values) for name, values in self.committed_qparams.items()
            },
        }


class QDropMultiStartReconstructor:
    """Run all QDrop candidates from one entry state and commit one winner."""

    def __init__(
        self,
        base_config: BlockReconstructionConfig,
        candidates: Sequence[QDropCandidate],
        *,
        acceptance_objective: ValidationObjective,
    ) -> None:
        if not candidates:
            raise ValueError("QDrop multi-start requires at least one candidate.")
        names = tuple(candidate.name for candidate in candidates)
        if len(set(names)) != len(names):
            raise ValueError("QDrop candidate names must be unique.")
        self.base_config = base_config
        self.candidates = tuple(candidates)
        self.acceptance_objective = acceptance_objective

    def reconstruct(
        self,
        *,
        block_name: str,
        observer_model: nn.Module,
        block: nn.Module,
        cache: ReconstructionCache,
        selection_cache: ReconstructionCache,
        observer_groups: Sequence[AffineObserverGroup],
        selection_evaluator: SelectionEvaluator,
        selection_objective: ValidationObjective,
        acceptance_evaluator: SelectionEvaluator,
        device: torch.device | str | None = None,
    ) -> QDropMultiStartResult:
        """Select checkpoints on one split and candidates on another split."""
        groups = tuple(observer_groups)
        entry_state = AffineQParamSnapshot.capture(observer_model, groups)
        entry_acceptance = copy_outputs(acceptance_evaluator())
        best_state = entry_state
        best_outputs = entry_acceptance
        best_candidate: QDropCandidate | None = None
        records: list[
            tuple[
                QDropCandidate,
                BlockReconstructionResult,
                OutputMetrics,
                float,
                bool,
                str,
            ]
        ] = []
        try:
            for candidate in self.candidates:
                entry_state.restore(observer_model)
                config = replace(
                    self.base_config,
                    qdrop_probability=candidate.probability,
                    qdrop_seed=candidate.seed,
                )
                reconstruction = BlockReconstructor(config).reconstruct(
                    block_name=block_name,
                    observer_model=observer_model,
                    block=block,
                    cache=cache,
                    selection_cache=selection_cache,
                    observer_groups=groups,
                    selection_evaluator=selection_evaluator,
                    selection_objective=selection_objective,
                    device=device,
                )
                candidate_state = AffineQParamSnapshot.capture(
                    observer_model,
                    groups,
                )
                acceptance_outputs = copy_outputs(acceptance_evaluator())
                acceptance_score = self.acceptance_objective.score(acceptance_outputs)
                eligible, reason = self.acceptance_objective.accepted(
                    acceptance_outputs,
                    entry_acceptance,
                )
                if eligible:
                    better, comparison = self.acceptance_objective.better(
                        acceptance_outputs,
                        best_outputs,
                        entry_acceptance,
                    )
                    if better:
                        best_state = candidate_state
                        best_outputs = acceptance_outputs
                        best_candidate = candidate
                        reason = comparison
                    else:
                        reason = f"eligible but not selected: {comparison}"
                records.append(
                    (
                        candidate,
                        reconstruction,
                        acceptance_outputs,
                        acceptance_score,
                        eligible,
                        reason,
                    )
                )
        except Exception:
            entry_state.restore(observer_model)
            raise

        if best_candidate is None:
            entry_state.restore(observer_model)
            acceptance_reason = "entry state won acceptance-set competition"
            committed = entry_state
        else:
            best_state.restore(observer_model)
            improvement = self.acceptance_objective.score(
                entry_acceptance
            ) - self.acceptance_objective.score(best_outputs)
            acceptance_reason = (
                f"{best_candidate.name} won acceptance-set competition; "
                f"primary improvement {improvement:.6e}"
            )
            committed = best_state

        candidate_results = tuple(
            QDropCandidateResult(
                candidate=candidate,
                reconstruction=reconstruction,
                acceptance_outputs=acceptance_outputs,
                acceptance_score=acceptance_score,
                acceptance_eligible=eligible,
                acceptance_reason=reason,
                selected_as_winner=(candidate == best_candidate),
            )
            for (
                candidate,
                reconstruction,
                acceptance_outputs,
                acceptance_score,
                eligible,
                reason,
            ) in records
        )
        return QDropMultiStartResult(
            block=block_name,
            accepted=best_candidate is not None,
            winner=best_candidate,
            acceptance_reason=acceptance_reason,
            entry_acceptance_outputs=entry_acceptance,
            selected_acceptance_outputs=best_outputs,
            candidates=candidate_results,
            committed_qparams=committed.to_dict(),
        )
