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

"""Validation-aware joint scale and rounding reconstruction for Conv2d blocks."""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from tico.quantization.algorithm.adaround.joint import (
    JointAdaRoundWeightGroup,
    JointAdaRoundWeightSet,
    JointAdaRoundWeightStatistics,
)
from tico.quantization.algorithm.block_reconstruction.cache import (
    invoke_block,
    ReconstructionCache,
)
from tico.quantization.algorithm.block_reconstruction.runner import (
    reconstruction_loss,
    ReconstructionLoss,
)
from tico.quantization.algorithm.block_reconstruction.selection import (
    copy_outputs,
    metric_value,
    OutputMetrics,
)


@dataclass(frozen=True)
class JointAdaRoundObjective:
    """Minimize one metric under absolute and optional relative output guards."""

    primary_output: str = "regressors"
    primary_metric: str = "mae"
    minimum_improvement: float = 0.0
    absolute_output_limits: Mapping[str, float] = field(
        default_factory=lambda: {"classifiers": 0.1}
    )
    relative_output_tolerances: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.primary_output or not self.primary_metric:
            raise ValueError("Joint AdaRound output and metric names must be set.")
        if (
            not math.isfinite(self.minimum_improvement)
            or self.minimum_improvement < 0.0
        ):
            raise ValueError("minimum_improvement must be finite and nonnegative.")
        for values, label in (
            (self.absolute_output_limits, "absolute output limit"),
            (self.relative_output_tolerances, "relative output tolerance"),
        ):
            for output_name, value in values.items():
                if not output_name:
                    raise ValueError(f"Every {label} needs a non-empty output name.")
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"Every {label} must be finite and nonnegative.")

    def score(self, outputs: OutputMetrics) -> float:
        return metric_value(outputs, self.primary_output, self.primary_metric)

    def admissible(
        self,
        candidate: OutputMetrics,
        reference: OutputMetrics,
    ) -> tuple[bool, str]:
        checked: set[str] = set()
        for output_name, limit in self.absolute_output_limits.items():
            candidate_value = metric_value(
                candidate,
                output_name,
                self.primary_metric,
            )
            reference_value = metric_value(
                reference,
                output_name,
                self.primary_metric,
            )
            tolerance = self.relative_output_tolerances.get(output_name, 0.0)
            ceiling = max(limit, reference_value + tolerance)
            if candidate_value >= ceiling:
                return (
                    False,
                    f"{output_name}.{self.primary_metric}={candidate_value:.6e} "
                    f"did not satisfy progressive ceiling < {ceiling:.6e} "
                    f"(absolute={limit:.6e}, entry={reference_value:.6e}, "
                    f"tolerance={tolerance:.6e})",
                )
            checked.add(output_name)
        for output_name, tolerance in self.relative_output_tolerances.items():
            if output_name in checked:
                continue
            candidate_value = metric_value(
                candidate,
                output_name,
                self.primary_metric,
            )
            reference_value = metric_value(
                reference,
                output_name,
                self.primary_metric,
            )
            regression = candidate_value - reference_value
            if regression > tolerance:
                return (
                    False,
                    f"{output_name}.{self.primary_metric} regressed by "
                    f"{regression:.6e}, exceeding {tolerance:.6e}",
                )
        return True, "all progressive output constraints satisfied"

    def better(
        self,
        candidate: OutputMetrics,
        incumbent: OutputMetrics,
        reference: OutputMetrics,
    ) -> tuple[bool, str]:
        admissible, reason = self.admissible(candidate, reference)
        if not admissible:
            return False, reason
        improvement = self.score(incumbent) - self.score(candidate)
        if improvement <= self.minimum_improvement:
            return (
                False,
                f"primary improvement {improvement:.6e} did not exceed "
                f"{self.minimum_improvement:.6e}",
            )
        return True, f"primary improvement {improvement:.6e}"

    def accepted(
        self,
        candidate: OutputMetrics,
        reference: OutputMetrics,
    ) -> tuple[bool, str]:
        admissible, reason = self.admissible(candidate, reference)
        if not admissible:
            return False, reason
        improvement = self.score(reference) - self.score(candidate)
        if improvement <= self.minimum_improvement:
            return (
                False,
                f"primary improvement {improvement:.6e} did not exceed "
                f"{self.minimum_improvement:.6e}",
            )
        return True, f"primary improvement {improvement:.6e}"


@dataclass(frozen=True)
class JointAdaRoundConfig:
    """Configure one joint depthwise/pointwise scale-and-rounding window."""

    steps: int = 2_000
    batch_size: int = 8
    evaluation_batch_size: int = 16
    evaluation_interval: int = 100
    alpha_learning_rate: float = 1.0e-3
    scale_learning_rate: float = 1.0e-4
    reconstruction_loss: ReconstructionLoss = ReconstructionLoss.NORMALIZED_L1
    loss_epsilon: float = 1.0e-8
    rounding_loss_weight: float = 1.0e-2
    scale_loss_weight: float = 1.0e-3
    warmup_fraction: float = 0.2
    beta_start: float = 20.0
    beta_end: float = 2.0
    gamma: float = -0.1
    zeta: float = 1.1
    initialization_epsilon: float = 1.0e-6
    max_scale_ratio: float = 1.25
    gradient_clip_norm: float | None = 1.0
    seed: int = 20260830

    def validate(self) -> None:
        if self.steps < 0:
            raise ValueError("Joint AdaRound steps must be nonnegative.")
        if self.batch_size <= 0 or self.evaluation_batch_size <= 0:
            raise ValueError("Joint AdaRound batch sizes must be positive.")
        if self.evaluation_interval <= 0:
            raise ValueError("Joint AdaRound evaluation_interval must be positive.")
        for name, value in (
            ("alpha_learning_rate", self.alpha_learning_rate),
            ("scale_learning_rate", self.scale_learning_rate),
            ("loss_epsilon", self.loss_epsilon),
            ("beta_start", self.beta_start),
            ("beta_end", self.beta_end),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (
            ("rounding_loss_weight", self.rounding_loss_weight),
            ("scale_loss_weight", self.scale_loss_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1).")
        if self.beta_start < self.beta_end:
            raise ValueError("beta_start must be at least beta_end.")
        if not math.isfinite(self.gamma) or self.gamma >= 0.0:
            raise ValueError("gamma must be finite and negative.")
        if not math.isfinite(self.zeta) or self.zeta <= 1.0:
            raise ValueError("zeta must be finite and greater than one.")
        if not 0.0 < self.initialization_epsilon < 0.5:
            raise ValueError("initialization_epsilon must be in (0, 0.5).")
        if not math.isfinite(self.max_scale_ratio) or self.max_scale_ratio <= 1.0:
            raise ValueError("max_scale_ratio must be finite and greater than one.")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be positive or None.")
        if not isinstance(self.reconstruction_loss, ReconstructionLoss):
            raise TypeError("reconstruction_loss must be a ReconstructionLoss value.")
        if not isinstance(self.seed, int):
            raise TypeError("Joint AdaRound seed must be an integer.")


@dataclass(frozen=True)
class JointAdaRoundCheckpoint:
    """Record one hard joint scale-and-rounding validation checkpoint."""

    step: int
    train_reconstruction_loss: float | None
    train_rounding_loss: float | None
    train_scale_loss: float | None
    train_total_loss: float | None
    local_hard_loss: float
    selection_outputs: OutputMetrics | None
    primary_score: float | None
    beta: float | None
    selected_as_best: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "train_reconstruction_loss": self.train_reconstruction_loss,
            "train_rounding_loss": self.train_rounding_loss,
            "train_scale_loss": self.train_scale_loss,
            "train_total_loss": self.train_total_loss,
            "local_hard_loss": self.local_hard_loss,
            "selection_outputs": (
                copy_outputs(self.selection_outputs)
                if self.selection_outputs is not None
                else None
            ),
            "primary_score": self.primary_score,
            "beta": self.beta,
            "selected_as_best": self.selected_as_best,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JointAdaRoundResult:
    """Summarize one joint DW/PW optimization and transactional decision."""

    block: str
    steps: int
    cache_samples: int
    selection_cache_samples: int
    weight_groups: tuple[str, ...]
    weight_families: tuple[str, ...]
    initial_hard_loss: float
    selected_hard_loss: float
    best_step: int
    accepted: bool
    acceptance_reason: str
    entry_selection_outputs: OutputMetrics
    selected_outputs: OutputMetrics
    entry_acceptance_outputs: OutputMetrics
    acceptance_outputs: OutputMetrics
    weight_statistics: tuple[JointAdaRoundWeightStatistics, ...]
    checkpoint_history: tuple[JointAdaRoundCheckpoint, ...]
    training_reconstruction_history: tuple[float, ...] = ()
    training_rounding_history: tuple[float, ...] = ()
    training_scale_history: tuple[float, ...] = ()
    training_total_history: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "block": self.block,
            "steps": self.steps,
            "cache_samples": self.cache_samples,
            "selection_cache_samples": self.selection_cache_samples,
            "weight_group_count": len(self.weight_groups),
            "weight_groups": list(self.weight_groups),
            "weight_families": list(self.weight_families),
            "initial_hard_loss": self.initial_hard_loss,
            "selected_hard_loss": self.selected_hard_loss,
            "hard_loss_improvement": self.initial_hard_loss - self.selected_hard_loss,
            "best_step": self.best_step,
            "accepted": self.accepted,
            "acceptance_reason": self.acceptance_reason,
            "entry_selection_outputs": copy_outputs(self.entry_selection_outputs),
            "selected_outputs": copy_outputs(self.selected_outputs),
            "entry_acceptance_outputs": copy_outputs(self.entry_acceptance_outputs),
            "acceptance_outputs": copy_outputs(self.acceptance_outputs),
            "weight_statistics": [value.to_dict() for value in self.weight_statistics],
            "checkpoint_history": [
                value.to_dict() for value in self.checkpoint_history
            ],
            "training_reconstruction_history": list(
                self.training_reconstruction_history
            ),
            "training_rounding_history": list(self.training_rounding_history),
            "training_scale_history": list(self.training_scale_history),
            "training_total_history": list(self.training_total_history),
        }


MetricsEvaluator = Callable[[], OutputMetrics]


class JointAdaRoundRunner:
    """Jointly optimize DW/PW scales and rounding with held-out rollback."""

    def __init__(self, config: JointAdaRoundConfig | None = None) -> None:
        self.config = config or JointAdaRoundConfig()
        self.config.validate()

    def reconstruct(
        self,
        *,
        block_name: str,
        observer_model: nn.Module,
        block: nn.Module,
        cache: ReconstructionCache,
        selection_cache: ReconstructionCache,
        weight_groups: Sequence[JointAdaRoundWeightGroup],
        selection_evaluator: MetricsEvaluator,
        selection_objective: JointAdaRoundObjective,
        acceptance_evaluator: MetricsEvaluator | None = None,
        acceptance_objective: JointAdaRoundObjective | None = None,
        device: torch.device | str | None = None,
    ) -> JointAdaRoundResult:
        if not block_name:
            raise ValueError("Joint AdaRound block_name must be non-empty.")
        groups = tuple(weight_groups)
        if not groups:
            raise ValueError("Joint AdaRound requires at least one weight group.")
        if acceptance_evaluator is None:
            acceptance_evaluator = selection_evaluator
        if acceptance_objective is None:
            acceptance_objective = selection_objective

        optimization_device = torch.device(device or _module_device(observer_model))
        block.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        entry_selection = copy_outputs(selection_evaluator())
        entry_acceptance = copy_outputs(acceptance_evaluator())

        with _RequiresGradState(observer_model):
            weights = JointAdaRoundWeightSet(
                observer_model,
                groups,
                gamma=self.config.gamma,
                zeta=self.config.zeta,
                initialization_epsilon=self.config.initialization_epsilon,
                max_scale_ratio=self.config.max_scale_ratio,
            )
            try:
                parameters = tuple(
                    (*weights.alpha_parameters(), *weights.scale_parameters())
                )
                optimizer = torch.optim.Adam(
                    (
                        {
                            "params": weights.alpha_parameters(),
                            "lr": self.config.alpha_learning_rate,
                        },
                        {
                            "params": weights.scale_parameters(),
                            "lr": self.config.scale_learning_rate,
                        },
                    )
                )
                weights.set_hard(True)
                initial_hard_loss = self.evaluate_loss(
                    block,
                    selection_cache,
                    device=optimization_device,
                )
                best_state = weights.state_snapshot()
                best_step = 0
                best_hard_loss = initial_hard_loss
                best_outputs = entry_selection
                checkpoints: list[JointAdaRoundCheckpoint] = [
                    JointAdaRoundCheckpoint(
                        step=0,
                        train_reconstruction_loss=None,
                        train_rounding_loss=None,
                        train_scale_loss=None,
                        train_total_loss=None,
                        local_hard_loss=initial_hard_loss,
                        selection_outputs=entry_selection,
                        primary_score=selection_objective.score(entry_selection),
                        beta=None,
                        selected_as_best=True,
                        reason="entry round-to-nearest and calibrated-scale state",
                    )
                ]
                reconstruction_history: list[float] = []
                rounding_history: list[float] = []
                scale_history: list[float] = []
                total_history: list[float] = []

                weights.set_hard(False)
                for step in range(1, self.config.steps + 1):
                    invocation, target = cache.random_batch(
                        self.config.batch_size,
                        generator=generator,
                        device=optimization_device,
                        use_quantized_input=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    output = invoke_block(block, invocation)
                    reconstruction = reconstruction_loss(
                        output,
                        target,
                        self.config.reconstruction_loss,
                        epsilon=self.config.loss_epsilon,
                    )
                    beta = self._beta(step)
                    if beta is None:
                        rounding = reconstruction.new_zeros(())
                    else:
                        rounding = weights.rounding_regularizer(beta)
                    scale = weights.scale_regularizer()
                    total = (
                        reconstruction
                        + self.config.rounding_loss_weight * rounding
                        + self.config.scale_loss_weight * scale
                    )
                    if not torch.isfinite(total):
                        raise FloatingPointError(
                            f"Non-finite joint AdaRound loss for {block_name!r}."
                        )
                    total.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            parameters,
                            self.config.gradient_clip_norm,
                        )
                    optimizer.step()

                    train_reconstruction = float(reconstruction.detach().cpu().item())
                    train_rounding = float(rounding.detach().cpu().item())
                    train_scale = float(scale.detach().cpu().item())
                    train_total = float(total.detach().cpu().item())
                    reconstruction_history.append(train_reconstruction)
                    rounding_history.append(train_rounding)
                    scale_history.append(train_scale)
                    total_history.append(train_total)

                    should_evaluate = (
                        step % self.config.evaluation_interval == 0
                        or step == self.config.steps
                    )
                    if not should_evaluate:
                        continue
                    weights.set_hard(True)
                    hard_loss = self.evaluate_loss(
                        block,
                        selection_cache,
                        device=optimization_device,
                    )
                    candidate_outputs = copy_outputs(selection_evaluator())
                    better, reason = selection_objective.better(
                        candidate_outputs,
                        best_outputs,
                        entry_selection,
                    )
                    if better:
                        best_state = weights.state_snapshot()
                        best_step = step
                        best_hard_loss = hard_loss
                        best_outputs = candidate_outputs
                    checkpoints.append(
                        JointAdaRoundCheckpoint(
                            step=step,
                            train_reconstruction_loss=train_reconstruction,
                            train_rounding_loss=train_rounding,
                            train_scale_loss=train_scale,
                            train_total_loss=train_total,
                            local_hard_loss=hard_loss,
                            selection_outputs=candidate_outputs,
                            primary_score=selection_objective.score(candidate_outputs),
                            beta=beta,
                            selected_as_best=better,
                            reason=reason,
                        )
                    )
                    weights.set_hard(False)

                weights.load_state_snapshot(best_state)
                weights.set_hard(True)
                selected_hard_loss = self.evaluate_loss(
                    block,
                    selection_cache,
                    device=optimization_device,
                )
                selected_outputs = copy_outputs(selection_evaluator())
                acceptance_outputs = copy_outputs(acceptance_evaluator())
                accepted, acceptance_reason = acceptance_objective.accepted(
                    acceptance_outputs,
                    entry_acceptance,
                )
                selected_statistics = weights.statistics()
                if accepted:
                    weight_statistics = weights.finalize()
                else:
                    weights.restore()
                    weight_statistics = selected_statistics
            except Exception:
                weights.restore()
                raise

        return JointAdaRoundResult(
            block=block_name,
            steps=self.config.steps,
            cache_samples=len(cache),
            selection_cache_samples=len(selection_cache),
            weight_groups=tuple(group.name for group in groups),
            weight_families=tuple(group.family for group in groups),
            initial_hard_loss=initial_hard_loss,
            selected_hard_loss=selected_hard_loss,
            best_step=best_step,
            accepted=accepted,
            acceptance_reason=acceptance_reason,
            entry_selection_outputs=entry_selection,
            selected_outputs=selected_outputs,
            entry_acceptance_outputs=entry_acceptance,
            acceptance_outputs=acceptance_outputs,
            weight_statistics=weight_statistics,
            checkpoint_history=tuple(checkpoints),
            training_reconstruction_history=tuple(reconstruction_history),
            training_rounding_history=tuple(rounding_history),
            training_scale_history=tuple(scale_history),
            training_total_history=tuple(total_history),
        )

    def evaluate_loss(
        self,
        block: nn.Module,
        cache: ReconstructionCache,
        *,
        device: torch.device | str,
    ) -> float:
        numerator = torch.zeros((), dtype=torch.float64, device=device)
        denominator = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for invocation, target in cache.sequential_batches(
                self.config.evaluation_batch_size,
                device=device,
                use_quantized_input=True,
            ):
                output = invoke_block(block, invocation)
                for candidate, reference in _tensor_pairs(output, target):
                    candidate64 = candidate.to(dtype=torch.float64)
                    reference64 = reference.to(dtype=torch.float64)
                    if (
                        self.config.reconstruction_loss
                        is ReconstructionLoss.NORMALIZED_L1
                    ):
                        numerator += (candidate64 - reference64).abs().sum()
                        denominator += reference64.abs().sum()
                    else:
                        numerator += (candidate64 - reference64).square().sum()
                        denominator += reference64.square().sum()
        value = numerator / denominator.clamp_min(self.config.loss_epsilon)
        return float(value.detach().cpu().item())

    def _beta(self, step: int) -> float | None:
        warmup_steps = int(round(self.config.steps * self.config.warmup_fraction))
        if step <= warmup_steps:
            return None
        remaining = max(self.config.steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
        return (
            self.config.beta_start
            + (self.config.beta_end - self.config.beta_start) * progress
        )


class _RequiresGradState(AbstractContextManager["_RequiresGradState"]):
    def __init__(self, module: nn.Module) -> None:
        self._states = tuple(
            (parameter, parameter.requires_grad) for parameter in module.parameters()
        )

    def __enter__(self) -> "_RequiresGradState":
        for parameter, _ in self._states:
            parameter.requires_grad_(False)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for parameter, requires_grad in self._states:
            parameter.requires_grad_(requires_grad)
        return None


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _tensor_pairs(candidate, target):
    from tico.quantization.algorithm.block_reconstruction.cache import tree_tensor_pairs

    return tree_tensor_pairs(candidate, target)
