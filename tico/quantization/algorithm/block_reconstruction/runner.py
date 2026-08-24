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

"""Validation-aware block reconstruction with fixed weights and learnable qparams."""

from __future__ import annotations

import math

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from tico.quantization.algorithm.block_reconstruction.cache import (
    invoke_block,
    ReconstructionCache,
    TensorTree,
    tree_tensor_pairs,
)
from tico.quantization.algorithm.block_reconstruction.observer import (
    AffineObserverGroup,
    LearnableObserverSet,
)
from tico.quantization.algorithm.block_reconstruction.qdrop import (
    qdrop_context,
    QDropController,
)
from tico.quantization.algorithm.block_reconstruction.selection import (
    copy_outputs,
    OutputMetrics,
    ValidationObjective,
)


class ReconstructionLoss(str, Enum):
    """Supported local reconstruction objectives."""

    NORMALIZED_MSE = "normalized-mse"
    NORMALIZED_L1 = "normalized-l1"


@dataclass(frozen=True)
class BlockReconstructionConfig:
    """Configure one activation-qparam block or joint-window run."""

    steps: int = 500
    batch_size: int = 8
    evaluation_batch_size: int = 16
    evaluation_interval: int = 25
    scale_learning_rate: float = 1.0e-3
    zero_point_learning_rate: float = 1.0e-2
    optimize_scale: bool = True
    optimize_zero_point: bool = True
    gradient_clip_norm: float | None = 1.0
    minimum_scale: float = 1.0e-12
    loss_epsilon: float = 1.0e-8
    seed: int = 20260816
    loss: ReconstructionLoss = ReconstructionLoss.NORMALIZED_MSE
    qdrop_probability: float = 0.0
    qdrop_seed: int = 20260817

    def validate(self) -> None:
        """Reject invalid or numerically unsafe reconstruction settings."""
        if self.steps < 0:
            raise ValueError("steps must be nonnegative.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive.")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive.")
        if not self.optimize_scale and not self.optimize_zero_point:
            raise ValueError("At least one activation qparam must be trainable.")
        for name, value in (
            ("scale_learning_rate", self.scale_learning_rate),
            ("zero_point_learning_rate", self.zero_point_learning_rate),
            ("minimum_scale", self.minimum_scale),
            ("loss_epsilon", self.loss_epsilon),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive, or None.")
        if not isinstance(self.loss, ReconstructionLoss):
            raise TypeError("loss must be a ReconstructionLoss value.")
        if not math.isfinite(self.qdrop_probability) or not (
            0.0 <= self.qdrop_probability <= 1.0
        ):
            raise ValueError("QDrop probability must be finite and in [0, 1].")
        if not isinstance(self.qdrop_seed, int):
            raise TypeError("qdrop_seed must be an integer.")


@dataclass(frozen=True)
class ReconstructionCheckpoint:
    """Record one locally and globally evaluated optimizer checkpoint."""

    step: int
    train_loss: float | None
    local_loss: float
    selection_outputs: OutputMetrics | None
    primary_score: float | None
    selected_as_best: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible checkpoint record."""
        return {
            "step": self.step,
            "train_loss": self.train_loss,
            "local_loss": self.local_loss,
            "selection_outputs": (
                copy_outputs(self.selection_outputs)
                if self.selection_outputs is not None
                else None
            ),
            "primary_score": self.primary_score,
            "selected_as_best": self.selected_as_best,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BlockReconstructionResult:
    """Summarize local optimization and held-out end-to-end acceptance."""

    block: str
    steps: int
    cache_samples: int
    qparam_groups: tuple[str, ...]
    initial_loss: float
    final_loss: float
    best_step: int
    qparams: Mapping[str, Mapping[str, object]]
    training_loss_history: tuple[float, ...]
    evaluation_loss_history: tuple[tuple[int, float], ...]
    selected_qparams: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    selection_cache_samples: int = 0
    loss_name: str = ReconstructionLoss.NORMALIZED_MSE.value
    accepted: bool = True
    acceptance_reason: str = "local best state committed"
    entry_selection_outputs: OutputMetrics | None = None
    selected_outputs: OutputMetrics | None = None
    checkpoint_history: tuple[ReconstructionCheckpoint, ...] = ()
    qdrop_probability: float = 0.0
    qdrop_seed: int = 20260817
    qdrop_statistics: Mapping[str, float | int | str] = field(default_factory=dict)

    @property
    def loss_improvement(self) -> float:
        """Return the selected-cache local-loss reduction."""
        return self.initial_loss - self.final_loss

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible reconstruction result."""
        return {
            "block": self.block,
            "steps": self.steps,
            "cache_samples": self.cache_samples,
            "selection_cache_samples": self.selection_cache_samples,
            "qparam_group_count": len(self.qparam_groups),
            "qparam_groups": list(self.qparam_groups),
            "loss": self.loss_name,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_step": self.best_step,
            "loss_improvement": self.loss_improvement,
            "accepted": self.accepted,
            "acceptance_reason": self.acceptance_reason,
            "entry_selection_outputs": (
                copy_outputs(self.entry_selection_outputs)
                if self.entry_selection_outputs is not None
                else None
            ),
            "selected_outputs": (
                copy_outputs(self.selected_outputs)
                if self.selected_outputs is not None
                else None
            ),
            "qparams": {name: dict(values) for name, values in self.qparams.items()},
            "selected_qparams": {
                name: dict(values) for name, values in self.selected_qparams.items()
            },
            "training_loss_history": list(self.training_loss_history),
            "evaluation_loss_history": [
                {"step": step, "loss": loss}
                for step, loss in self.evaluation_loss_history
            ],
            "checkpoint_history": [
                checkpoint.to_dict() for checkpoint in self.checkpoint_history
            ],
            "qdrop_probability": self.qdrop_probability,
            "qdrop_seed": self.qdrop_seed,
            "qdrop_statistics": dict(self.qdrop_statistics),
        }


SelectionEvaluator = Callable[[], OutputMetrics]


class BlockReconstructor:
    """Optimize activation qparams and commit only validated states."""

    def __init__(self, config: BlockReconstructionConfig | None = None) -> None:
        self.config = config or BlockReconstructionConfig()
        self.config.validate()

    def reconstruct(
        self,
        *,
        block_name: str,
        observer_model: nn.Module,
        block: nn.Module,
        cache: ReconstructionCache,
        observer_groups: Sequence[AffineObserverGroup],
        selection_cache: ReconstructionCache | None = None,
        selection_evaluator: SelectionEvaluator | None = None,
        selection_objective: ValidationObjective | None = None,
        device: torch.device | str | None = None,
    ) -> BlockReconstructionResult:
        """Optimize one block/window and accept or rollback its best state."""
        if not block_name:
            raise ValueError("block_name must be non-empty.")
        groups = tuple(observer_groups)
        if not groups:
            raise ValueError("At least one trainable observer group is required.")
        if selection_evaluator is None and selection_objective is not None:
            raise ValueError("selection_objective requires selection_evaluator.")
        if selection_evaluator is not None and selection_objective is None:
            selection_objective = ValidationObjective()
        selected_cache = selection_cache or cache
        optimization_device = torch.device(device or _module_device(observer_model))
        block.eval()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        qdrop = QDropController(
            self.config.qdrop_probability,
            seed=self.config.qdrop_seed,
        )
        with _RequiresGradState(observer_model):
            observers = LearnableObserverSet(
                observer_model,
                groups,
                optimize_scale=self.config.optimize_scale,
                optimize_zero_point=self.config.optimize_zero_point,
                minimum_scale=self.config.minimum_scale,
            )
            try:
                parameters = observers.trainable_parameters()
                if not parameters:
                    raise ValueError(
                        "The reconstruction configuration has no parameters."
                    )
                optimizer = _make_optimizer(observers, self.config)
                initial_loss = self.evaluate_loss(
                    block,
                    selected_cache,
                    device=optimization_device,
                )
                entry_outputs = (
                    copy_outputs(selection_evaluator())
                    if selection_evaluator is not None
                    else None
                )
                best_loss = initial_loss
                best_step = 0
                entry_state = observers.state_snapshot()
                entry_qparams = observers.qparams_dict()
                best_state = entry_state
                best_outputs = entry_outputs
                training_history: list[float] = []
                evaluation_history: list[tuple[int, float]] = [(0, initial_loss)]
                checkpoint_history: list[ReconstructionCheckpoint] = [
                    ReconstructionCheckpoint(
                        step=0,
                        train_loss=None,
                        local_loss=initial_loss,
                        selection_outputs=entry_outputs,
                        primary_score=(
                            selection_objective.score(entry_outputs)
                            if selection_objective is not None
                            and entry_outputs is not None
                            else initial_loss
                        ),
                        selected_as_best=True,
                        reason="entry state",
                    )
                ]
                last_train_loss: float | None = None

                for step in range(1, self.config.steps + 1):
                    indices = torch.randint(
                        len(cache),
                        (self.config.batch_size,),
                        generator=generator,
                    ).tolist()
                    quantized_invocation, target = cache.batch(
                        indices,
                        device=optimization_device,
                        use_quantized_input=True,
                    )
                    if qdrop.enabled:
                        float_invocation, _ = cache.batch(
                            indices,
                            device=optimization_device,
                            use_quantized_input=False,
                        )
                        invocation = qdrop.mix_invocations(
                            float_invocation,
                            quantized_invocation,
                        )
                    else:
                        invocation = quantized_invocation
                    optimizer.zero_grad(set_to_none=True)
                    with qdrop_context(qdrop):
                        output = invoke_block(block, invocation)
                    loss = reconstruction_loss(
                        output,
                        target,
                        self.config.loss,
                        epsilon=self.config.loss_epsilon,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite block reconstruction loss for {block_name!r}."
                        )
                    loss.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            parameters,
                            self.config.gradient_clip_norm,
                        )
                    optimizer.step()
                    last_train_loss = float(loss.detach().cpu().item())
                    training_history.append(last_train_loss)

                    evaluate = (
                        step % self.config.evaluation_interval == 0
                        or step == self.config.steps
                    )
                    if not evaluate:
                        continue
                    local_value = self.evaluate_loss(
                        block,
                        selected_cache,
                        device=optimization_device,
                    )
                    evaluation_history.append((step, local_value))
                    candidate_outputs = (
                        copy_outputs(selection_evaluator())
                        if selection_evaluator is not None
                        else None
                    )
                    selected, reason = _candidate_is_better(
                        local_value=local_value,
                        candidate_outputs=candidate_outputs,
                        best_local=best_loss,
                        best_outputs=best_outputs,
                        entry_outputs=entry_outputs,
                        objective=selection_objective,
                    )
                    if selected:
                        best_loss = local_value
                        best_step = step
                        best_state = observers.state_snapshot()
                        best_outputs = candidate_outputs
                    checkpoint_history.append(
                        ReconstructionCheckpoint(
                            step=step,
                            train_loss=last_train_loss,
                            local_loss=local_value,
                            selection_outputs=candidate_outputs,
                            primary_score=(
                                selection_objective.score(candidate_outputs)
                                if selection_objective is not None
                                and candidate_outputs is not None
                                else local_value
                            ),
                            selected_as_best=selected,
                            reason=reason,
                        )
                    )

                observers.load_state_snapshot(best_state)
                final_loss = self.evaluate_loss(
                    block,
                    selected_cache,
                    device=optimization_device,
                )
                selected_qparams = observers.qparams_dict()
                if (
                    selection_objective is not None
                    and best_outputs is not None
                    and entry_outputs is not None
                ):
                    accepted, acceptance_reason = selection_objective.accepted(
                        best_outputs,
                        entry_outputs,
                    )
                else:
                    # Preserve local-only behavior when no held-out evaluator is supplied:
                    # commit the best local state, including unchanged step zero.
                    accepted = True
                    acceptance_reason = (
                        "local best state committed; improvement "
                        f"{initial_loss - final_loss:.6e}"
                    )
                if accepted:
                    committed_qparams = observers.finalize()
                else:
                    observers.restore()
                    committed_qparams = entry_qparams
            except Exception:
                observers.restore()
                raise

        return BlockReconstructionResult(
            block=block_name,
            steps=self.config.steps,
            cache_samples=len(cache),
            selection_cache_samples=len(selected_cache),
            qparam_groups=tuple(group.name for group in groups),
            loss_name=self.config.loss.value,
            initial_loss=initial_loss,
            final_loss=final_loss,
            best_step=best_step,
            qparams=committed_qparams,
            selected_qparams=selected_qparams,
            training_loss_history=tuple(training_history),
            evaluation_loss_history=tuple(evaluation_history),
            accepted=accepted,
            acceptance_reason=acceptance_reason,
            entry_selection_outputs=entry_outputs,
            selected_outputs=best_outputs,
            checkpoint_history=tuple(checkpoint_history),
            qdrop_probability=self.config.qdrop_probability,
            qdrop_seed=self.config.qdrop_seed,
            qdrop_statistics=qdrop.statistics().to_dict(),
        )

    def evaluate_loss(
        self,
        block: nn.Module,
        cache: ReconstructionCache,
        *,
        device: torch.device | str,
    ) -> float:
        """Evaluate the selected normalized local loss over the full cache."""
        numerator = torch.zeros((), dtype=torch.float64, device=device)
        denominator = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for invocation, target in cache.sequential_batches(
                self.config.evaluation_batch_size,
                device=device,
                use_quantized_input=True,
            ):
                output = invoke_block(block, invocation)
                batch_numerator, batch_denominator = reconstruction_loss_terms(
                    output,
                    target,
                    self.config.loss,
                )
                numerator += batch_numerator.to(dtype=torch.float64)
                denominator += batch_denominator.to(dtype=torch.float64)
        value = numerator / denominator.clamp_min(self.config.loss_epsilon)
        if not torch.isfinite(value):
            raise FloatingPointError("Block reconstruction evaluation is non-finite.")
        return float(value.detach().cpu().item())


def reconstruction_loss(
    candidate: TensorTree,
    target: TensorTree,
    loss: ReconstructionLoss,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return normalized MSE or normalized L1 over all output tensors."""
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive.")
    numerator, denominator = reconstruction_loss_terms(candidate, target, loss)
    return numerator / denominator.clamp_min(epsilon)


def normalized_mse_loss(
    candidate: TensorTree,
    target: TensorTree,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return the normalized-MSE public compatibility helper."""
    return reconstruction_loss(
        candidate,
        target,
        ReconstructionLoss.NORMALIZED_MSE,
        epsilon=epsilon,
    )


def normalized_l1_loss(
    candidate: TensorTree,
    target: TensorTree,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return magnitude-normalized absolute error over all output tensors."""
    return reconstruction_loss(
        candidate,
        target,
        ReconstructionLoss.NORMALIZED_L1,
        epsilon=epsilon,
    )


def normalized_mse_terms(
    candidate: TensorTree,
    target: TensorTree,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable normalized-MSE numerator and denominator terms."""
    return reconstruction_loss_terms(
        candidate,
        target,
        ReconstructionLoss.NORMALIZED_MSE,
    )


def reconstruction_loss_terms(
    candidate: TensorTree,
    target: TensorTree,
    loss: ReconstructionLoss,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable error and target-normalization terms."""
    pairs = tree_tensor_pairs(candidate, target)
    if loss is ReconstructionLoss.NORMALIZED_MSE:
        numerator = sum(
            (candidate_tensor - target_tensor).to(dtype=torch.float32).square().sum()
            for candidate_tensor, target_tensor in pairs
        )
        denominator = sum(
            target_tensor.to(dtype=torch.float32).square().sum()
            for _, target_tensor in pairs
        )
    elif loss is ReconstructionLoss.NORMALIZED_L1:
        numerator = sum(
            (candidate_tensor - target_tensor).to(dtype=torch.float32).abs().sum()
            for candidate_tensor, target_tensor in pairs
        )
        denominator = sum(
            target_tensor.to(dtype=torch.float32).abs().sum()
            for _, target_tensor in pairs
        )
    else:
        raise ValueError(f"Unsupported reconstruction loss: {loss!r}")
    return numerator, denominator


def _candidate_is_better(
    *,
    local_value: float,
    candidate_outputs: OutputMetrics | None,
    best_local: float,
    best_outputs: OutputMetrics | None,
    entry_outputs: OutputMetrics | None,
    objective: ValidationObjective | None,
) -> tuple[bool, str]:
    if objective is None:
        improvement = best_local - local_value
        return improvement > 0.0, f"local improvement {improvement:.6e}"
    assert candidate_outputs is not None
    assert best_outputs is not None
    assert entry_outputs is not None
    return objective.better(candidate_outputs, best_outputs, entry_outputs)


def _make_optimizer(
    observers: LearnableObserverSet,
    config: BlockReconstructionConfig,
) -> torch.optim.Optimizer:
    scale_parameters = tuple(
        group.proxy.log_scale
        for group in observers.groups
        if group.proxy.log_scale.requires_grad
    )
    zero_point_parameters = tuple(
        parameter
        for group in observers.groups
        if (parameter := group.proxy.zero_point_parameter) is not None
        and parameter.requires_grad
    )
    parameter_groups: list[dict[str, object]] = []
    if scale_parameters:
        parameter_groups.append(
            {"params": scale_parameters, "lr": config.scale_learning_rate}
        )
    if zero_point_parameters:
        parameter_groups.append(
            {"params": zero_point_parameters, "lr": config.zero_point_learning_rate}
        )
    if not parameter_groups:
        raise ValueError("No learnable activation qparams were configured.")
    return torch.optim.Adam(parameter_groups)


class _RequiresGradState(AbstractContextManager["_RequiresGradState"]):
    """Temporarily freeze existing model parameters during qparam optimization."""

    def __init__(self, model: nn.Module) -> None:
        self._states = tuple(
            (parameter, parameter.requires_grad) for parameter in model.parameters()
        )

    def __enter__(self) -> "_RequiresGradState":
        for parameter, _ in self._states:
            parameter.requires_grad_(False)
            parameter.grad = None
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for parameter, requires_grad in self._states:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
        return None


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")
