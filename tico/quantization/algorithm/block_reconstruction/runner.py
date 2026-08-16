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

"""Gradient-based block reconstruction with fixed weights and learnable qparams."""

from __future__ import annotations

import math

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Mapping, Sequence

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


@dataclass(frozen=True)
class BlockReconstructionConfig:
    """Configure one activation-qparam block reconstruction run."""

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


@dataclass(frozen=True)
class BlockReconstructionResult:
    """Summarize one optimized block and its persisted affine qparams."""

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

    @property
    def loss_improvement(self) -> float:
        """Return the local normalized-MSE reduction."""
        return self.initial_loss - self.final_loss

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible reconstruction result."""
        return {
            "block": self.block,
            "steps": self.steps,
            "cache_samples": self.cache_samples,
            "qparam_group_count": len(self.qparam_groups),
            "qparam_groups": list(self.qparam_groups),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_step": self.best_step,
            "loss_improvement": self.loss_improvement,
            "qparams": {name: dict(values) for name, values in self.qparams.items()},
            "training_loss_history": list(self.training_loss_history),
            "evaluation_loss_history": [
                {"step": step, "loss": loss}
                for step, loss in self.evaluation_loss_history
            ],
        }


class BlockReconstructor:
    """Optimize affine activation qparams for one executable block."""

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
        device: torch.device | str | None = None,
    ) -> BlockReconstructionResult:
        """Optimize selected activation qparams while keeping weights fixed."""
        if not block_name:
            raise ValueError("block_name must be non-empty.")
        groups = tuple(observer_groups)
        if not groups:
            raise ValueError("At least one trainable observer group is required.")
        optimization_device = torch.device(device or _module_device(observer_model))
        block.eval()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
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
                    cache,
                    device=optimization_device,
                )
                best_loss = initial_loss
                best_step = 0
                best_state = observers.state_snapshot()
                training_history: list[float] = []
                evaluation_history: list[tuple[int, float]] = [(0, initial_loss)]

                for step in range(1, self.config.steps + 1):
                    invocation, target = cache.random_batch(
                        self.config.batch_size,
                        generator=generator,
                        device=optimization_device,
                        use_quantized_input=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    output = invoke_block(block, invocation)
                    loss = normalized_mse_loss(
                        output,
                        target,
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
                    training_history.append(float(loss.detach().cpu().item()))

                    if (
                        step % self.config.evaluation_interval == 0
                        or step == self.config.steps
                    ):
                        evaluation_loss = self.evaluate_loss(
                            block,
                            cache,
                            device=optimization_device,
                        )
                        evaluation_history.append((step, evaluation_loss))
                        if evaluation_loss < best_loss:
                            best_loss = evaluation_loss
                            best_step = step
                            best_state = observers.state_snapshot()

                observers.load_state_snapshot(best_state)
                final_loss = self.evaluate_loss(
                    block,
                    cache,
                    device=optimization_device,
                )
                qparams = observers.finalize()
            except Exception:
                observers.restore()
                raise

        return BlockReconstructionResult(
            block=block_name,
            steps=self.config.steps,
            cache_samples=len(cache),
            qparam_groups=tuple(group.name for group in groups),
            initial_loss=initial_loss,
            final_loss=final_loss,
            best_step=best_step,
            qparams=qparams,
            training_loss_history=tuple(training_history),
            evaluation_loss_history=tuple(evaluation_history),
        )

    def evaluate_loss(
        self,
        block: nn.Module,
        cache: ReconstructionCache,
        *,
        device: torch.device | str,
    ) -> float:
        """Evaluate normalized reconstruction loss over the full cache."""
        numerator = torch.zeros((), dtype=torch.float64, device=device)
        denominator = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for invocation, target in cache.sequential_batches(
                self.config.evaluation_batch_size,
                device=device,
                use_quantized_input=True,
            ):
                output = invoke_block(block, invocation)
                batch_numerator, batch_denominator = normalized_mse_terms(
                    output,
                    target,
                )
                numerator += batch_numerator.to(dtype=torch.float64)
                denominator += batch_denominator.to(dtype=torch.float64)
        value = numerator / denominator.clamp_min(self.config.loss_epsilon)
        if not torch.isfinite(value):
            raise FloatingPointError("Block reconstruction evaluation is non-finite.")
        return float(value.detach().cpu().item())


def normalized_mse_loss(
    candidate: TensorTree,
    target: TensorTree,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return energy-normalized squared error over all block-output tensors."""
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive.")
    numerator, denominator = normalized_mse_terms(candidate, target)
    return numerator / denominator.clamp_min(epsilon)


def normalized_mse_terms(
    candidate: TensorTree,
    target: TensorTree,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable squared-error and target-energy sums."""
    pairs = tree_tensor_pairs(candidate, target)
    numerator = sum(
        (candidate_tensor - target_tensor).to(dtype=torch.float32).square().sum()
        for candidate_tensor, target_tensor in pairs
    )
    denominator = sum(
        target_tensor.to(dtype=torch.float32).square().sum()
        for _, target_tensor in pairs
    )
    return numerator, denominator


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
            {
                "params": scale_parameters,
                "lr": config.scale_learning_rate,
            }
        )
    if zero_point_parameters:
        parameter_groups.append(
            {
                "params": zero_point_parameters,
                "lr": config.zero_point_learning_rate,
            }
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
