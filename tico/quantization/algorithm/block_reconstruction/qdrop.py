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

"""Element-wise QDrop for activation-qparam block reconstruction."""

from __future__ import annotations

import math

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import torch

from tico.quantization.algorithm.block_reconstruction.cache import (
    BlockInvocation,
    TensorTree,
)


_ACTIVE_QDROP: ContextVar[QDropController | None] = ContextVar(
    "block_reconstruction_qdrop",
    default=None,
)


@dataclass(frozen=True)
class QDropStatistics:
    """Summarize configured QDrop coverage without synchronizing random masks."""

    probability: float
    seed: int
    input_tensor_count: int
    input_element_count: int
    activation_tensor_count: int
    activation_element_count: int

    @property
    def total_element_count(self) -> int:
        """Return all floating-point elements exposed to QDrop."""
        return self.input_element_count + self.activation_element_count

    @property
    def expected_dropped_element_count(self) -> float:
        """Return the expectation implied by the configured Bernoulli rate."""
        return self.probability * self.total_element_count

    def to_dict(self) -> dict[str, float | int | str]:
        """Return JSON-compatible QDrop diagnostics."""
        return {
            "probability": self.probability,
            "seed": self.seed,
            "granularity": "element",
            "input_tensor_count": self.input_tensor_count,
            "input_element_count": self.input_element_count,
            "activation_tensor_count": self.activation_tensor_count,
            "activation_element_count": self.activation_element_count,
            "total_element_count": self.total_element_count,
            "expected_dropped_element_count": (self.expected_dropped_element_count),
        }


class QDropController:
    """Generate deterministic element-wise masks for one reconstruction run.

    ``probability`` is the probability of dropping quantization, meaning the
    floating-point value is used. A value of zero preserves the existing fully
    quantized reconstruction path, while one uses floating-point activations
    throughout the training forward pass.
    """

    def __init__(self, probability: float, *, seed: int) -> None:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("QDrop probability must be finite and in [0, 1].")
        if not isinstance(seed, int):
            raise TypeError("QDrop seed must be an integer.")
        self.probability = float(probability)
        self.seed = seed
        self._generators: dict[str, torch.Generator] = {}
        self._input_tensor_count = 0
        self._input_element_count = 0
        self._activation_tensor_count = 0
        self._activation_element_count = 0

    @property
    def enabled(self) -> bool:
        """Return whether any activation quantization may be dropped."""
        return self.probability > 0.0

    def mix_invocations(
        self,
        float_invocation: BlockInvocation,
        quantized_invocation: BlockInvocation,
    ) -> BlockInvocation:
        """Mix matching float and quantized-prefix block inputs."""
        if len(float_invocation.args) != len(quantized_invocation.args):
            raise ValueError("QDrop block-input positional argument counts differ.")
        if tuple(float_invocation.kwargs) != tuple(quantized_invocation.kwargs):
            raise ValueError("QDrop block-input keyword argument keys differ.")
        return BlockInvocation(
            args=tuple(
                self._mix_tree(float_value, quantized_value, source="input")
                for float_value, quantized_value in zip(
                    float_invocation.args,
                    quantized_invocation.args,
                )
            ),
            kwargs={
                key: self._mix_tree(
                    float_invocation.kwargs[key],
                    quantized_invocation.kwargs[key],
                    source="input",
                )
                for key in float_invocation.kwargs
            },
        )

    def mix_activation(
        self,
        float_value: torch.Tensor,
        quantized_value: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly bypass one internal activation quantizer element-wise."""
        return self._mix_tensor(
            float_value,
            quantized_value,
            source="activation",
        )

    def statistics(self) -> QDropStatistics:
        """Return lightweight coverage counters for report metadata."""
        return QDropStatistics(
            probability=self.probability,
            seed=self.seed,
            input_tensor_count=self._input_tensor_count,
            input_element_count=self._input_element_count,
            activation_tensor_count=self._activation_tensor_count,
            activation_element_count=self._activation_element_count,
        )

    def _mix_tree(
        self,
        float_tree: TensorTree,
        quantized_tree: TensorTree,
        *,
        source: str,
    ) -> TensorTree:
        if isinstance(float_tree, torch.Tensor) and isinstance(
            quantized_tree,
            torch.Tensor,
        ):
            return self._mix_tensor(float_tree, quantized_tree, source=source)
        if isinstance(float_tree, tuple) and isinstance(quantized_tree, tuple):
            if len(float_tree) != len(quantized_tree):
                raise ValueError("QDrop tuple lengths differ.")
            return tuple(
                self._mix_tree(left, right, source=source)
                for left, right in zip(float_tree, quantized_tree)
            )
        if isinstance(float_tree, list) and isinstance(quantized_tree, list):
            if len(float_tree) != len(quantized_tree):
                raise ValueError("QDrop list lengths differ.")
            return [
                self._mix_tree(left, right, source=source)
                for left, right in zip(float_tree, quantized_tree)
            ]
        if isinstance(float_tree, Mapping) and isinstance(
            quantized_tree,
            Mapping,
        ):
            if tuple(float_tree) != tuple(quantized_tree):
                raise ValueError("QDrop mapping keys differ.")
            return {
                key: self._mix_tree(
                    float_tree[key],
                    quantized_tree[key],
                    source=source,
                )
                for key in float_tree
            }
        raise TypeError(
            "QDrop tensor-tree types differ: "
            f"{type(float_tree).__name__} != "
            f"{type(quantized_tree).__name__}."
        )

    def _mix_tensor(
        self,
        float_value: torch.Tensor,
        quantized_value: torch.Tensor,
        *,
        source: str,
    ) -> torch.Tensor:
        if float_value.shape != quantized_value.shape:
            raise ValueError(
                "QDrop tensor shapes differ: "
                f"{tuple(float_value.shape)} != {tuple(quantized_value.shape)}."
            )
        if float_value.device != quantized_value.device:
            raise ValueError("QDrop tensors must be on the same device.")
        if float_value.dtype != quantized_value.dtype:
            raise ValueError("QDrop tensors must have the same dtype.")
        if not quantized_value.is_floating_point():
            if not torch.equal(float_value, quantized_value):
                raise ValueError(
                    "Non-floating QDrop inputs must be identical in float and "
                    "quantized-prefix caches."
                )
            return quantized_value

        self._record(source, quantized_value.numel())
        if self.probability == 0.0:
            return quantized_value
        if self.probability == 1.0:
            # Keep a zero-gradient edge to the quantized branch so a full-drop
            # training step still has a valid autograd graph for qparams.
            if source == "activation" and quantized_value.requires_grad:
                return float_value + quantized_value * 0.0
            return float_value
        generator = self._generator_for(quantized_value.device)
        mask = (
            torch.rand(
                quantized_value.shape,
                device=quantized_value.device,
                generator=generator,
                dtype=torch.float32,
            )
            < self.probability
        )
        return torch.where(mask, float_value, quantized_value)

    def _record(self, source: str, element_count: int) -> None:
        if source == "input":
            self._input_tensor_count += 1
            self._input_element_count += element_count
            return
        if source == "activation":
            self._activation_tensor_count += 1
            self._activation_element_count += element_count
            return
        raise ValueError(f"Unknown QDrop source: {source!r}.")

    def _generator_for(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[key] = generator
        return generator


@contextmanager
def qdrop_context(controller: QDropController | None) -> Iterator[None]:
    """Enable QDrop only within one reconstruction training forward pass."""
    token = _ACTIVE_QDROP.set(
        controller if controller is not None and controller.enabled else None
    )
    try:
        yield
    finally:
        _ACTIVE_QDROP.reset(token)


def maybe_qdrop_activation(
    float_value: torch.Tensor,
    quantized_value: torch.Tensor,
) -> torch.Tensor:
    """Apply the active QDrop mask or return fully quantized activation."""
    controller = _ACTIVE_QDROP.get()
    if controller is None:
        return quantized_value
    return controller.mix_activation(float_value, quantized_value)
