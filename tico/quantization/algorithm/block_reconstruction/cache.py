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

"""Cached block inputs and targets used by reconstruction algorithms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch


TensorTree: TypeAlias = (
    torch.Tensor
    | tuple["TensorTree", ...]
    | list["TensorTree"]
    | Mapping[str | int, "TensorTree"]
)


@dataclass(frozen=True)
class BlockInvocation:
    """Store positional and keyword arguments for one block invocation."""

    args: tuple[TensorTree, ...] = ()
    kwargs: Mapping[str, TensorTree] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructionSample:
    """Store float inputs, quantized-prefix inputs, and float block targets."""

    float_input: BlockInvocation
    quantized_input: BlockInvocation
    target: TensorTree

    def detached_cpu(self) -> "ReconstructionSample":
        """Return a detached CPU copy suitable for a bounded cache."""
        return ReconstructionSample(
            float_input=_invocation_map(self.float_input, _detach_cpu),
            quantized_input=_invocation_map(self.quantized_input, _detach_cpu),
            target=tree_map(self.target, _detach_cpu),
        )


class ReconstructionCache:
    """Own deterministic per-sample block reconstruction data."""

    def __init__(self, samples: Sequence[ReconstructionSample]) -> None:
        if not samples:
            raise ValueError(
                "Block reconstruction requires at least one cached sample."
            )
        self._samples = tuple(sample.detached_cpu() for sample in samples)
        _validate_sample_structures(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> tuple[ReconstructionSample, ...]:
        """Return immutable cached samples for diagnostics."""
        return self._samples

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device | str,
        use_quantized_input: bool = True,
    ) -> tuple[BlockInvocation, TensorTree]:
        """Batch selected samples and move them to one optimization device."""
        if not indices:
            raise ValueError("A reconstruction batch requires at least one index.")
        selected = tuple(self._samples[index] for index in indices)
        invocations = tuple(
            sample.quantized_input if use_quantized_input else sample.float_input
            for sample in selected
        )
        invocation = _batch_invocations(invocations, device=device)
        target = _batch_trees(
            tuple(sample.target for sample in selected),
            device=device,
        )
        return invocation, target

    def random_batch(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device | str,
        use_quantized_input: bool = True,
    ) -> tuple[BlockInvocation, TensorTree]:
        """Draw one deterministic-with-generator minibatch with replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        indices = torch.randint(
            len(self),
            (batch_size,),
            generator=generator,
        ).tolist()
        return self.batch(
            indices,
            device=device,
            use_quantized_input=use_quantized_input,
        )

    def sequential_batches(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        use_quantized_input: bool = True,
    ):
        """Yield deterministic non-overlapping batches over the entire cache."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(self), batch_size):
            indices = tuple(range(start, min(start + batch_size, len(self))))
            yield self.batch(
                indices,
                device=device,
                use_quantized_input=use_quantized_input,
            )


def invoke_block(module: torch.nn.Module, invocation: BlockInvocation) -> Any:
    """Invoke a reconstruction block from cached positional and keyword inputs."""
    return module(*invocation.args, **dict(invocation.kwargs))


def tree_map(tree: TensorTree, function) -> TensorTree:
    """Apply a tensor function while preserving nested container structure."""
    if isinstance(tree, torch.Tensor):
        return function(tree)
    if isinstance(tree, tuple):
        return tuple(tree_map(value, function) for value in tree)
    if isinstance(tree, list):
        return [tree_map(value, function) for value in tree]
    if isinstance(tree, Mapping):
        return {key: tree_map(value, function) for key, value in tree.items()}
    raise TypeError(f"Unsupported tensor-tree value: {type(tree).__name__}.")


def tree_tensor_pairs(
    candidate: TensorTree,
    target: TensorTree,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Flatten two matching tensor trees into aligned tensor pairs."""
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    _append_tensor_pairs(candidate, target, pairs, path="root")
    if not pairs:
        raise ValueError("A reconstruction target must contain at least one tensor.")
    return tuple(pairs)


def _append_tensor_pairs(
    candidate: TensorTree,
    target: TensorTree,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    path: str,
) -> None:
    if isinstance(candidate, torch.Tensor) and isinstance(target, torch.Tensor):
        if candidate.shape != target.shape:
            raise ValueError(
                f"Tensor shape mismatch at {path}: "
                f"{tuple(candidate.shape)} != {tuple(target.shape)}."
            )
        pairs.append((candidate, target))
        return
    if isinstance(candidate, tuple) and isinstance(target, tuple):
        if len(candidate) != len(target):
            raise ValueError(f"Tuple length mismatch at {path}.")
        for index, (candidate_value, target_value) in enumerate(zip(candidate, target)):
            _append_tensor_pairs(
                candidate_value,
                target_value,
                pairs,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(candidate, list) and isinstance(target, list):
        if len(candidate) != len(target):
            raise ValueError(f"List length mismatch at {path}.")
        for index, (candidate_value, target_value) in enumerate(zip(candidate, target)):
            _append_tensor_pairs(
                candidate_value,
                target_value,
                pairs,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(candidate, Mapping) and isinstance(target, Mapping):
        if tuple(candidate) != tuple(target):
            raise ValueError(f"Mapping key mismatch at {path}.")
        for key in candidate:
            _append_tensor_pairs(
                candidate[key],
                target[key],
                pairs,
                path=f"{path}[{key!r}]",
            )
        return
    raise TypeError(
        f"Tensor-tree type mismatch at {path}: "
        f"{type(candidate).__name__} != {type(target).__name__}."
    )


def _validate_sample_structures(samples: Sequence[ReconstructionSample]) -> None:
    reference = samples[0]
    for index, sample in enumerate(samples[1:], start=1):
        _assert_same_structure(
            reference.float_input.args,
            sample.float_input.args,
            f"sample[{index}].float_input.args",
        )
        _assert_same_structure(
            reference.float_input.kwargs,
            sample.float_input.kwargs,
            f"sample[{index}].float_input.kwargs",
        )
        _assert_same_structure(
            reference.quantized_input.args,
            sample.quantized_input.args,
            f"sample[{index}].quantized_input.args",
        )
        _assert_same_structure(
            reference.quantized_input.kwargs,
            sample.quantized_input.kwargs,
            f"sample[{index}].quantized_input.kwargs",
        )
        _assert_same_structure(
            reference.target,
            sample.target,
            f"sample[{index}].target",
        )


def _assert_same_structure(reference: Any, value: Any, path: str) -> None:
    if isinstance(reference, torch.Tensor) and isinstance(value, torch.Tensor):
        if reference.shape[1:] != value.shape[1:]:
            raise ValueError(
                f"Non-batch tensor shape mismatch at {path}: "
                f"{tuple(reference.shape[1:])} != {tuple(value.shape[1:])}."
            )
        return
    if isinstance(reference, tuple) and isinstance(value, tuple):
        if len(reference) != len(value):
            raise ValueError(f"Tuple length mismatch at {path}.")
        for index, (left, right) in enumerate(zip(reference, value)):
            _assert_same_structure(left, right, f"{path}[{index}]")
        return
    if isinstance(reference, list) and isinstance(value, list):
        if len(reference) != len(value):
            raise ValueError(f"List length mismatch at {path}.")
        for index, (left, right) in enumerate(zip(reference, value)):
            _assert_same_structure(left, right, f"{path}[{index}]")
        return
    if isinstance(reference, Mapping) and isinstance(value, Mapping):
        if tuple(reference) != tuple(value):
            raise ValueError(f"Mapping key mismatch at {path}.")
        for key in reference:
            _assert_same_structure(reference[key], value[key], f"{path}[{key!r}]")
        return
    raise TypeError(
        f"Tensor-tree type mismatch at {path}: "
        f"{type(reference).__name__} != {type(value).__name__}."
    )


def _batch_invocations(
    values: Sequence[BlockInvocation],
    *,
    device: torch.device | str,
) -> BlockInvocation:
    first = values[0]
    args = tuple(
        _batch_trees(tuple(value.args[index] for value in values), device=device)
        for index in range(len(first.args))
    )
    kwargs = {
        key: _batch_trees(tuple(value.kwargs[key] for value in values), device=device)
        for key in first.kwargs
    }
    return BlockInvocation(args=args, kwargs=kwargs)


def _batch_trees(
    values: Sequence[TensorTree],
    *,
    device: torch.device | str,
) -> TensorTree:
    first = values[0]
    if isinstance(first, torch.Tensor):
        tensors = tuple(value for value in values if isinstance(value, torch.Tensor))
        if len(tensors) != len(values):
            raise TypeError("Cannot batch tensor and non-tensor values together.")
        moved = tuple(value.to(device=device) for value in tensors)
        if first.ndim == 0:
            return torch.stack(moved, dim=0)
        return torch.cat(moved, dim=0)
    if isinstance(first, tuple):
        return tuple(
            _batch_trees(
                tuple(value[index] for value in values),  # type: ignore[index]
                device=device,
            )
            for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            _batch_trees(
                tuple(value[index] for value in values),  # type: ignore[index]
                device=device,
            )
            for index in range(len(first))
        ]
    if isinstance(first, Mapping):
        return {
            key: _batch_trees(
                tuple(value[key] for value in values),  # type: ignore[index]
                device=device,
            )
            for key in first
        }
    raise TypeError(f"Unsupported tensor-tree value: {type(first).__name__}.")


def _invocation_map(invocation: BlockInvocation, function) -> BlockInvocation:
    return BlockInvocation(
        args=tuple(tree_map(value, function) for value in invocation.args),
        kwargs={
            key: tree_map(value, function) for key, value in invocation.kwargs.items()
        },
    )


def _detach_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()
