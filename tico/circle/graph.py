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

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from tico.circle._schema import decode_text
from tico.circle.errors import CircleRewriteError, CircleSelectionError

OPTIONAL_TENSOR_INDEX = -1


def as_list(value: Any) -> list[Any]:
    """Convert a generated vector field to a plain Python list."""

    if value is None:
        return []
    return list(value)


def as_indices(value: Any) -> list[int]:
    """Convert a generated index vector to a plain list of integers."""

    return [int(index) for index in as_list(value)]


def has_buffer_payload(buffer: Any) -> bool:
    """Return whether a Circle buffer carries inline or external data."""

    if buffer is None:
        return False
    data = getattr(buffer, "data", None)
    if data is not None:
        try:
            if len(data) > 0:
                return True
        except TypeError:
            return True
    return bool(getattr(buffer, "size", 0) or getattr(buffer, "offset", 0))


def is_constant_tensor(model: Any, subgraph: Any, tensor_index: int) -> bool:
    """Return whether a tensor is backed by a non-empty model buffer."""

    tensors = as_list(subgraph.tensors)
    if tensor_index < 0 or tensor_index >= len(tensors):
        return False
    tensor = tensors[tensor_index]
    if bool(getattr(tensor, "isVariable", False)):
        return False
    buffer_index = int(getattr(tensor, "buffer", 0) or 0)
    buffers = as_list(model.buffers)
    if buffer_index <= 0 or buffer_index >= len(buffers):
        return False
    return has_buffer_payload(buffers[buffer_index])


@dataclass(frozen=True)
class GraphBoundary:
    """Describe the inputs and outputs induced by an operator region."""

    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


class CircleGraph:
    """Index producer, consumer, and traversal relationships for one subgraph."""

    def __init__(self, model: Any, subgraph_index: int = 0):
        self.model = model
        self.subgraph_index = subgraph_index
        subgraphs = as_list(model.subgraphs)
        if subgraph_index < 0 or subgraph_index >= len(subgraphs):
            raise IndexError(
                f"Subgraph index {subgraph_index} is outside the valid range "
                f"0..{len(subgraphs) - 1}."
            )
        self.subgraph = subgraphs[subgraph_index]
        self._producer: dict[int, int] = {}
        self._consumers: dict[int, list[int]] = {}
        self._build_index()

    @property
    def tensor_count(self) -> int:
        """Return the number of tensors in the subgraph."""

        return len(as_list(self.subgraph.tensors))

    @property
    def operator_count(self) -> int:
        """Return the number of operators in the subgraph."""

        return len(as_list(self.subgraph.operators))

    @property
    def inputs(self) -> tuple[int, ...]:
        """Return subgraph input tensor indices."""

        return tuple(as_indices(self.subgraph.inputs))

    @property
    def outputs(self) -> tuple[int, ...]:
        """Return subgraph output tensor indices."""

        return tuple(as_indices(self.subgraph.outputs))

    def _build_index(self) -> None:
        operators = as_list(self.subgraph.operators)
        for operator_index, operator in enumerate(operators):
            for tensor_index in as_indices(getattr(operator, "outputs", None)):
                if tensor_index == OPTIONAL_TENSOR_INDEX:
                    continue
                previous = self._producer.get(tensor_index)
                if previous is not None:
                    raise CircleRewriteError(
                        f"Tensor {tensor_index} is produced by operators {previous} "
                        f"and {operator_index} in subgraph {self.subgraph_index}."
                    )
                self._producer[tensor_index] = operator_index
            for tensor_index in as_indices(getattr(operator, "inputs", None)):
                if tensor_index == OPTIONAL_TENSOR_INDEX:
                    continue
                self._consumers.setdefault(tensor_index, []).append(operator_index)

    def producer(self, tensor_index: int) -> int | None:
        """Return the operator that produces a tensor, if one exists."""

        return self._producer.get(tensor_index)

    def consumers(self, tensor_index: int) -> tuple[int, ...]:
        """Return operators that consume a tensor in graph order."""

        return tuple(self._consumers.get(tensor_index, ()))

    def operator_inputs(self, operator_index: int) -> tuple[int, ...]:
        """Return input tensor indices for an operator."""

        operator = self._operator(operator_index)
        return tuple(as_indices(getattr(operator, "inputs", None)))

    def operator_outputs(self, operator_index: int) -> tuple[int, ...]:
        """Return output tensor indices for an operator."""

        operator = self._operator(operator_index)
        return tuple(as_indices(getattr(operator, "outputs", None)))

    def _operator(self, operator_index: int) -> Any:
        operators = as_list(self.subgraph.operators)
        if operator_index < 0 or operator_index >= len(operators):
            raise IndexError(
                f"Operator index {operator_index} is outside the valid range "
                f"0..{len(operators) - 1}."
            )
        return operators[operator_index]

    def tensor_name(self, tensor_index: int) -> str:
        """Return a decoded tensor name."""

        tensors = as_list(self.subgraph.tensors)
        if tensor_index < 0 or tensor_index >= len(tensors):
            raise IndexError(
                f"Tensor index {tensor_index} is outside the valid range "
                f"0..{len(tensors) - 1}."
            )
        return decode_text(getattr(tensors[tensor_index], "name", ""))

    def is_constant(self, tensor_index: int) -> bool:
        """Return whether a tensor has a constant buffer payload."""

        return is_constant_tensor(self.model, self.subgraph, tensor_index)

    def predecessors(self, operator_index: int) -> tuple[int, ...]:
        """Return operators that directly feed an operator."""

        result: list[int] = []
        seen: set[int] = set()
        for tensor_index in self.operator_inputs(operator_index):
            if tensor_index == OPTIONAL_TENSOR_INDEX:
                continue
            producer = self.producer(tensor_index)
            if producer is not None and producer not in seen:
                seen.add(producer)
                result.append(producer)
        return tuple(result)

    def successors(self, operator_index: int) -> tuple[int, ...]:
        """Return operators that directly consume an operator output."""

        result: list[int] = []
        seen: set[int] = set()
        for tensor_index in self.operator_outputs(operator_index):
            if tensor_index == OPTIONAL_TENSOR_INDEX:
                continue
            for consumer in self.consumers(tensor_index):
                if consumer not in seen:
                    seen.add(consumer)
                    result.append(consumer)
        return tuple(result)

    def forward_operators(self, tensor_indices: Iterable[int]) -> set[int]:
        """Return all operators reachable forward from a set of tensors."""

        queue = deque(int(index) for index in tensor_indices)
        visited_tensors: set[int] = set(queue)
        visited_operators: set[int] = set()
        while queue:
            tensor_index = queue.popleft()
            for operator_index in self.consumers(tensor_index):
                if operator_index in visited_operators:
                    continue
                visited_operators.add(operator_index)
                for output_index in self.operator_outputs(operator_index):
                    if output_index == OPTIONAL_TENSOR_INDEX:
                        continue
                    if output_index not in visited_tensors:
                        visited_tensors.add(output_index)
                        queue.append(output_index)
        return visited_operators

    def backward_operators(self, tensor_indices: Iterable[int]) -> set[int]:
        """Return all operators needed to produce a set of tensors."""

        queue = deque(int(index) for index in tensor_indices)
        visited_tensors: set[int] = set(queue)
        visited_operators: set[int] = set()
        while queue:
            tensor_index = queue.popleft()
            producer = self.producer(tensor_index)
            if producer is None or producer in visited_operators:
                continue
            visited_operators.add(producer)
            for input_index in self.operator_inputs(producer):
                if input_index == OPTIONAL_TENSOR_INDEX:
                    continue
                if input_index not in visited_tensors:
                    visited_tensors.add(input_index)
                    queue.append(input_index)
        return visited_operators

    def region_boundary(self, operator_indices: Iterable[int]) -> GraphBoundary:
        """Compute graph inputs and outputs for an operator-induced region."""

        selected = {int(index) for index in operator_indices}
        if not selected:
            raise CircleSelectionError("At least one operator must be selected.")
        invalid = sorted(
            index for index in selected if index < 0 or index >= self.operator_count
        )
        if invalid:
            raise CircleSelectionError(
                f"Operator indices {invalid} are outside subgraph "
                f"{self.subgraph_index}."
            )

        input_candidates: list[int] = []
        output_candidates: list[int] = []
        original_outputs = set(self.outputs)

        for operator_index in sorted(selected):
            for tensor_index in self.operator_inputs(operator_index):
                if tensor_index == OPTIONAL_TENSOR_INDEX:
                    continue
                producer = self.producer(tensor_index)
                if producer not in selected and not self.is_constant(tensor_index):
                    input_candidates.append(tensor_index)

            for tensor_index in self.operator_outputs(operator_index):
                if tensor_index == OPTIONAL_TENSOR_INDEX:
                    continue
                consumers = self.consumers(tensor_index)
                has_external_consumer = any(
                    consumer not in selected for consumer in consumers
                )
                is_terminal = not any(consumer in selected for consumer in consumers)
                if (
                    tensor_index in original_outputs
                    or has_external_consumer
                    or is_terminal
                ):
                    output_candidates.append(tensor_index)

        ordered_inputs = self._ordered_unique_inputs(input_candidates)
        ordered_outputs = self._ordered_unique(output_candidates)
        return GraphBoundary(tuple(ordered_inputs), tuple(ordered_outputs))

    def _ordered_unique_inputs(self, candidates: Sequence[int]) -> list[int]:
        candidate_set = set(candidates)
        result = [index for index in self.inputs if index in candidate_set]
        result.extend(index for index in candidates if index not in result)
        return self._ordered_unique(result)

    @staticmethod
    def _ordered_unique(indices: Iterable[int]) -> list[int]:
        result: list[int] = []
        seen: set[int] = set()
        for index in indices:
            if index in seen:
                continue
            seen.add(index)
            result.append(index)
        return result

    def iter_tensor_names(self) -> Iterator[tuple[int, str]]:
        """Yield tensor indices and decoded names in tensor order."""

        for tensor_index in range(self.tensor_count):
            yield tensor_index, self.tensor_name(tensor_index)
