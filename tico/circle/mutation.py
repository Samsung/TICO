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

import dataclasses
from contextvars import ContextVar, Token
from typing import Any, TYPE_CHECKING

from tico.circle._object import clone_object
from tico.circle.graph import as_list

if TYPE_CHECKING:
    from tico.circle.session import CircleOptimizationSession


_ACTIVE_MUTATION: ContextVar[CircleMutationTransaction | None] = ContextVar(
    "tico_circle_active_mutation",
    default=None,
)


class CircleMutationTransaction:
    """Rollback one local rewrite and commit cache invalidation atomically.

    The transaction records sequence boundaries, rewired public fields, and the
    operator/tensor snapshots already present in a rewrite plan. It therefore avoids
    cloning an entire model or all large constant buffers for every local rewrite.
    """

    def __init__(
        self,
        session: CircleOptimizationSession,
        *,
        subgraph_index: int,
        plan: Any = None,
    ) -> None:
        """Capture inexpensive sequence and interface state before mutation."""

        self.session = session
        self.model = session.model
        self.subgraph_index = int(subgraph_index)
        subgraphs = as_list(getattr(self.model, "subgraphs", None))
        if self.subgraph_index < 0 or self.subgraph_index >= len(subgraphs):
            raise IndexError(
                f"Subgraph index {self.subgraph_index} is outside 0.."
                f"{len(subgraphs) - 1}."
            )
        self.subgraph = subgraphs[self.subgraph_index]
        self._initial_buffers = tuple(as_list(getattr(self.model, "buffers", None)))
        self._initial_operator_codes = tuple(
            as_list(getattr(self.model, "operatorCodes", None))
        )
        self._initial_tensors = tuple(as_list(getattr(self.subgraph, "tensors", None)))
        self._initial_operators = tuple(
            as_list(getattr(self.subgraph, "operators", None))
        )
        self._operator_snapshots: dict[int, Any] = {}
        self._operator_code_snapshots: dict[int, Any] = {}
        self._tensor_snapshots: dict[int, Any] = {}
        self._buffer_snapshots: dict[int, Any] = {}
        self._subgraph_fields: dict[str, Any] = {
            "inputs": clone_object(getattr(self.subgraph, "inputs", None)),
            "outputs": clone_object(getattr(self.subgraph, "outputs", None)),
        }
        self._model_fields: dict[str, Any] = {}
        self._touched_tensors: set[int] = set()
        self._committed = False
        self._entered = False
        self._token: Token[CircleMutationTransaction | None] | None = None
        if plan is not None:
            self._capture_plan_objects(plan)

    def __enter__(self) -> CircleMutationTransaction:
        """Make this transaction visible to builders and rewrite utilities."""

        active = _ACTIVE_MUTATION.get()
        if active is not None:
            raise RuntimeError("Nested Circle mutation transactions are not supported.")
        self._token = _ACTIVE_MUTATION.set(self)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """Rollback on exceptions or missing commit, then restore context state."""

        try:
            if exc_type is not None or not self._committed:
                self.rollback()
        finally:
            if self._token is not None:
                _ACTIVE_MUTATION.reset(self._token)
                self._token = None
            self._entered = False
        return False

    @property
    def committed(self) -> bool:
        """Return whether this transaction has committed its mutation."""

        return self._committed

    def watch_operator(self, operator_index: int) -> None:
        """Preserve one existing operator before in-place mutation or deletion."""

        index = int(operator_index)
        if index in self._operator_snapshots:
            return
        if index < 0 or index >= len(self._initial_operators):
            raise IndexError(
                f"Operator index {index} is outside the transaction's original "
                f"range 0..{len(self._initial_operators) - 1}."
            )
        operator = self._initial_operators[index]
        self._operator_snapshots[index] = clone_object(operator)
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        if 0 <= opcode_index < len(self._initial_operator_codes):
            self.watch_operator_code(opcode_index)

    def watch_operator_code(self, opcode_index: int) -> None:
        """Preserve one existing operator-code record before in-place mutation."""

        index = int(opcode_index)
        if index in self._operator_code_snapshots:
            return
        if index < 0 or index >= len(self._initial_operator_codes):
            raise IndexError(
                f"Operator-code index {index} is outside the transaction's original "
                f"range 0..{len(self._initial_operator_codes) - 1}."
            )
        self._operator_code_snapshots[index] = clone_object(
            self._initial_operator_codes[index]
        )

    def watch_tensor(self, tensor_index: int) -> None:
        """Preserve one existing tensor before changing its metadata or storage."""

        index = int(tensor_index)
        if index in self._tensor_snapshots:
            return
        if index < 0 or index >= len(self._initial_tensors):
            raise IndexError(
                f"Tensor index {index} is outside the transaction's original "
                f"range 0..{len(self._initial_tensors) - 1}."
            )
        tensor = self._initial_tensors[index]
        self._tensor_snapshots[index] = clone_object(tensor)
        self._touched_tensors.add(index)

    def watch_buffer(self, buffer_index: int) -> None:
        """Preserve one existing buffer before changing its payload in place."""

        index = int(buffer_index)
        if index in self._buffer_snapshots:
            return
        if index < 0 or index >= len(self._initial_buffers):
            raise IndexError(
                f"Buffer index {index} is outside the transaction's original "
                f"range 0..{len(self._initial_buffers) - 1}."
            )
        self._buffer_snapshots[index] = clone_object(self._initial_buffers[index])

    def touch_tensor(self, tensor_index: int) -> None:
        """Record a tensor whose constant-pool entry must be refreshed at commit."""

        self._touched_tensors.add(int(tensor_index))

    def watch_subgraph_field(self, field_name: str) -> None:
        """Preserve one subgraph field before changing its nested value."""

        name = str(field_name)
        self._subgraph_fields.setdefault(
            name,
            clone_object(getattr(self.subgraph, name, None)),
        )

    def watch_model_field(self, field_name: str) -> None:
        """Preserve one model-global field before changing nested references."""

        name = str(field_name)
        self._model_fields.setdefault(
            name,
            clone_object(getattr(self.model, name, None)),
        )

    def commit(self) -> None:
        """Publish one successful rewrite and invalidate dependent analyses."""

        if not self._entered:
            raise RuntimeError("Circle mutation transaction is not active.")
        if self._committed:
            raise RuntimeError("Circle mutation transaction was already committed.")
        touched = tuple(sorted(self._touched_tensors))
        self.session.mark_modified(
            (self.subgraph_index,),
            touched_tensors=({self.subgraph_index: touched} if touched else None),
        )
        self._committed = True

    def rollback(self) -> None:
        """Restore sequence order, watched values, interfaces, and shared indexes."""

        restored_operators = list(self._initial_operators)
        for index, operator in self._operator_snapshots.items():
            restored_operators[index] = operator
        restored_tensors = list(self._initial_tensors)
        for index, tensor in self._tensor_snapshots.items():
            restored_tensors[index] = tensor

        restored_buffers = list(self._initial_buffers)
        for index, buffer in self._buffer_snapshots.items():
            restored_buffers[index] = buffer

        restored_operator_codes = list(self._initial_operator_codes)
        for index, operator_code in self._operator_code_snapshots.items():
            restored_operator_codes[index] = operator_code

        self.model.buffers = restored_buffers
        self.model.operatorCodes = restored_operator_codes
        self.subgraph.tensors = restored_tensors
        self.subgraph.operators = restored_operators
        for field_name, value in self._subgraph_fields.items():
            setattr(self.subgraph, field_name, clone_object(value))
        for field_name, value in self._model_fields.items():
            setattr(self.model, field_name, clone_object(value))

        self.session.invalidate(
            (self.subgraph_index,),
            rebuild_constant_pools=True,
        )

    def _capture_plan_objects(self, plan: Any) -> None:
        """Discover operator and tensor snapshots recursively from one rewrite plan."""

        pending = [plan]
        visited: set[int] = set()
        while pending:
            value = pending.pop()
            if value is None or isinstance(
                value,
                (bool, int, float, str, bytes, bytearray, memoryview),
            ):
                continue
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)

            if _is_operator_snapshot(value):
                self.watch_operator(int(value.operator_index))
                continue
            if _is_tensor_snapshot(value):
                self.watch_tensor(int(value.tensor_index))
                continue
            if isinstance(value, dict):
                pending.extend(value.keys())
                pending.extend(value.values())
                continue
            if isinstance(value, (tuple, list, set, frozenset)):
                pending.extend(value)
                continue
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                pending.extend(
                    getattr(value, field.name) for field in dataclasses.fields(value)
                )


def current_mutation(
    *,
    model: Any | None = None,
    subgraph_index: int | None = None,
) -> CircleMutationTransaction | None:
    """Return the active transaction when model and subgraph constraints agree."""

    transaction = _ACTIVE_MUTATION.get()
    if transaction is None:
        return None
    if model is not None and transaction.model is not model:
        return None
    if subgraph_index is not None and transaction.subgraph_index != int(subgraph_index):
        return None
    return transaction


def _is_operator_snapshot(value: Any) -> bool:
    return (
        type(value).__name__ == "OperatorSnapshot"
        and hasattr(value, "operator_index")
        and hasattr(value, "operator_fingerprint")
    )


def _is_tensor_snapshot(value: Any) -> bool:
    return (
        type(value).__name__ == "TensorSnapshot"
        and hasattr(value, "tensor_index")
        and hasattr(value, "tensor_fingerprint")
    )


__all__ = ["CircleMutationTransaction", "current_mutation"]
