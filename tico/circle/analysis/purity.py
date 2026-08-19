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

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable

from tico.circle.graph import as_indices, as_list, CircleGraph, OPTIONAL_TENSOR_INDEX


@dataclass(frozen=True, init=False)
class OperatorPurityAnalysis:
    """Classify operators that are safe for common-subexpression elimination.

    The analysis is intentionally conservative. It rejects operators with explicit
    mutation, variable tensors, constant-backed outputs or intermediates, zero
    outputs, subgraph references, or known stateful and non-deterministic behavior.
    Custom operators are rejected unless the caller explicitly opts in because the
    Circle schema does not describe their side effects.
    """

    impure_builtin_codes: frozenset[int]
    custom_builtin_code: int | None
    allow_custom_operators: bool

    def __init__(
        self,
        *,
        impure_builtin_codes: Iterable[int] = (),
        custom_builtin_code: int | None = None,
        allow_custom_operators: bool = False,
    ) -> None:
        """Create a purity analysis from resolved schema enum values."""

        object.__setattr__(
            self,
            "impure_builtin_codes",
            frozenset(int(code) for code in impure_builtin_codes),
        )
        object.__setattr__(
            self,
            "custom_builtin_code",
            None if custom_builtin_code is None else int(custom_builtin_code),
        )
        object.__setattr__(
            self,
            "allow_custom_operators",
            bool(allow_custom_operators),
        )

    def is_pure(
        self,
        graph: CircleGraph,
        operator_index: int,
        *,
        builtin_code: int,
    ) -> bool:
        """Return whether an operator may be evaluated once and reused by value."""

        operators = as_list(getattr(graph.subgraph, "operators", None))
        if operator_index < 0 or operator_index >= len(operators):
            raise IndexError(
                f"Operator index {operator_index} is outside 0..{len(operators) - 1}."
            )
        operator = operators[operator_index]
        builtin_code = int(builtin_code)

        if builtin_code in self.impure_builtin_codes:
            return False
        if (
            self.custom_builtin_code is not None
            and builtin_code == self.custom_builtin_code
            and not self.allow_custom_operators
        ):
            return False
        if (
            int(getattr(operator, "largeCustomOptionsSize", 0) or 0) > 0
            and not self.allow_custom_operators
        ):
            return False
        if any(
            bool(value)
            for value in as_list(getattr(operator, "mutatingVariableInputs", None))
        ):
            return False
        if _operator_references_subgraph(operator):
            return False

        outputs = as_indices(getattr(operator, "outputs", None))
        if not outputs or any(index == OPTIONAL_TENSOR_INDEX for index in outputs):
            return False

        inputs = as_indices(getattr(operator, "inputs", None))
        intermediates = as_indices(getattr(operator, "intermediates", None))
        tensors = as_list(getattr(graph.subgraph, "tensors", None))
        for tensor_index in (*inputs, *outputs, *intermediates):
            if tensor_index == OPTIONAL_TENSOR_INDEX:
                continue
            if tensor_index < 0 or tensor_index >= len(tensors):
                return False
            if bool(getattr(tensors[tensor_index], "isVariable", False)):
                return False

        # Constant folding owns producers with pre-populated storage. Treating such
        # tensors as ordinary runtime expressions could erase storage semantics.
        if any(graph.is_constant(tensor_index) for tensor_index in outputs):
            return False
        if any(
            tensor_index != OPTIONAL_TENSOR_INDEX and graph.is_constant(tensor_index)
            for tensor_index in intermediates
        ):
            return False
        return True


def _operator_references_subgraph(operator: Any) -> bool:
    """Return whether builtin options contain an observable subgraph reference."""

    for options_field in ("builtinOptions", "builtinOptions2"):
        options = getattr(operator, options_field, None)
        if options is None:
            continue
        for field_name in dir(options):
            if field_name.startswith("_"):
                continue
            normalized = field_name.replace("_", "").lower()
            if not (
                normalized.endswith("subgraphindex")
                or normalized.endswith("subgraphindices")
                or normalized == "subgraph"
            ):
                continue
            try:
                value = getattr(options, field_name)
            except Exception:
                continue
            if callable(value):
                continue
            if normalized.endswith("subgraphindices"):
                try:
                    if as_indices(value):
                        return True
                except TypeError:
                    return True
                continue
            if isinstance(value, Integral):
                return True
    return False
