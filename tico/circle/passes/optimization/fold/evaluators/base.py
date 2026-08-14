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

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Iterable

from tico.circle.analysis import TensorContract
from tico.circle.document import CircleDocument
from tico.circle.graph import CircleGraph
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True)
class ConstantEvaluation:
    """Return all logical output values produced by one constant evaluation."""

    outputs: tuple[TensorValue, ...]

    def __post_init__(self) -> None:
        """Normalize the output sequence to an immutable tuple."""

        outputs = tuple(self.outputs)
        if any(not isinstance(output, TensorValue) for output in outputs):
            raise TypeError("ConstantEvaluation outputs must be TensorValue objects.")
        object.__setattr__(self, "outputs", outputs)


@dataclass(frozen=True)
class ConstantEvaluationContext:
    """Provide immutable operator metadata and decoded constant inputs to evaluators."""

    document: CircleDocument
    graph: CircleGraph
    operator_index: int
    operator: Any
    operator_code: Any
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    input_contracts: tuple[TensorContract | None, ...]
    output_contracts: tuple[TensorContract, ...]
    input_values: tuple[TensorValue | None, ...]
    codec: TensorValueCodec

    def __post_init__(self) -> None:
        """Validate positional metadata used by evaluator implementations."""

        if self.operator_index < 0:
            raise ValueError("operator_index must not be negative.")
        if len(self.input_indices) != len(self.input_contracts):
            raise ValueError(
                "input_indices and input_contracts must have equal length."
            )
        if len(self.input_indices) != len(self.input_values):
            raise ValueError("input_indices and input_values must have equal length.")
        if len(self.output_indices) != len(self.output_contracts):
            raise ValueError(
                "output_indices and output_contracts must have equal length."
            )

    @property
    def builtin_code(self) -> int:
        """Return the serialized builtin operator code."""

        return int(getattr(self.operator_code, "builtinCode", -1))

    @property
    def options(self) -> Any:
        """Return the primary Object API builtin-options table when present."""

        return getattr(self.operator, "builtinOptions", None)

    def input_contract(self, position: int) -> TensorContract:
        """Return one non-optional input contract by position."""

        contract = self.input_contracts[position]
        if contract is None:
            raise ValueError(f"Input position {position} is optional and absent.")
        return contract

    def input_value(self, position: int) -> TensorValue:
        """Return one decoded constant input value by position."""

        value = self.input_values[position]
        if value is None:
            raise ValueError(
                f"Input position {position} was not decoded as a constant."
            )
        return value

    def output_contract(self, position: int = 0) -> TensorContract:
        """Return one output contract by position."""

        return self.output_contracts[position]

    def with_input_values(
        self,
        values: Iterable[TensorValue | None],
    ) -> ConstantEvaluationContext:
        """Return a copy carrying decoded input values."""

        return replace(self, input_values=tuple(values))


class ConstantEvaluator(ABC):
    """Evaluate one supported Circle builtin without mutating the source graph."""

    @property
    def name(self) -> str:
        """Return the stable evaluator name used in diagnostics."""

        return self.__class__.__name__

    @abstractmethod
    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Return input positions that must be backed by inline constants."""

    def estimate_compute_cost(self, context: ConstantEvaluationContext) -> int:
        """Estimate logical element operations before decoding input payloads."""

        return sum(contract.element_count for contract in context.output_contracts)

    @abstractmethod
    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Return exact output values or None when the candidate is unsupported."""


class ConstantEvaluatorRegistry:
    """Map builtin operator codes to reusable constant evaluator implementations."""

    def __init__(
        self,
        entries: Iterable[tuple[int, ConstantEvaluator]] = (),
    ) -> None:
        """Create a registry and reject duplicate builtin operator codes."""

        self._evaluators: dict[int, ConstantEvaluator] = {}
        for builtin_code, evaluator in entries:
            self.register(builtin_code, evaluator)

    def register(self, builtin_code: int, evaluator: ConstantEvaluator) -> None:
        """Register one evaluator for a builtin operator code."""

        code = int(builtin_code)
        if code < 0:
            raise ValueError("builtin_code must not be negative.")
        if not isinstance(evaluator, ConstantEvaluator):
            raise TypeError("evaluator must implement ConstantEvaluator.")
        if code in self._evaluators:
            raise ValueError(
                f"A constant evaluator is already registered for builtin code {code}."
            )
        self._evaluators[code] = evaluator

    def get(self, builtin_code: int) -> ConstantEvaluator | None:
        """Return the evaluator registered for a builtin code, if any."""

        return self._evaluators.get(int(builtin_code))

    @property
    def entries(self) -> tuple[tuple[int, ConstantEvaluator], ...]:
        """Return registered entries ordered by builtin operator code."""

        return tuple(
            (code, self._evaluators[code]) for code in sorted(self._evaluators)
        )

    def copy(self) -> ConstantEvaluatorRegistry:
        """Return an independently mutable registry with the same evaluator objects."""

        return ConstantEvaluatorRegistry(self.entries)


def contract_is_fully_static(contract: TensorContract) -> bool:
    """Return whether a tensor contract contains no dynamic signature dimensions."""

    signature = contract.shape_signature
    return signature is None or all(dimension >= 0 for dimension in signature)


def contract_is_dense_value(contract: TensorContract) -> bool:
    """Return whether a contract can be represented by the dense value codec."""

    return (
        not contract.is_variable
        and contract.sparsity is None
        and contract.variant_tensors is None
    )
