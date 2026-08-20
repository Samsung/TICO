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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from tico.circle._schema import circle_schema, object_api_type
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list, CircleGraph, OPTIONAL_TENSOR_INDEX


DEFAULT_EFFECTFUL_BUILTIN_NAMES = (
    "ASSIGN_VARIABLE",
    "BIDIRECTIONAL_SEQUENCE_LSTM",
    "BIDIRECTIONAL_SEQUENCE_RNN",
    "CALL",
    "CALL_ONCE",
    "DELEGATE",
    "HASHTABLE",
    "HASHTABLE_FIND",
    "HASHTABLE_IMPORT",
    "HASHTABLE_LOOKUP",
    "HASHTABLE_SIZE",
    "IF",
    "LSTM",
    "MULTINOMIAL",
    "RANDOM_STANDARD_NORMAL",
    "RANDOM_UNIFORM",
    "READ_VARIABLE",
    "RESOURCE_GATHER",
    "RESOURCE_SCATTER_ADD",
    "RNN",
    "STABLEHLO_CASE",
    "STABLEHLO_CUSTOM_CALL",
    "STABLEHLO_RNG_BIT_GENERATOR",
    "STABLEHLO_WHILE",
    "SVDF",
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "UNIDIRECTIONAL_SEQUENCE_RNN",
    "VAR_HANDLE",
    "WHILE",
)


@dataclass(frozen=True, init=False)
class OperatorEffectAnalysis:
    """Classify operators that must remain observable during dead-code elimination.

    The analysis is intentionally conservative. In addition to a configured set of
    stateful or non-deterministic builtins, it preserves custom operators, operators
    that reference subgraphs, operators with explicit mutation metadata, and
    operators that touch variable tensors.
    """

    effectful_builtin_codes: frozenset[int]
    custom_builtin_code: int | None
    preserve_custom_operators: bool
    preserve_variable_operators: bool

    def __init__(
        self,
        *,
        effectful_builtin_codes: Iterable[int] = (),
        custom_builtin_code: int | None = None,
        preserve_custom_operators: bool = True,
        preserve_variable_operators: bool = True,
    ) -> None:
        """Create an effect analysis from resolved Circle builtin values."""

        object.__setattr__(
            self,
            "effectful_builtin_codes",
            frozenset(int(code) for code in effectful_builtin_codes),
        )
        object.__setattr__(
            self,
            "custom_builtin_code",
            None if custom_builtin_code is None else int(custom_builtin_code),
        )
        object.__setattr__(
            self,
            "preserve_custom_operators",
            bool(preserve_custom_operators),
        )
        object.__setattr__(
            self,
            "preserve_variable_operators",
            bool(preserve_variable_operators),
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        effectful_builtin_names: Iterable[str] = DEFAULT_EFFECTFUL_BUILTIN_NAMES,
        builtin_codes: Mapping[str, int] | None = None,
        preserve_custom_operators: bool = True,
        preserve_variable_operators: bool = True,
    ) -> OperatorEffectAnalysis:
        """Create a default analysis appropriate for one Object API model.

        Generated Circle models use the real ``BuiltinOperator`` numeric values, so
        their effectful builtin set can be resolved from the schema. Tests and other
        schema-independent clients often use arbitrary integers for operator codes;
        interpreting those integers as generated enum values would incorrectly keep
        pure dead operators. Such clients still receive conservative custom,
        subgraph-reference, mutation, and variable-tensor handling, and may supply an
        explicit ``builtin_codes`` mapping when numeric builtin classification is
        required.
        """

        if builtin_codes is not None or _uses_generated_circle_model(model):
            return cls.from_schema(
                effectful_builtin_names=effectful_builtin_names,
                builtin_codes=builtin_codes,
                preserve_custom_operators=preserve_custom_operators,
                preserve_variable_operators=preserve_variable_operators,
            )
        return cls(
            preserve_custom_operators=preserve_custom_operators,
            preserve_variable_operators=preserve_variable_operators,
        )

    @classmethod
    def from_schema(
        cls,
        *,
        effectful_builtin_names: Iterable[str] = DEFAULT_EFFECTFUL_BUILTIN_NAMES,
        builtin_codes: Mapping[str, int] | None = None,
        preserve_custom_operators: bool = True,
        preserve_variable_operators: bool = True,
    ) -> OperatorEffectAnalysis:
        """Resolve a conservative default analysis from the generated schema."""

        configured = {
            str(name).strip().upper(): int(code)
            for name, code in (builtin_codes or {}).items()
        }
        resolved_codes: set[int] = set()
        for raw_name in effectful_builtin_names:
            name = str(raw_name).strip().upper()
            if not name:
                raise ValueError(
                    "effectful_builtin_names must not contain empty names."
                )
            code = configured.get(name)
            if code is None:
                code = _maybe_builtin_code(name)
            if code is not None:
                resolved_codes.add(code)

        custom_code = configured.get("CUSTOM")
        if custom_code is None:
            custom_code = _maybe_builtin_code("CUSTOM")
        return cls(
            effectful_builtin_codes=resolved_codes,
            custom_builtin_code=custom_code,
            preserve_custom_operators=preserve_custom_operators,
            preserve_variable_operators=preserve_variable_operators,
        )

    def has_observable_effect(
        self,
        graph: CircleGraph,
        operator_index: int,
    ) -> bool:
        """Return whether removing an otherwise dead operator may be observable."""

        operators = as_list(getattr(graph.subgraph, "operators", None))
        if operator_index < 0 or operator_index >= len(operators):
            raise IndexError(
                f"Operator index {operator_index} is outside "
                f"0..{len(operators) - 1}."
            )
        operator = operators[operator_index]
        builtin_code = effective_builtin_code(graph.model, operator)

        if builtin_code in self.effectful_builtin_codes:
            return True
        if self.preserve_custom_operators and self._has_custom_semantics(
            graph.model,
            operator,
            builtin_code=builtin_code,
        ):
            return True
        if any(
            bool(value)
            for value in as_list(getattr(operator, "mutatingVariableInputs", None))
        ):
            return True
        if _operator_references_subgraph(operator):
            return True
        if self.preserve_variable_operators and _operator_touches_variable_tensor(
            graph,
            operator,
        ):
            return True
        return False

    def _has_custom_semantics(
        self,
        model: Any,
        operator: Any,
        *,
        builtin_code: int,
    ) -> bool:
        """Return whether an operator carries an unknown custom implementation."""

        if (
            self.custom_builtin_code is not None
            and builtin_code == self.custom_builtin_code
        ):
            return True
        operator_code = _operator_code(model, operator)
        custom_code = getattr(operator_code, "customCode", None)
        if custom_code not in (None, "", b""):
            return True
        if int(getattr(operator, "largeCustomOptionsSize", 0) or 0) > 0:
            return True
        custom_options = getattr(operator, "customOptions", None)
        if custom_options is None:
            return False
        try:
            return len(custom_options) > 0
        except TypeError:
            return True


def effective_builtin_code(model: Any, operator: Any) -> int:
    """Return the effective Circle builtin code referenced by one operator."""

    operator_code = _operator_code(model, operator)
    builtin_code = int(getattr(operator_code, "builtinCode", 0) or 0)
    deprecated_code = int(
        getattr(operator_code, "deprecatedBuiltinCode", builtin_code) or 0
    )
    placeholder = _maybe_builtin_code("PLACEHOLDER_FOR_GREATER_OP_CODES")
    if builtin_code == 0 and deprecated_code != 0:
        return deprecated_code
    if (
        placeholder is not None
        and builtin_code == placeholder
        and deprecated_code != placeholder
    ):
        return deprecated_code
    return builtin_code


def _operator_code(model: Any, operator: Any) -> Any:
    """Return one valid operator-code record with a descriptive failure."""

    operator_codes = as_list(getattr(model, "operatorCodes", None))
    opcode_index = int(getattr(operator, "opcodeIndex", -1))
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        raise CircleRewriteError(
            f"Operator references invalid opcode index {opcode_index}."
        )
    return operator_codes[opcode_index]


def _operator_touches_variable_tensor(graph: CircleGraph, operator: Any) -> bool:
    """Return whether an operator reads, writes, or uses a variable tensor."""

    tensors = as_list(getattr(graph.subgraph, "tensors", None))
    indices = (
        *as_indices(getattr(operator, "inputs", None)),
        *as_indices(getattr(operator, "outputs", None)),
        *as_indices(getattr(operator, "intermediates", None)),
    )
    for tensor_index in indices:
        if tensor_index == OPTIONAL_TENSOR_INDEX:
            continue
        if tensor_index < 0 or tensor_index >= len(tensors):
            return True
        if bool(getattr(tensors[tensor_index], "isVariable", False)):
            return True
    return False


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


def _uses_generated_circle_model(model: Any) -> bool:
    """Return whether a model uses the generated Circle Object API type."""

    try:
        return isinstance(model, object_api_type("Model"))
    except (AttributeError, ImportError, RuntimeError, TypeError):
        return False


def _maybe_builtin_code(name: str) -> int | None:
    """Resolve one generated BuiltinOperator member when it is available."""

    try:
        schema = circle_schema()
        enum_module = getattr(schema, "BuiltinOperator", None)
        enum_type = (
            getattr(enum_module, "BuiltinOperator", None)
            if enum_module is not None
            else None
        )
        if enum_type is None:
            enum_type = enum_module
        if enum_type is None or not hasattr(enum_type, name):
            return None
        return int(getattr(enum_type, name))
    except (AttributeError, ImportError, RuntimeError):
        return None


__all__ = [
    "DEFAULT_EFFECTFUL_BUILTIN_NAMES",
    "OperatorEffectAnalysis",
    "effective_builtin_code",
]
