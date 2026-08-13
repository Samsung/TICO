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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from tico.circle.graph import as_indices, as_list
from tico.circle.passes.optimization.remove.layout_ops import (
    _builtin_operator_value,
    _get_const_data,
)


_ADD_BUILTIN_CODE = _builtin_operator_value("ADD")
_PAD_BUILTIN_CODE = _builtin_operator_value("PAD")


def _shape(tensor: Any) -> tuple[int, ...]:
    """Return one tensor shape as a tuple of integers."""

    value = getattr(tensor, "shape", None)
    if value is None:
        return ()
    return tuple(int(dimension) for dimension in value)


def _permuted_shape(
    shape: tuple[int, ...],
    permutation: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Return a permuted shape when rank and permutation are valid."""

    if len(shape) != len(permutation):
        return None
    if sorted(permutation) != list(range(len(permutation))):
        return None
    return tuple(shape[index] for index in permutation)


def _operator_builtin_code(
    operator: Any,
    operator_codes: list[Any],
) -> int | None:
    """Return one operator's Circle builtin code when its index is valid."""

    try:
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            return None
        return int(getattr(operator_codes[opcode_index], "builtinCode", -1))
    except (TypeError, ValueError, AttributeError):
        return None


def _flatten_rows(rows: list[tuple[int, int]]) -> tuple[int, ...]:
    """Flatten rank-by-two rows into one immutable integer tuple."""

    return tuple(value for row in rows for value in row)


def _remap_axis_rows(
    values: tuple[int, ...],
    rank: int,
    source_to_region: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Reorder region-layout rank-by-two rows into source-layout order."""

    if len(values) != rank * 2 or len(source_to_region) != rank:
        return None
    old_rows = [(values[axis * 2], values[axis * 2 + 1]) for axis in range(rank)]
    new_rows: list[tuple[int, int] | None] = [None] * rank
    for region_axis, source_axis in enumerate(source_to_region):
        new_rows[source_axis] = old_rows[region_axis]
    if any(row is None for row in new_rows):
        return None
    return _flatten_rows([row for row in new_rows if row is not None])


def _padding_matches_shape(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    values: tuple[int, ...],
) -> bool:
    """Return whether constant padding transforms one shape into another."""

    if len(input_shape) != len(output_shape):
        return False
    if len(values) != len(input_shape) * 2:
        return False
    for axis, input_size in enumerate(input_shape):
        before = values[axis * 2]
        after = values[axis * 2 + 1]
        if before < 0 or after < 0:
            return False
        if input_size + before + after != output_shape[axis]:
            return False
    return True


@dataclass(frozen=True)
class _ConstantInputRewrite:
    """Describe one cloned INT32 constant input with replacement values."""

    input_position: int
    source_tensor_index: int
    values: tuple[int, ...]


@dataclass(frozen=True)
class _OperatorRewritePlan:
    """Describe operator-local metadata rewrites for source-layout execution."""

    operator_index: int
    constant_input_rewrites: tuple[_ConstantInputRewrite, ...] = ()


@dataclass(frozen=True)
class _RegionOpContext:
    """Provide graph and layout information to one registered operator rule."""

    graph: Any
    operator_index: int
    operator: Any
    source_to_region_permutation: tuple[int, ...]
    region_to_source_permutation: tuple[int, ...]

    @property
    def inputs(self) -> tuple[int, ...]:
        """Return operator input tensor indices."""

        return tuple(as_indices(getattr(self.operator, "inputs", None)))

    @property
    def outputs(self) -> tuple[int, ...]:
        """Return operator output tensor indices."""

        return tuple(as_indices(getattr(self.operator, "outputs", None)))

    @property
    def tensors(self) -> list[Any]:
        """Return the subgraph tensor vector as a Python list."""

        return as_list(self.graph.subgraph.tensors)

    @property
    def rank(self) -> int:
        """Return the fixed rank represented by the region permutation."""

        return len(self.source_to_region_permutation)


class _RegionOpRule(ABC):
    """Define how one Circle builtin participates in a layout region."""

    def __init__(self, builtin_code: int) -> None:
        """Create a rule for one Circle builtin code."""

        self._builtin_code = builtin_code

    @property
    def builtin_code(self) -> int:
        """Return the Circle builtin code handled by this rule."""

        return self._builtin_code

    @abstractmethod
    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return inputs whose tensor layout follows the region data."""

    @abstractmethod
    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return outputs whose tensor layout follows the region data."""

    @abstractmethod
    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Validate one operator and plan its source-layout metadata rewrite."""


class _SameShapeBinaryElementwiseRule(_RegionOpRule):
    """Support one binary elementwise operator without broadcasting."""

    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return both binary data input positions."""

        del operator
        return (0, 1)

    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the single data output position."""

        del operator
        return (0,)

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Accept a binary operator only when no broadcasting is required."""

        if len(context.inputs) != 2 or len(context.outputs) != 1:
            return None
        tensor_count = len(context.tensors)
        tensor_indices = (*context.inputs, *context.outputs)
        if any(index < 0 or index >= tensor_count for index in tensor_indices):
            return None
        left_shape = _shape(context.tensors[context.inputs[0]])
        right_shape = _shape(context.tensors[context.inputs[1]])
        output_shape = _shape(context.tensors[context.outputs[0]])
        if left_shape != right_shape or left_shape != output_shape:
            return None
        return _OperatorRewritePlan(context.operator_index)


class _PadRule(_RegionOpRule):
    """Support constant PAD by remapping its rank-by-two axis rows."""

    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the PAD data input position."""

        del operator
        return (0,)

    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the PAD data output position."""

        del operator
        return (0,)

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Validate PAD and plan a cloned source-layout padding constant."""

        if len(context.inputs) != 2 or len(context.outputs) != 1:
            return None
        tensor_count = len(context.tensors)
        tensor_indices = (*context.inputs, *context.outputs)
        if any(index < 0 or index >= tensor_count for index in tensor_indices):
            return None

        padding_tensor_index = context.inputs[1]
        padding_shape = _shape(context.tensors[padding_tensor_index])
        if padding_shape != (context.rank, 2):
            return None
        raw_values = _get_const_data(context.graph, padding_tensor_index)
        if raw_values is None:
            return None
        values = tuple(int(value) for value in raw_values)

        input_shape = _shape(context.tensors[context.inputs[0]])
        output_shape = _shape(context.tensors[context.outputs[0]])
        if not _padding_matches_shape(input_shape, output_shape, values):
            return None

        remapped_values = _remap_axis_rows(
            values,
            context.rank,
            context.source_to_region_permutation,
        )
        if remapped_values is None:
            return None
        source_input_shape = _permuted_shape(
            input_shape,
            context.region_to_source_permutation,
        )
        source_output_shape = _permuted_shape(
            output_shape,
            context.region_to_source_permutation,
        )
        if source_input_shape is None or source_output_shape is None:
            return None
        if not _padding_matches_shape(
            source_input_shape,
            source_output_shape,
            remapped_values,
        ):
            return None

        return _OperatorRewritePlan(
            operator_index=context.operator_index,
            constant_input_rewrites=(
                _ConstantInputRewrite(
                    input_position=1,
                    source_tensor_index=padding_tensor_index,
                    values=remapped_values,
                ),
            ),
        )


_REGION_OP_RULES: Mapping[int, _RegionOpRule] = MappingProxyType(
    {
        _ADD_BUILTIN_CODE: _SameShapeBinaryElementwiseRule(_ADD_BUILTIN_CODE),
        _PAD_BUILTIN_CODE: _PadRule(_PAD_BUILTIN_CODE),
    }
)


def _rule_for_builtin_code(builtin_code: int) -> _RegionOpRule | None:
    """Return the registered rule for one Circle builtin code."""

    return _REGION_OP_RULES.get(builtin_code)


def _rule_for_operator(
    operator: Any,
    operator_codes: list[Any],
) -> _RegionOpRule | None:
    """Return the registered rule for one Circle operator."""

    builtin_code = _operator_builtin_code(operator, operator_codes)
    if builtin_code is None:
        return None
    return _rule_for_builtin_code(builtin_code)
