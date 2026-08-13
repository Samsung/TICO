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
_ADD_N_BUILTIN_CODE = _builtin_operator_value("ADD_N")
_CONCATENATION_BUILTIN_CODE = _builtin_operator_value("CONCATENATION")
_MIRROR_PAD_BUILTIN_CODE = _builtin_operator_value("MIRROR_PAD")
_PAD_BUILTIN_CODE = _builtin_operator_value("PAD")
_PADV2_BUILTIN_CODE = _builtin_operator_value("PADV2")
_SLICE_BUILTIN_CODE = _builtin_operator_value("SLICE")
_TILE_BUILTIN_CODE = _builtin_operator_value("TILE")

_AXIS_REMAP_BUILTIN_CODES: Mapping[str, int] = MappingProxyType(
    {
        "CONCATENATION": _CONCATENATION_BUILTIN_CODE,
        "MIRROR_PAD": _MIRROR_PAD_BUILTIN_CODE,
        "PADV2": _PADV2_BUILTIN_CODE,
        "SLICE": _SLICE_BUILTIN_CODE,
        "TILE": _TILE_BUILTIN_CODE,
    }
)

_UNARY_LAYOUT_INVARIANT_BUILTIN_CODES: Mapping[str, int] = MappingProxyType(
    {
        name: _builtin_operator_value(name)
        for name in (
            "ABS",
            "CAST",
            "CEIL",
            "COS",
            "DEQUANTIZE",
            "ELU",
            "EXP",
            "FLOOR",
            "LEAKY_RELU",
            "LOG",
            "LOGICAL_NOT",
            "LOGISTIC",
            "NEG",
            "QUANTIZE",
            "RELU",
            "RELU6",
            "RELU_N1_TO_1",
            "RSQRT",
            "SIN",
            "SQRT",
            "SQUARE",
            "TANH",
            "ZEROS_LIKE",
        )
    }
)

_BINARY_LAYOUT_INVARIANT_BUILTIN_CODES: Mapping[str, int] = MappingProxyType(
    {
        name: _builtin_operator_value(name)
        for name in (
            "ADD",
            "DIV",
            "EQUAL",
            "FLOOR_DIV",
            "FLOOR_MOD",
            "GREATER",
            "GREATER_EQUAL",
            "LESS",
            "LESS_EQUAL",
            "LOGICAL_AND",
            "LOGICAL_OR",
            "MAXIMUM",
            "MINIMUM",
            "MUL",
            "NOT_EQUAL",
            "POW",
            "SQUARED_DIFFERENCE",
            "SUB",
        )
    }
)

_VARIADIC_LAYOUT_INVARIANT_BUILTIN_CODES: Mapping[str, int] = MappingProxyType(
    {"ADD_N": _ADD_N_BUILTIN_CODE}
)


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


def _remap_axis_vector(
    values: tuple[int, ...],
    rank: int,
    source_to_region: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Reorder one region-layout rank vector into source-layout order."""

    if len(values) != rank or len(source_to_region) != rank:
        return None
    remapped = [0] * rank
    for region_axis, source_axis in enumerate(source_to_region):
        remapped[source_axis] = values[region_axis]
    return tuple(remapped)


def _normalize_axis(axis: int, rank: int) -> int | None:
    """Normalize one possibly negative axis against a known rank."""

    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        return None
    return normalized


def _remap_axis(
    axis: int,
    rank: int,
    source_to_region: tuple[int, ...],
) -> int | None:
    """Map one region-layout axis into the source layout."""

    normalized = _normalize_axis(axis, rank)
    if normalized is None or len(source_to_region) != rank:
        return None
    return source_to_region[normalized]


def _rank_vector_constant(
    context: _RegionOpContext,
    input_position: int,
) -> tuple[int, tuple[int, ...]] | None:
    """Return one constant INT32 rank vector and its tensor index."""

    if input_position < 0 or input_position >= len(context.inputs):
        return None
    tensor_index = context.inputs[input_position]
    if tensor_index < 0 or tensor_index >= len(context.tensors):
        return None
    if _shape(context.tensors[tensor_index]) != (context.rank,):
        return None
    raw_values = _get_const_data(context.graph, tensor_index)
    if raw_values is None or len(raw_values) != context.rank:
        return None
    return tensor_index, tuple(int(value) for value in raw_values)


def _source_layout_shapes(
    context: _RegionOpContext,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return input and output shapes converted into source-layout order."""

    source_input = _permuted_shape(
        input_shape,
        context.region_to_source_permutation,
    )
    source_output = _permuted_shape(
        output_shape,
        context.region_to_source_permutation,
    )
    if source_input is None or source_output is None:
        return None
    return source_input, source_output


def _tile_matches_shape(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    multiples: tuple[int, ...],
) -> bool:
    """Return whether TILE multiples transform one shape into another."""

    if len(input_shape) != len(output_shape) or len(multiples) != len(input_shape):
        return False
    if any(dimension < 0 for dimension in (*input_shape, *output_shape)):
        return False
    return all(
        multiple >= 0 and input_size * multiple == output_size
        for input_size, output_size, multiple in zip(
            input_shape,
            output_shape,
            multiples,
        )
    )


def _slice_matches_shape(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    begin: tuple[int, ...],
    size: tuple[int, ...],
) -> bool:
    """Return whether one static SLICE specification yields the output shape."""

    rank = len(input_shape)
    if len(output_shape) != rank or len(begin) != rank or len(size) != rank:
        return False
    if any(dimension < 0 for dimension in (*input_shape, *output_shape)):
        return False
    for input_size, output_size, begin_value, size_value in zip(
        input_shape,
        output_shape,
        begin,
        size,
    ):
        if begin_value < 0 or begin_value > input_size:
            return False
        if size_value == -1:
            sliced_size = input_size - begin_value
        elif size_value >= 0:
            sliced_size = size_value
        else:
            return False
        if begin_value + sliced_size > input_size or sliced_size != output_size:
            return False
    return True


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


def _tensor_shapes(
    context: _RegionOpContext,
    tensor_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Return tensor shapes when every index is valid."""

    tensor_count = len(context.tensors)
    if any(index < 0 or index >= tensor_count for index in tensor_indices):
        return None
    return tuple(_shape(context.tensors[index]) for index in tensor_indices)


def _all_shapes_equal(shapes: tuple[tuple[int, ...], ...]) -> bool:
    """Return whether a non-empty sequence contains one common shape."""

    return bool(shapes) and all(shape == shapes[0] for shape in shapes[1:])


@dataclass(frozen=True)
class _ConstantInputRewrite:
    """Describe one cloned INT32 constant input with replacement values."""

    input_position: int
    source_tensor_index: int
    values: tuple[int, ...]


@dataclass(frozen=True)
class _BuiltinOptionRewrite:
    """Describe one integer builtin-option field replacement."""

    field_name: str
    value: int


@dataclass(frozen=True)
class _OperatorRewritePlan:
    """Describe operator-local metadata rewrites for source-layout execution."""

    operator_index: int
    constant_input_rewrites: tuple[_ConstantInputRewrite, ...] = ()
    builtin_option_rewrites: tuple[_BuiltinOptionRewrite, ...] = ()


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


class _SameShapeUnaryElementwiseRule(_RegionOpRule):
    """Support one rank-preserving unary layout-invariant operator."""

    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the single unary data input position."""

        del operator
        return (0,)

    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the single unary data output position."""

        del operator
        return (0,)

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Accept a unary operator only when its shape remains unchanged."""

        if len(context.inputs) != 1 or len(context.outputs) != 1:
            return None
        shapes = _tensor_shapes(
            context,
            (context.inputs[0], context.outputs[0]),
        )
        if shapes is None or not _all_shapes_equal(shapes):
            return None
        return _OperatorRewritePlan(context.operator_index)


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
        shapes = _tensor_shapes(
            context,
            (*context.inputs, context.outputs[0]),
        )
        if shapes is None or not _all_shapes_equal(shapes):
            return None
        return _OperatorRewritePlan(context.operator_index)


class _SameShapeVariadicElementwiseRule(_RegionOpRule):
    """Support one variadic elementwise operator without broadcasting."""

    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return every variadic data input position."""

        return tuple(range(len(as_indices(getattr(operator, "inputs", None)))))

    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the single variadic data output position."""

        del operator
        return (0,)

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Accept a variadic operator when all data tensor shapes are equal."""

        if len(context.inputs) < 2 or len(context.outputs) != 1:
            return None
        shapes = _tensor_shapes(
            context,
            (*context.inputs, context.outputs[0]),
        )
        if shapes is None or not _all_shapes_equal(shapes):
            return None
        return _OperatorRewritePlan(context.operator_index)


class _ConstantPaddingRule(_RegionOpRule):
    """Support one constant-padding operator by remapping axis rows."""

    def __init__(
        self,
        builtin_code: int,
        *,
        input_count: int,
        scalar_value_input_position: int | None = None,
    ) -> None:
        """Create one padding rule with an explicit Circle input contract."""

        super().__init__(builtin_code)
        self._input_count = input_count
        self._scalar_value_input_position = scalar_value_input_position

    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the padding operator data input position."""

        del operator
        return (0,)

    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the padding operator data output position."""

        del operator
        return (0,)

    def _has_valid_scalar_value(self, context: _RegionOpContext) -> bool:
        """Return whether an optional padding value input contains one element."""

        position = self._scalar_value_input_position
        if position is None:
            return True
        if position < 0 or position >= len(context.inputs):
            return False
        tensor_index = context.inputs[position]
        if tensor_index < 0 or tensor_index >= len(context.tensors):
            return False
        shape = _shape(context.tensors[tensor_index])
        element_count = 1
        for dimension in shape:
            if dimension < 0:
                return False
            element_count *= dimension
        return element_count == 1

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Validate padding and plan a cloned source-layout constant."""

        if len(context.inputs) != self._input_count or len(context.outputs) != 1:
            return None
        tensor_count = len(context.tensors)
        tensor_indices = (*context.inputs, *context.outputs)
        if any(index < 0 or index >= tensor_count for index in tensor_indices):
            return None
        if not self._has_valid_scalar_value(context):
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
        source_shapes = _source_layout_shapes(
            context,
            input_shape,
            output_shape,
        )
        if source_shapes is None:
            return None
        if not _padding_matches_shape(
            source_shapes[0],
            source_shapes[1],
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


class _PadRule(_ConstantPaddingRule):
    """Support PAD by remapping its constant rank-by-two axis rows."""

    def __init__(self, builtin_code: int) -> None:
        """Create the two-input PAD rule."""

        super().__init__(builtin_code, input_count=2)


class _PadV2Rule(_ConstantPaddingRule):
    """Support PADV2 while preserving its scalar padding value input."""

    def __init__(self, builtin_code: int) -> None:
        """Create the three-input PADV2 rule."""

        super().__init__(
            builtin_code,
            input_count=3,
            scalar_value_input_position=2,
        )


class _MirrorPadRule(_ConstantPaddingRule):
    """Support MIRROR_PAD while preserving its reflection mode option."""

    def __init__(self, builtin_code: int) -> None:
        """Create the two-input MIRROR_PAD rule."""

        super().__init__(builtin_code, input_count=2)

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Validate MIRROR_PAD mode and plan its remapped padding constant."""

        options = getattr(context.operator, "builtinOptions", None)
        if options is None or not hasattr(options, "mode"):
            return None
        return super().plan_rewrite(context)


class _ConcatenationRule(_RegionOpRule):
    """Support CONCATENATION by remapping its axis builtin option."""

    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return every CONCATENATION data input position."""

        return tuple(range(len(as_indices(getattr(operator, "inputs", None)))))

    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the single CONCATENATION data output position."""

        del operator
        return (0,)

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Validate concatenation shapes and plan source-layout axis remapping."""

        if len(context.inputs) < 2 or len(context.outputs) != 1:
            return None
        shapes = _tensor_shapes(
            context,
            (*context.inputs, context.outputs[0]),
        )
        if shapes is None or any(len(shape) != context.rank for shape in shapes):
            return None

        options = getattr(context.operator, "builtinOptions", None)
        if options is None or not hasattr(options, "axis"):
            return None
        try:
            region_axis = _normalize_axis(int(options.axis), context.rank)
        except (TypeError, ValueError):
            return None
        if region_axis is None:
            return None
        source_axis = _remap_axis(
            region_axis,
            context.rank,
            context.source_to_region_permutation,
        )
        if source_axis is None:
            return None

        input_shapes = shapes[:-1]
        output_shape = shapes[-1]
        if any(dimension < 0 for shape in shapes for dimension in shape):
            return None
        reference = input_shapes[0]
        for shape in input_shapes[1:]:
            if any(
                shape[axis] != reference[axis]
                for axis in range(context.rank)
                if axis != region_axis
            ):
                return None
        if any(
            output_shape[axis] != reference[axis]
            for axis in range(context.rank)
            if axis != region_axis
        ):
            return None
        if output_shape[region_axis] != sum(
            shape[region_axis] for shape in input_shapes
        ):
            return None

        return _OperatorRewritePlan(
            operator_index=context.operator_index,
            builtin_option_rewrites=(_BuiltinOptionRewrite("axis", source_axis),),
        )


class _TileRule(_RegionOpRule):
    """Support TILE by remapping its constant multiples vector."""

    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the TILE data input position."""

        del operator
        return (0,)

    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the TILE data output position."""

        del operator
        return (0,)

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Validate TILE and plan a source-layout multiples constant."""

        if len(context.inputs) != 2 or len(context.outputs) != 1:
            return None
        vector = _rank_vector_constant(context, 1)
        shapes = _tensor_shapes(
            context,
            (context.inputs[0], context.outputs[0]),
        )
        if vector is None or shapes is None:
            return None
        tensor_index, multiples = vector
        if not _tile_matches_shape(shapes[0], shapes[1], multiples):
            return None
        remapped = _remap_axis_vector(
            multiples,
            context.rank,
            context.source_to_region_permutation,
        )
        source_shapes = _source_layout_shapes(context, shapes[0], shapes[1])
        if remapped is None or source_shapes is None:
            return None
        if not _tile_matches_shape(source_shapes[0], source_shapes[1], remapped):
            return None
        return _OperatorRewritePlan(
            operator_index=context.operator_index,
            constant_input_rewrites=(_ConstantInputRewrite(1, tensor_index, remapped),),
        )


class _SliceRule(_RegionOpRule):
    """Support static SLICE by remapping begin and size vectors."""

    def data_input_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the SLICE data input position."""

        del operator
        return (0,)

    def data_output_positions(self, operator: Any) -> tuple[int, ...]:
        """Return the SLICE data output position."""

        del operator
        return (0,)

    def plan_rewrite(
        self,
        context: _RegionOpContext,
    ) -> _OperatorRewritePlan | None:
        """Validate static SLICE and plan source-layout begin and size vectors."""

        if len(context.inputs) != 3 or len(context.outputs) != 1:
            return None
        begin_vector = _rank_vector_constant(context, 1)
        size_vector = _rank_vector_constant(context, 2)
        shapes = _tensor_shapes(
            context,
            (context.inputs[0], context.outputs[0]),
        )
        if begin_vector is None or size_vector is None or shapes is None:
            return None
        begin_index, begin = begin_vector
        size_index, size = size_vector
        if not _slice_matches_shape(shapes[0], shapes[1], begin, size):
            return None

        remapped_begin = _remap_axis_vector(
            begin,
            context.rank,
            context.source_to_region_permutation,
        )
        remapped_size = _remap_axis_vector(
            size,
            context.rank,
            context.source_to_region_permutation,
        )
        source_shapes = _source_layout_shapes(context, shapes[0], shapes[1])
        if remapped_begin is None or remapped_size is None or source_shapes is None:
            return None
        if not _slice_matches_shape(
            source_shapes[0],
            source_shapes[1],
            remapped_begin,
            remapped_size,
        ):
            return None
        return _OperatorRewritePlan(
            operator_index=context.operator_index,
            constant_input_rewrites=(
                _ConstantInputRewrite(1, begin_index, remapped_begin),
                _ConstantInputRewrite(2, size_index, remapped_size),
            ),
        )


def _build_region_op_rules() -> Mapping[int, _RegionOpRule]:
    """Create the immutable builtin-to-rule registry."""

    registry: dict[int, _RegionOpRule] = {}
    for builtin_code in _UNARY_LAYOUT_INVARIANT_BUILTIN_CODES.values():
        registry[builtin_code] = _SameShapeUnaryElementwiseRule(builtin_code)
    for builtin_code in _BINARY_LAYOUT_INVARIANT_BUILTIN_CODES.values():
        registry[builtin_code] = _SameShapeBinaryElementwiseRule(builtin_code)
    for builtin_code in _VARIADIC_LAYOUT_INVARIANT_BUILTIN_CODES.values():
        registry[builtin_code] = _SameShapeVariadicElementwiseRule(builtin_code)
    registry[_CONCATENATION_BUILTIN_CODE] = _ConcatenationRule(
        _CONCATENATION_BUILTIN_CODE
    )
    registry[_MIRROR_PAD_BUILTIN_CODE] = _MirrorPadRule(_MIRROR_PAD_BUILTIN_CODE)
    registry[_PAD_BUILTIN_CODE] = _PadRule(_PAD_BUILTIN_CODE)
    registry[_PADV2_BUILTIN_CODE] = _PadV2Rule(_PADV2_BUILTIN_CODE)
    registry[_SLICE_BUILTIN_CODE] = _SliceRule(_SLICE_BUILTIN_CODE)
    registry[_TILE_BUILTIN_CODE] = _TileRule(_TILE_BUILTIN_CODE)
    return MappingProxyType(registry)


_REGION_OP_RULES: Mapping[int, _RegionOpRule] = _build_region_op_rules()


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
