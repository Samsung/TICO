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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tico.circle._object import create_object, ObjectFactory
from tico.circle._schema import circle_schema
from tico.circle.analysis import TensorContract
from tico.circle.errors import CircleRewriteError, CircleValueError
from tico.circle.graph import as_indices, as_list
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True)
class AppendedObjectCheckpoint:
    """Record mutable sequence lengths before a rewrite allocates objects."""

    subgraph_index: int
    buffer_count: int
    operator_code_count: int
    tensor_count: int

    @classmethod
    def capture(
        cls,
        document: Any,
        *,
        subgraph_index: int,
    ) -> "AppendedObjectCheckpoint":
        """Capture model-global and subgraph-local allocation boundaries."""

        subgraphs = as_list(getattr(document.model, "subgraphs", None))
        if subgraph_index < 0 or subgraph_index >= len(subgraphs):
            raise IndexError(
                f"Subgraph index {subgraph_index} is outside "
                f"0..{len(subgraphs) - 1}."
            )
        subgraph = subgraphs[subgraph_index]
        return cls(
            subgraph_index=int(subgraph_index),
            buffer_count=len(as_list(getattr(document.model, "buffers", None))),
            operator_code_count=len(
                as_list(getattr(document.model, "operatorCodes", None))
            ),
            tensor_count=len(as_list(getattr(subgraph, "tensors", None))),
        )

    def rollback(self, document: Any) -> None:
        """Remove objects appended after this checkpoint was captured."""

        subgraph = as_list(document.model.subgraphs)[self.subgraph_index]
        _truncate_field(document.model, "buffers", self.buffer_count)
        _truncate_field(
            document.model,
            "operatorCodes",
            self.operator_code_count,
        )
        _truncate_field(subgraph, "tensors", self.tensor_count)


class OptimizationSchemaResolver:
    """Resolve Circle schema enums with optional schema-independent overrides."""

    def __init__(
        self,
        *,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        activation_none: int | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Store enum overrides and an optional Object API factory."""

        self._builtin_codes = _normalize_mapping(builtin_codes)
        self._builtin_options_types = _normalize_mapping(builtin_options_types)
        self._tensor_types = _normalize_mapping(tensor_types)
        self._activation_none = (
            None if activation_none is None else int(activation_none)
        )
        self.object_factory = object_factory

    def builtin_code(self, name: str) -> int:
        """Return one required BuiltinOperator enum value."""

        normalized = name.upper()
        if normalized in self._builtin_codes:
            return self._builtin_codes[normalized]
        return _schema_enum_value("BuiltinOperator", normalized)

    def maybe_builtin_code(self, name: str) -> int | None:
        """Return an optional BuiltinOperator value when the schema provides it."""

        try:
            return self.builtin_code(name)
        except (AttributeError, ImportError, RuntimeError):
            return None

    def builtin_options_type(self, name: str) -> int:
        """Return one required BuiltinOptions enum value."""

        normalized = name
        if normalized in self._builtin_options_types:
            return self._builtin_options_types[normalized]
        return _schema_enum_value("BuiltinOptions", normalized)

    def tensor_type(self, name: str) -> int:
        """Return one required TensorType enum value."""

        normalized = name.upper()
        if normalized in self._tensor_types:
            return self._tensor_types[normalized]
        return _schema_enum_value("TensorType", normalized)

    @property
    def activation_none(self) -> int:
        """Return the enum value representing no fused activation."""

        if self._activation_none is not None:
            return self._activation_none
        return _schema_enum_value("ActivationFunctionType", "NONE")

    def create(self, table_name: str) -> Any:
        """Create one generated Object API table or a test substitute."""

        return create_object(table_name, self.object_factory)


def operator_builtin_code(model: Any, operator: Any) -> int:
    """Return the effective builtin code referenced by one operator."""

    operator_codes = as_list(getattr(model, "operatorCodes", None))
    opcode_index = int(getattr(operator, "opcodeIndex", -1))
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        raise CircleRewriteError(
            f"Operator references invalid operator-code index {opcode_index}."
        )
    operator_code = operator_codes[opcode_index]
    builtin_code = int(getattr(operator_code, "builtinCode", 0) or 0)
    deprecated_code = int(
        getattr(operator_code, "deprecatedBuiltinCode", builtin_code) or 0
    )
    placeholder = 127
    try:
        placeholder = _schema_enum_value(
            "BuiltinOperator",
            "PLACEHOLDER_FOR_GREATER_OP_CODES",
        )
    except (AttributeError, ImportError, RuntimeError):
        pass
    if builtin_code == 0 and deprecated_code != 0:
        return deprecated_code
    if builtin_code == placeholder and deprecated_code != placeholder:
        return deprecated_code
    return builtin_code


def operator_version(model: Any, operator: Any) -> int:
    """Return the positive version stored in an operator-code record."""

    operator_codes = as_list(getattr(model, "operatorCodes", None))
    opcode_index = int(getattr(operator, "opcodeIndex", -1))
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        raise CircleRewriteError(
            f"Operator references invalid operator-code index {opcode_index}."
        )
    return max(1, int(getattr(operator_codes[opcode_index], "version", 1) or 1))


def tensor_contract(graph: Any, tensor_index: int) -> TensorContract:
    """Capture the contract of one subgraph-local tensor."""

    tensors = as_list(getattr(graph.subgraph, "tensors", None))
    if tensor_index < 0 or tensor_index >= len(tensors):
        raise IndexError(
            f"Tensor index {tensor_index} is outside 0..{len(tensors) - 1}."
        )
    return TensorContract.from_tensor(tensors[tensor_index])


def tensor_object(graph: Any, tensor_index: int) -> Any:
    """Return one subgraph-local tensor with a bounds check."""

    tensors = as_list(getattr(graph.subgraph, "tensors", None))
    if tensor_index < 0 or tensor_index >= len(tensors):
        raise IndexError(
            f"Tensor index {tensor_index} is outside 0..{len(tensors) - 1}."
        )
    return tensors[tensor_index]


def contract_is_fully_static(contract: TensorContract) -> bool:
    """Return whether a tensor contract has no dynamic signature dimensions."""

    signature = contract.shape_signature
    return signature is None or all(dimension >= 0 for dimension in signature)


def contract_is_dense_nonvariable(contract: TensorContract) -> bool:
    """Return whether a contract represents a dense immutable activation value."""

    return (
        not contract.is_variable
        and contract.sparsity is None
        and contract.variant_tensors is None
    )


def quantization_is_per_tensor_or_absent(contract: TensorContract) -> bool:
    """Return whether a contract avoids per-axis quantization semantics."""

    quantization = contract.quantization
    return quantization is None or len(quantization.scale) <= 1


def view_contracts_compatible(
    source: TensorContract,
    target: TensorContract,
    *,
    allow_per_axis: bool = False,
) -> bool:
    """Compare non-shape metadata required by a storage-preserving view rewrite."""

    if not (
        contract_is_fully_static(source)
        and contract_is_fully_static(target)
        and contract_is_dense_nonvariable(source)
        and contract_is_dense_nonvariable(target)
    ):
        return False
    if not allow_per_axis and not (
        quantization_is_per_tensor_or_absent(source)
        and quantization_is_per_tensor_or_absent(target)
    ):
        return False
    return (
        source.tensor_type == target.tensor_type
        and source.quantization == target.quantization
        and source.is_variable == target.is_variable
        and source.sparsity_fingerprint == target.sparsity_fingerprint
        and source.has_rank == target.has_rank
        and source.variant_fingerprint == target.variant_fingerprint
    )


def decode_integer_vector(
    codec: TensorValueCodec,
    model: Any,
    *,
    subgraph_index: int,
    tensor_index: int,
    expected_count: int | None = None,
) -> tuple[int, ...] | None:
    """Decode one inline integer constant as a flat tuple."""

    try:
        value = codec.decode_tensor(
            model,
            subgraph_index=subgraph_index,
            tensor_index=tensor_index,
        )
    except (CircleValueError, IndexError):
        return None
    if value.data.dtype.kind not in {"i", "u"}:
        return None
    flattened = tuple(int(item) for item in value.data.reshape(-1))
    if expected_count is not None and len(flattened) != expected_count:
        return None
    return flattened


def decode_constant_value(
    codec: TensorValueCodec,
    model: Any,
    *,
    subgraph_index: int,
    tensor_index: int,
) -> TensorValue | None:
    """Decode one inline constant or return None for unsupported storage."""

    try:
        return codec.decode_tensor(
            model,
            subgraph_index=subgraph_index,
            tensor_index=tensor_index,
        )
    except (CircleValueError, IndexError):
        return None


def constant_represents_real_zero(
    value: TensorValue,
    *,
    reference_contract: TensorContract,
) -> bool:
    """Return whether a stored scalar or tensor represents exact real zero."""

    if value.quantization != reference_contract.quantization:
        return False
    quantization = value.quantization
    if quantization is None:
        return bool(np.all(np.equal(value.data, 0)))
    if len(quantization.scale) != 1 or len(quantization.zero_point) != 1:
        return False
    return bool(np.all(np.equal(value.data, np.asarray(quantization.zero_point[0]))))


def normalize_axis(axis: int, rank: int, *, allow_end: bool = False) -> int | None:
    """Normalize one possibly negative axis within a tensor rank."""

    upper = rank + (1 if allow_end else 0)
    normalized = int(axis)
    if normalized < 0:
        normalized += upper
    if normalized < 0 or normalized >= upper:
        return None
    return normalized


def infer_reshape_shape(
    input_shape: Sequence[int],
    requested_shape: Sequence[int],
) -> tuple[int, ...] | None:
    """Resolve one optional inferred dimension in a static reshape target."""

    requested = [int(dimension) for dimension in requested_shape]
    if any(dimension < -1 for dimension in requested):
        return None
    inferred = [index for index, dimension in enumerate(requested) if dimension == -1]
    if len(inferred) > 1:
        return None
    input_elements = _element_count(input_shape)
    known_elements = _element_count(
        dimension for dimension in requested if dimension != -1
    )
    if inferred:
        if known_elements == 0 or input_elements % known_elements != 0:
            return None
        requested[inferred[0]] = input_elements // known_elements
    elif known_elements != input_elements:
        return None
    return tuple(requested)


def strided_slice_view_shape(
    input_shape: Sequence[int],
    begin: Sequence[int],
    end: Sequence[int],
    strides: Sequence[int],
    *,
    begin_mask: int,
    end_mask: int,
    ellipsis_mask: int,
    new_axis_mask: int,
    shrink_axis_mask: int,
) -> tuple[int, ...] | None:
    """Return a view-only StridedSlice output shape or None when data is sliced."""

    shape = tuple(int(dimension) for dimension in input_shape)
    begin_values = tuple(int(value) for value in begin)
    end_values = tuple(int(value) for value in end)
    stride_values = tuple(int(value) for value in strides)
    if not (len(begin_values) == len(end_values) == len(stride_values)):
        return None
    spec_count = len(begin_values)
    if ellipsis_mask != 0 or any(stride != 1 for stride in stride_values):
        return None
    all_masks = (
        int(begin_mask),
        int(end_mask),
        int(new_axis_mask),
        int(shrink_axis_mask),
    )
    if any(mask < 0 or mask >> spec_count for mask in all_masks):
        return None

    input_axis = 0
    output_shape: list[int] = []
    for position in range(spec_count):
        is_new_axis = _mask_bit(new_axis_mask, position)
        is_shrink_axis = _mask_bit(shrink_axis_mask, position)
        if is_new_axis and is_shrink_axis:
            return None
        if is_new_axis:
            output_shape.append(1)
            continue
        if input_axis >= len(shape):
            return None
        dimension = shape[input_axis]
        if is_shrink_axis:
            if dimension != 1:
                return None
            index = 0 if _mask_bit(begin_mask, position) else begin_values[position]
            if index < 0:
                index += dimension
            if index != 0:
                return None
        else:
            start = 0 if _mask_bit(begin_mask, position) else begin_values[position]
            stop = dimension if _mask_bit(end_mask, position) else end_values[position]
            if start < 0:
                start += dimension
            if stop < 0:
                stop += dimension
            if start != 0 or stop != dimension:
                return None
            output_shape.append(dimension)
        input_axis += 1

    if input_axis != len(shape):
        return None
    return tuple(output_shape)


def transpose_is_view_only(
    input_shape: Sequence[int],
    permutation: Sequence[int],
) -> bool:
    """Return whether a transpose only relocates unit dimensions."""

    shape = tuple(int(dimension) for dimension in input_shape)
    perm = tuple(int(axis) for axis in permutation)
    if len(perm) != len(shape) or sorted(perm) != list(range(len(shape))):
        return False
    non_unit_axes = tuple(
        axis for axis, dimension in enumerate(shape) if dimension != 1
    )
    permuted_non_unit_axes = tuple(axis for axis in perm if shape[axis] != 1)
    return permuted_non_unit_axes == non_unit_axes


def output_shape_matches_transpose(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    permutation: Sequence[int],
) -> bool:
    """Return whether tensor shapes agree with a static transpose permutation."""

    shape = tuple(int(dimension) for dimension in input_shape)
    perm = tuple(int(axis) for axis in permutation)
    if len(perm) != len(shape) or sorted(perm) != list(range(len(shape))):
        return False
    return tuple(shape[axis] for axis in perm) == tuple(output_shape)


def _truncate_field(owner: Any, field_name: str, length: int) -> None:
    """Truncate one generated vector field to a previously captured length."""

    values = as_list(getattr(owner, field_name, None))
    if length < 0 or length > len(values):
        raise CircleRewriteError(
            f"Cannot restore {field_name} to length {length} from {len(values)}."
        )
    setattr(owner, field_name, values[:length])


def _normalize_mapping(value: Mapping[str, int] | None) -> dict[str, int]:
    """Normalize optional string-to-enum overrides to plain dictionaries."""

    if value is None:
        return {}
    return {str(name): int(enum_value) for name, enum_value in value.items()}


def _schema_enum_value(enum_name: str, member_name: str) -> int:
    """Return a generated Circle enum member by symbolic name."""

    module = circle_schema()
    enum_module = getattr(module, enum_name, None)
    enum_type = (
        getattr(enum_module, enum_name, None) if enum_module is not None else None
    )
    if enum_type is None:
        enum_type = getattr(module, enum_name, None)
    if enum_type is None or not hasattr(enum_type, member_name):
        raise RuntimeError(f"Circle schema does not provide {enum_name}.{member_name}.")
    return int(getattr(enum_type, member_name))


def _mask_bit(mask: int, position: int) -> bool:
    """Return whether one bit is set in a StridedSlice option mask."""

    return bool(int(mask) & (1 << position))


def _element_count(shape: Sequence[int] | Any) -> int:
    """Return the product of a shape-like sequence."""

    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count


def operator_is_plain(operator: Any) -> bool:
    """Return whether an operator has no mutation or auxiliary tensor state."""

    if as_indices(getattr(operator, "intermediates", None)):
        return False
    if any(
        bool(value)
        for value in as_list(getattr(operator, "mutatingVariableInputs", None))
    ):
        return False
    if int(getattr(operator, "builtinOptions2Type", 0) or 0):
        return False
    if getattr(operator, "builtinOptions2", None) is not None:
        return False
    if int(getattr(operator, "customOptionsFormat", 0) or 0):
        return False
    if int(getattr(operator, "largeCustomOptionsOffset", 0) or 0):
        return False
    if int(getattr(operator, "largeCustomOptionsSize", 0) or 0):
        return False
    custom_options = getattr(operator, "customOptions", None)
    if custom_options is None:
        return True
    try:
        return len(custom_options) == 0
    except TypeError:
        return False


def tensor_is_signature_bound(
    model: Any,
    *,
    subgraph_index: int,
    tensor_index: int,
) -> bool:
    """Return whether any signature input or output maps to a tensor."""

    for signature in as_list(getattr(model, "signatureDefs", None)):
        if int(getattr(signature, "subgraphIndex", -1)) != subgraph_index:
            continue
        for field_name in ("inputs", "outputs"):
            for tensor_map in as_list(getattr(signature, field_name, None)):
                if int(getattr(tensor_map, "tensorIndex", -1)) == tensor_index:
                    return True
    return False
