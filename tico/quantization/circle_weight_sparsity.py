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

import argparse
import csv
import hashlib
import io
import json
import math
import mmap
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_DEFAULT_CHUNK_NUMEL = 1024 * 1024
_SCOPE_NAME = "All model weights"
_SELECTION_MODES = ("auto", "operator-inputs", "quantized-constants")

# Input positions follow the Circle operator signatures emitted by TICO.
# Dynamic inputs are ignored because only tensors backed by constant buffers
# are accepted as weights.
_WEIGHT_INPUT_INDICES: Mapping[str, tuple[int, ...]] = {
    "FULLY_CONNECTED": (1,),
    "CONV_2D": (1,),
    "DEPTHWISE_CONV_2D": (1,),
    "TRANSPOSE_CONV": (1,),
    "GATHER": (0,),
    "RMS_NORM": (1,),
    "PRELU": (1,),
    "INSTANCE_NORM": (1, 2),
    "BATCH_MATMUL": (0, 1),
    "SVDF": (1, 2),
}

# These operators preserve the number and location of zero values in a
# constant weight tensor. Following input zero lets the analyzer resolve a
# stored weight even when a serializer leaves a lightweight transform between
# the buffer and the weight-bearing operator.
_PASSTHROUGH_DATA_INPUT: Mapping[str, int] = {
    "DEQUANTIZE": 0,
    "QUANTIZE": 0,
    "RESHAPE": 0,
    "TRANSPOSE": 0,
    "CAST": 0,
    "SQUEEZE": 0,
    "EXPAND_DIMS": 0,
}

_WEIGHT_NAME_MARKERS = (
    "weight",
    "kernel",
    "embedding",
    "embed_tokens",
    "lm_head",
    "rotation",
    "hadamard",
)
_NAME_EXCLUSION_MARKERS = (
    "bias",
    "zero_point",
    "zeropoint",
)

_DTYPE_ORDER = {
    "int4": 0,
    "uint4": 1,
    "int8": 2,
    "uint8": 3,
    "int16": 4,
    "uint16": 5,
    "int32": 6,
    "uint32": 7,
    "int64": 8,
    "uint64": 9,
    "float16": 10,
    "bfloat16": 11,
    "float32": 12,
    "float64": 13,
    "bool": 14,
}

_NUMPY_DTYPES: Mapping[str, np.dtype[Any]] = {
    "int8": np.dtype("i1"),
    "uint8": np.dtype("u1"),
    "int16": np.dtype("<i2"),
    "uint16": np.dtype("<u2"),
    "int32": np.dtype("<i4"),
    "uint32": np.dtype("<u4"),
    "int64": np.dtype("<i8"),
    "uint64": np.dtype("<u8"),
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "bool": np.dtype("?"),
}


class CircleWeightSparsityError(RuntimeError):
    """Report an invalid or unsupported Circle weight representation."""


@dataclass(frozen=True)
class QuantizationInfo:
    """Store affine metadata needed for zero counting and deduplication."""

    zero_points: tuple[int, ...]
    quantized_dimension: int | None
    is_affine: bool
    scales: tuple[float, ...] = ()


@dataclass(frozen=True)
class CircleWeightTensorStats:
    """Store semantic-zero statistics for one Circle weight tensor."""

    source: str
    subgraph_index: int
    tensor_index: int
    tensor_name: str
    buffer_index: int
    qdtype: str
    shape: tuple[int, ...]
    zero_count: int
    numel: int
    roles: tuple[str, ...]
    fingerprint: str | None = None


@dataclass(frozen=True)
class CircleWeightSparsityRow:
    """Represent the single user-facing model-level sparsity row."""

    scope: str
    qdtype: str
    sparsity_pct: float


@dataclass(frozen=True)
class CircleWeightSparsityReport:
    """Store a model-level result and internal validation counters."""

    row: CircleWeightSparsityRow
    source_count: int
    tensor_count: int
    zero_count: int
    numel: int
    duplicate_tensor_count: int
    skipped_tensor_count: int
    skipped_messages: tuple[str, ...]


def _load_circle_schema() -> Any:
    """Import the Circle schema lazily so pure helper tests need no FlatBuffers."""

    try:
        from circle_schema import circle
    except ImportError as exc:
        raise CircleWeightSparsityError(
            "circle-schema is required. Install TICO or run "
            "`pip install circle-schema`."
        ) from exc
    return circle


def _call_int(obj: Any, method_name: str, default: int = 0) -> int:
    """Call an integer-valued FlatBuffer accessor when it exists."""

    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return default
    return int(method())


def _vector_as_numpy(obj: Any, prefix: str, dtype: np.dtype[Any]) -> np.ndarray:
    """Read a generated FlatBuffer vector through NumPy or scalar accessors."""

    numpy_method = getattr(obj, f"{prefix}AsNumpy", None)
    if numpy_method is not None and callable(numpy_method):
        values = numpy_method()
        if isinstance(values, np.ndarray):
            return np.asarray(values, dtype=dtype).reshape(-1)
        if isinstance(values, memoryview):
            return np.frombuffer(values, dtype=dtype)

    length = _call_int(obj, f"{prefix}Length")
    scalar_method = getattr(obj, prefix, None)
    if length == 0 or scalar_method is None or not callable(scalar_method):
        return np.empty(0, dtype=dtype)
    return np.asarray([scalar_method(index) for index in range(length)], dtype=dtype)


def _decode_name(value: Any, default: str) -> str:
    """Decode a FlatBuffer string accessor into a stable Python string."""

    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _enum_name(enum_class: Any, value: int, prefix: str) -> str:
    """Return the symbolic name of a generated FlatBuffer enum value."""

    for name, enum_value in vars(enum_class).items():
        if name.startswith("_") or not isinstance(enum_value, int):
            continue
        if int(enum_value) == int(value):
            return name
    return f"{prefix}_{value}"


def _tensor_dtype_name(circle: Any, tensor_type: int) -> str:
    """Return a lowercase Circle tensor dtype name."""

    enum_class = circle.TensorType.TensorType
    return _enum_name(enum_class, tensor_type, "TENSOR_TYPE").lower()


def _operator_builtin_code(operator_code: Any) -> int:
    """Resolve an operator code across old and new Circle schema fields."""

    builtin_method = getattr(operator_code, "BuiltinCode", None)
    deprecated_method = getattr(operator_code, "DeprecatedBuiltinCode", None)
    builtin = int(builtin_method()) if callable(builtin_method) else None
    deprecated = int(deprecated_method()) if callable(deprecated_method) else None

    if builtin is None:
        if deprecated is None:
            raise CircleWeightSparsityError("OperatorCode has no builtin code field.")
        return deprecated
    if builtin == 0 and deprecated not in (None, 0):
        return deprecated
    return builtin


def _operator_name(circle: Any, model: Any, operator: Any) -> str:
    """Return a symbolic Circle builtin operator name."""

    opcode_index = _call_int(operator, "OpcodeIndex", -1)
    if opcode_index < 0 or opcode_index >= _call_int(model, "OperatorCodesLength"):
        raise CircleWeightSparsityError(f"Invalid operator code index {opcode_index}.")
    operator_code = model.OperatorCodes(opcode_index)
    builtin_code = _operator_builtin_code(operator_code)
    return _enum_name(
        circle.BuiltinOperator.BuiltinOperator,
        builtin_code,
        "BUILTIN_OPERATOR",
    )


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    """Read a static Circle tensor shape."""

    values = _vector_as_numpy(tensor, "Shape", np.dtype(np.int64))
    shape = tuple(int(value) for value in values)
    if any(dimension < 0 for dimension in shape):
        raise CircleWeightSparsityError(
            f"Negative tensor dimension is unsupported: {shape}."
        )
    return shape


def _tensor_name(tensor: Any, tensor_index: int) -> str:
    """Return a readable Circle tensor name."""

    name_method = getattr(tensor, "Name", None)
    value = name_method() if callable(name_method) else None
    return _decode_name(value, f"tensor_{tensor_index}")


def _buffer_data(model: Any, buffer_index: int) -> np.ndarray:
    """Return a zero-copy uint8 view of one Circle buffer when possible."""

    buffer_count = _call_int(model, "BuffersLength")
    if buffer_index < 0 or buffer_index >= buffer_count:
        raise CircleWeightSparsityError(
            f"Invalid buffer index {buffer_index}; model has {buffer_count} buffers."
        )
    buffer = model.Buffers(buffer_index)
    data = _vector_as_numpy(buffer, "Data", np.dtype(np.uint8))
    if data.dtype != np.uint8:
        data = data.astype(np.uint8, copy=False)
    return np.ascontiguousarray(data).reshape(-1)


def _buffer_length(model: Any, buffer_index: int) -> int:
    """Return a Circle buffer byte length without decoding its payload."""

    buffer_count = _call_int(model, "BuffersLength")
    if buffer_index < 0 or buffer_index >= buffer_count:
        return 0
    buffer = model.Buffers(buffer_index)
    return _call_int(buffer, "DataLength")


def _is_constant_tensor(model: Any, tensor: Any) -> bool:
    """Return whether a Circle tensor owns a non-empty immutable buffer."""

    is_variable = bool(_call_int(tensor, "IsVariable", 0))
    if is_variable:
        return False
    buffer_index = _call_int(tensor, "Buffer", -1)
    return _buffer_length(model, buffer_index) > 0


def _has_quantization_metadata(tensor: Any) -> bool:
    """Return whether a Circle tensor contains affine quantization metadata."""

    quantization_method = getattr(tensor, "Quantization", None)
    if quantization_method is None or not callable(quantization_method):
        return False
    quantization = quantization_method()
    if quantization is None:
        return False
    return (
        _call_int(quantization, "ScaleLength") > 0
        or _call_int(quantization, "ZeroPointLength") > 0
    )


def _quantization_info(tensor: Any) -> QuantizationInfo:
    """Extract zero points and the quantized dimension from a Circle tensor."""

    quantization_method = getattr(tensor, "Quantization", None)
    quantization = (
        quantization_method()
        if quantization_method is not None and callable(quantization_method)
        else None
    )
    if quantization is None:
        return QuantizationInfo((), None, False)

    scales = _vector_as_numpy(quantization, "Scale", np.dtype(np.float64))
    zero_points = _vector_as_numpy(quantization, "ZeroPoint", np.dtype(np.int64))
    if scales.size == 0 and zero_points.size == 0:
        return QuantizationInfo((), None, False)

    if zero_points.size == 0:
        zero_points = np.asarray([0], dtype=np.int64)
    axis = _call_int(quantization, "QuantizedDimension", 0)
    return QuantizationInfo(
        tuple(int(value) for value in zero_points),
        axis,
        True,
        tuple(float(value) for value in scales),
    )


def _operator_inputs(operator: Any) -> tuple[int, ...]:
    """Return all tensor IDs consumed by one Circle operator."""

    values = _vector_as_numpy(operator, "Inputs", np.dtype(np.int64))
    return tuple(int(value) for value in values)


def _operator_outputs(operator: Any) -> tuple[int, ...]:
    """Return all tensor IDs produced by one Circle operator."""

    values = _vector_as_numpy(operator, "Outputs", np.dtype(np.int64))
    return tuple(int(value) for value in values)


def _subgraph_inputs(subgraph: Any) -> set[int]:
    """Return the tensor IDs declared as runtime subgraph inputs."""

    values = _vector_as_numpy(subgraph, "Inputs", np.dtype(np.int64))
    return {int(value) for value in values if int(value) >= 0}


def _build_producer_map(
    circle: Any,
    model: Any,
    subgraph: Any,
) -> dict[int, tuple[str, Any]]:
    """Map every produced tensor ID to its operator and symbolic name."""

    producers: dict[int, tuple[str, Any]] = {}
    for operator_index in range(_call_int(subgraph, "OperatorsLength")):
        operator = subgraph.Operators(operator_index)
        name = _operator_name(circle, model, operator)
        for tensor_id in _operator_outputs(operator):
            if tensor_id >= 0:
                producers[tensor_id] = (name, operator)
    return producers


def _resolve_constant_source(
    model: Any,
    subgraph: Any,
    tensor_id: int,
    producers: Mapping[int, tuple[str, Any]],
) -> int | None:
    """Follow zero-preserving transforms until a stored constant is found."""

    tensor_count = _call_int(subgraph, "TensorsLength")
    current = tensor_id
    visited: set[int] = set()

    while current not in visited:
        visited.add(current)
        if current < 0 or current >= tensor_count:
            return None
        tensor = subgraph.Tensors(current)
        if _is_constant_tensor(model, tensor):
            return current

        producer = producers.get(current)
        if producer is None:
            return None
        operator_name, operator = producer
        data_input_index = _PASSTHROUGH_DATA_INPUT.get(operator_name)
        if data_input_index is None:
            return None
        inputs = _operator_inputs(operator)
        if data_input_index >= len(inputs):
            return None
        current = inputs[data_input_index]

    return None


def _looks_like_weight_name(name: str) -> bool:
    """Return whether a tensor name strongly suggests a model weight."""

    normalized = name.lower()
    if any(marker in normalized for marker in _NAME_EXCLUSION_MARKERS):
        return False
    return any(marker in normalized for marker in _WEIGHT_NAME_MARKERS)


def _select_weight_tensor_ids(
    circle: Any,
    model: Any,
    subgraph: Any,
    selection: str,
) -> dict[int, set[str]]:
    """Select stored weight tensors and record how each tensor was recognized."""

    if selection not in _SELECTION_MODES:
        raise ValueError(
            f"Unsupported selection mode {selection!r}; "
            f"choose one of {_SELECTION_MODES}."
        )

    tensor_count = _call_int(subgraph, "TensorsLength")
    graph_inputs = _subgraph_inputs(subgraph)
    producers = _build_producer_map(circle, model, subgraph)
    selected: dict[int, set[str]] = defaultdict(set)

    if selection in ("auto", "operator-inputs"):
        for operator_index in range(_call_int(subgraph, "OperatorsLength")):
            operator = subgraph.Operators(operator_index)
            operator_name = _operator_name(circle, model, operator)
            input_indices = _WEIGHT_INPUT_INDICES.get(operator_name, ())
            inputs = _operator_inputs(operator)
            for input_index in input_indices:
                if input_index >= len(inputs):
                    continue
                source_id = _resolve_constant_source(
                    model,
                    subgraph,
                    inputs[input_index],
                    producers,
                )
                if source_id is None or source_id in graph_inputs:
                    continue
                selected[source_id].add(f"{operator_name}:input{input_index}")

    if selection == "auto":
        for tensor_index in range(tensor_count):
            if tensor_index in graph_inputs:
                continue
            tensor = subgraph.Tensors(tensor_index)
            if not _is_constant_tensor(model, tensor):
                continue
            name = _tensor_name(tensor, tensor_index)
            if _looks_like_weight_name(name):
                selected[tensor_index].add("name-fallback")

    if selection == "quantized-constants":
        for tensor_index in range(tensor_count):
            if tensor_index in graph_inputs:
                continue
            tensor = subgraph.Tensors(tensor_index)
            if not _is_constant_tensor(model, tensor):
                continue
            if _has_quantization_metadata(tensor):
                selected[tensor_index].add("quantized-constant")

    return selected


def _validate_shape_and_numel(shape: tuple[int, ...]) -> int:
    """Return the number of logical elements in a static tensor shape."""

    if not shape:
        return 1
    return math.prod(shape)


def _required_buffer_bytes(qdtype: str, numel: int) -> int:
    """Return the minimum number of bytes required by a logical tensor."""

    if qdtype in ("uint4", "int4"):
        return (numel + 1) // 2
    if qdtype == "bfloat16":
        return numel * 2
    numpy_dtype = _NUMPY_DTYPES.get(qdtype)
    if numpy_dtype is None:
        raise CircleWeightSparsityError(
            f"Unsupported Circle weight dtype {qdtype!r}. "
            "Affine integer, standard floating-point, and boolean weights "
            "are supported; MX formats are not yet decoded."
        )
    return numel * numpy_dtype.itemsize


def _semantic_zero_points(
    qdtype: str,
    quantization: QuantizationInfo,
) -> tuple[int, ...]:
    """Return integer codes that represent real-valued zero."""

    if qdtype.startswith("float") or qdtype == "bfloat16":
        return ()
    if quantization.is_affine and quantization.zero_points:
        return quantization.zero_points
    return (0,)


def _count_integer_chunk_zeros(
    values: np.ndarray,
    start_index: int,
    shape: tuple[int, ...],
    zero_points: tuple[int, ...],
    quantized_dimension: int | None,
) -> int:
    """Count integer semantic zeros in one flattened logical tensor chunk."""

    if len(zero_points) == 1:
        return int(np.count_nonzero(values == zero_points[0]))

    if not shape:
        raise CircleWeightSparsityError(
            "A scalar tensor cannot use multiple zero points."
        )
    axis = int(quantized_dimension or 0) % len(shape)
    channel_count = shape[axis]
    if len(zero_points) != channel_count:
        raise CircleWeightSparsityError(
            "Zero-point count does not match the quantized dimension: "
            f"shape={shape}, axis={axis}, zero_points={len(zero_points)}."
        )

    elements_after_axis = math.prod(shape[axis + 1 :])
    indices = np.arange(
        start_index,
        start_index + values.size,
        dtype=np.int64,
    )
    channel_indices = (indices // elements_after_axis) % channel_count
    targets = np.asarray(zero_points, dtype=np.int64)[channel_indices]
    return int(np.count_nonzero(values == targets))


def _count_packed_4bit_zeros(
    raw: np.ndarray,
    qdtype: str,
    shape: tuple[int, ...],
    numel: int,
    zero_points: tuple[int, ...],
    quantized_dimension: int | None,
    chunk_numel: int,
) -> int:
    """Count semantic zeros in low-nibble-first packed 4-bit data."""

    logical_chunk = max(2, chunk_numel)
    logical_chunk -= logical_chunk % 2
    zero_count = 0

    for start in range(0, numel, logical_chunk):
        end = min(start + logical_chunk, numel)
        byte_start = start // 2
        byte_end = (end + 1) // 2
        packed = raw[byte_start:byte_end]
        decoded = np.empty(packed.size * 2, dtype=np.uint8)
        decoded[0::2] = packed & np.uint8(0x0F)
        decoded[1::2] = packed >> np.uint8(4)
        values = decoded[: end - start]
        if qdtype == "int4":
            signed = values.astype(np.int8)
            signed[signed >= 8] -= 16
            values = signed
        zero_count += _count_integer_chunk_zeros(
            values,
            start,
            shape,
            zero_points,
            quantized_dimension,
        )
    return zero_count


def _count_dense_zeros(
    raw: np.ndarray,
    qdtype: str,
    shape: tuple[int, ...],
    numel: int,
    zero_points: tuple[int, ...],
    quantized_dimension: int | None,
    chunk_numel: int,
) -> int:
    """Count semantic zeros in a byte-aligned Circle tensor buffer."""

    if qdtype == "bfloat16":
        values = np.frombuffer(raw, dtype=np.dtype("<u2"), count=numel)
        zero_count = 0
        for start in range(0, numel, chunk_numel):
            chunk = values[start : start + chunk_numel]
            zero_count += int(np.count_nonzero((chunk & np.uint16(0x7FFF)) == 0))
        return zero_count

    numpy_dtype = _NUMPY_DTYPES.get(qdtype)
    if numpy_dtype is None:
        raise CircleWeightSparsityError(f"Unsupported dense dtype {qdtype!r}.")
    values = np.frombuffer(raw, dtype=numpy_dtype, count=numel)

    zero_count = 0
    is_float = np.issubdtype(numpy_dtype, np.floating)
    for start in range(0, numel, chunk_numel):
        chunk = values[start : start + chunk_numel]
        if is_float:
            zero_count += int(np.count_nonzero(chunk == 0))
        else:
            zero_count += _count_integer_chunk_zeros(
                chunk,
                start,
                shape,
                zero_points,
                quantized_dimension,
            )
    return zero_count


def _count_tensor_semantic_zeros(
    raw: np.ndarray,
    qdtype: str,
    shape: tuple[int, ...],
    quantization: QuantizationInfo,
    chunk_numel: int,
) -> tuple[int, int]:
    """Count semantic zeros in one stored Circle weight tensor."""

    if chunk_numel <= 0:
        raise ValueError(f"chunk_numel must be positive, got {chunk_numel}.")
    numel = _validate_shape_and_numel(shape)
    if numel == 0:
        return 0, 0

    required_bytes = _required_buffer_bytes(qdtype, numel)
    if raw.size < required_bytes:
        raise CircleWeightSparsityError(
            f"Buffer is too small for {qdtype} tensor {shape}: "
            f"required={required_bytes}, available={raw.size}."
        )
    raw = raw[:required_bytes]
    zero_points = _semantic_zero_points(qdtype, quantization)

    if qdtype in ("uint4", "int4"):
        zero_count = _count_packed_4bit_zeros(
            raw,
            qdtype,
            shape,
            numel,
            zero_points,
            quantization.quantized_dimension,
            chunk_numel,
        )
    else:
        zero_count = _count_dense_zeros(
            raw,
            qdtype,
            shape,
            numel,
            zero_points,
            quantization.quantized_dimension,
            chunk_numel,
        )
    return zero_count, numel


def _tensor_fingerprint(
    raw: np.ndarray,
    qdtype: str,
    shape: tuple[int, ...],
    quantization: QuantizationInfo,
) -> str:
    """Hash a stored tensor and its zero-relevant metadata."""

    digest = hashlib.sha256()
    digest.update(qdtype.encode("utf-8"))
    digest.update(repr(shape).encode("utf-8"))
    digest.update(repr(quantization.scales).encode("utf-8"))
    digest.update(repr(quantization.zero_points).encode("utf-8"))
    digest.update(repr(quantization.quantized_dimension).encode("utf-8"))
    numel = _validate_shape_and_numel(shape)
    required_bytes = _required_buffer_bytes(qdtype, numel)
    digest.update(memoryview(raw[:required_bytes]))
    return digest.hexdigest()


def _analyze_root_model(
    model: Any,
    circle: Any,
    source: str,
    *,
    selection: str,
    chunk_numel: int,
    strict: bool,
    fingerprint: bool,
) -> tuple[list[CircleWeightTensorStats], int, list[str]]:
    """Analyze all subgraphs in one parsed Circle root model."""

    stats: list[CircleWeightTensorStats] = []
    duplicate_count = 0
    skipped_messages: list[str] = []
    seen_model_buffers: set[int] = set()

    for subgraph_index in range(_call_int(model, "SubgraphsLength")):
        subgraph = model.Subgraphs(subgraph_index)
        selected = _select_weight_tensor_ids(
            circle,
            model,
            subgraph,
            selection,
        )
        for tensor_index in sorted(selected):
            tensor = subgraph.Tensors(tensor_index)
            tensor_name = _tensor_name(tensor, tensor_index)
            try:
                shape = _tensor_shape(tensor)
                tensor_type = _call_int(tensor, "Type", -1)
                qdtype = _tensor_dtype_name(circle, tensor_type)
                buffer_index = _call_int(tensor, "Buffer", -1)
                quantization = _quantization_info(tensor)
                if buffer_index in seen_model_buffers:
                    duplicate_count += 1
                    continue

                raw = _buffer_data(model, buffer_index)
                zero_count, numel = _count_tensor_semantic_zeros(
                    raw,
                    qdtype,
                    shape,
                    quantization,
                    chunk_numel,
                )
                tensor_digest = (
                    _tensor_fingerprint(raw, qdtype, shape, quantization)
                    if fingerprint
                    else None
                )
                stats.append(
                    CircleWeightTensorStats(
                        source=source,
                        subgraph_index=subgraph_index,
                        tensor_index=tensor_index,
                        tensor_name=tensor_name,
                        buffer_index=buffer_index,
                        qdtype=qdtype,
                        shape=shape,
                        zero_count=zero_count,
                        numel=numel,
                        roles=tuple(sorted(selected[tensor_index])),
                        fingerprint=tensor_digest,
                    )
                )
                seen_model_buffers.add(buffer_index)
            except (CircleWeightSparsityError, ValueError, TypeError) as exc:
                message = (
                    f"{source}:subgraph={subgraph_index}:tensor={tensor_index}"
                    f"({tensor_name}): {exc}"
                )
                if strict:
                    raise CircleWeightSparsityError(message) from exc
                skipped_messages.append(message)

    return stats, duplicate_count, skipped_messages


def analyze_circle_binary(
    circle_binary: Any,
    *,
    source: str = "<memory>",
    selection: str = "auto",
    chunk_numel: int = _DEFAULT_CHUNK_NUMEL,
    strict: bool = True,
    fingerprint: bool = False,
    circle_module: Any | None = None,
) -> tuple[list[CircleWeightTensorStats], int, list[str]]:
    """Analyze weight tensors from an in-memory Circle FlatBuffer."""

    circle = circle_module or _load_circle_schema()
    model = circle.Model.Model.GetRootAsModel(circle_binary, 0)
    return _analyze_root_model(
        model,
        circle,
        source,
        selection=selection,
        chunk_numel=chunk_numel,
        strict=strict,
        fingerprint=fingerprint,
    )


def analyze_circle_file(
    path: str | Path,
    *,
    selection: str = "auto",
    chunk_numel: int = _DEFAULT_CHUNK_NUMEL,
    strict: bool = True,
    fingerprint: bool = False,
) -> tuple[list[CircleWeightTensorStats], int, list[str]]:
    """Memory-map and analyze one Circle model file."""

    circle_path = Path(path)
    if not circle_path.is_file():
        raise CircleWeightSparsityError(f"Circle file does not exist: {circle_path}")
    if circle_path.stat().st_size == 0:
        raise CircleWeightSparsityError(f"Circle file is empty: {circle_path}")

    circle = _load_circle_schema()
    with circle_path.open("rb") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            model = circle.Model.Model.GetRootAsModel(mapped, 0)
            result = _analyze_root_model(
                model,
                circle,
                str(circle_path),
                selection=selection,
                chunk_numel=chunk_numel,
                strict=strict,
                fingerprint=fingerprint,
            )
            del model
            return result


def discover_circle_files(
    inputs: Iterable[str | Path],
    *,
    recursive: bool = False,
) -> list[Path]:
    """Resolve file and directory inputs into a sorted unique Circle file list."""

    discovered: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        if path.is_file():
            discovered.append(path.resolve())
            continue
        if path.is_dir():
            iterator = path.rglob("*.circle") if recursive else path.glob("*.circle")
            discovered.extend(candidate.resolve() for candidate in iterator)
            continue
        raise CircleWeightSparsityError(f"Input path does not exist: {path}")

    unique = sorted(dict.fromkeys(discovered))
    if not unique:
        raise CircleWeightSparsityError("No Circle files were found.")
    return unique


def _format_qdtype(stats: Sequence[CircleWeightTensorStats]) -> str:
    """Return a single dtype or a stable compact mixed-dtype label."""

    qdtypes = sorted(
        {item.qdtype for item in stats},
        key=lambda value: (_DTYPE_ORDER.get(value, 10_000), value),
    )
    if len(qdtypes) == 1:
        return qdtypes[0]
    return f"mixed ({', '.join(qdtypes)})"


def _deduplicate_across_files(
    stats: Sequence[CircleWeightTensorStats],
) -> tuple[list[CircleWeightTensorStats], int]:
    """Deduplicate matching tensor names and payloads across Circle files."""

    unique: list[CircleWeightTensorStats] = []
    seen: set[tuple[str, str, tuple[int, ...], str]] = set()
    duplicates = 0
    for item in stats:
        if item.fingerprint is None:
            raise CircleWeightSparsityError(
                "Cross-file deduplication requires tensor fingerprints."
            )
        key = (
            item.tensor_name,
            item.qdtype,
            item.shape,
            item.fingerprint,
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(item)
    return unique, duplicates


def aggregate_circle_weight_stats(
    stats: Sequence[CircleWeightTensorStats],
    *,
    source_count: int,
    duplicate_tensor_count: int = 0,
    skipped_messages: Sequence[str] = (),
) -> CircleWeightSparsityReport:
    """Aggregate tensor statistics into the single model-level report row."""

    if not stats:
        details = ""
        if skipped_messages:
            details = "\n" + "\n".join(f"  - {item}" for item in skipped_messages)
        raise CircleWeightSparsityError(
            "No supported Circle weight tensors were selected." + details
        )

    zero_count = sum(item.zero_count for item in stats)
    numel = sum(item.numel for item in stats)
    if numel == 0:
        raise CircleWeightSparsityError("Selected Circle weights contain no elements.")
    row = CircleWeightSparsityRow(
        scope=_SCOPE_NAME,
        qdtype=_format_qdtype(stats),
        sparsity_pct=100.0 * zero_count / numel,
    )
    return CircleWeightSparsityReport(
        row=row,
        source_count=source_count,
        tensor_count=len(stats),
        zero_count=zero_count,
        numel=numel,
        duplicate_tensor_count=duplicate_tensor_count,
        skipped_tensor_count=len(skipped_messages),
        skipped_messages=tuple(skipped_messages),
    )


def measure_circle_weight_sparsity(
    inputs: Iterable[str | Path],
    *,
    recursive: bool = False,
    selection: str = "auto",
    chunk_numel: int = _DEFAULT_CHUNK_NUMEL,
    strict: bool = True,
    deduplicate_across_files: bool = False,
) -> CircleWeightSparsityReport:
    """Measure model-level Circle weight sparsity across one or more files."""

    files = discover_circle_files(inputs, recursive=recursive)
    all_stats: list[CircleWeightTensorStats] = []
    duplicate_count = 0
    skipped_messages: list[str] = []

    for path in files:
        file_stats, file_duplicates, file_skipped = analyze_circle_file(
            path,
            selection=selection,
            chunk_numel=chunk_numel,
            strict=strict,
            fingerprint=deduplicate_across_files,
        )
        all_stats.extend(file_stats)
        duplicate_count += file_duplicates
        skipped_messages.extend(file_skipped)

    if deduplicate_across_files:
        all_stats, cross_file_duplicates = _deduplicate_across_files(all_stats)
        duplicate_count += cross_file_duplicates

    return aggregate_circle_weight_stats(
        all_stats,
        source_count=len(files),
        duplicate_tensor_count=duplicate_count,
        skipped_messages=skipped_messages,
    )


def render_markdown(
    report: CircleWeightSparsityReport,
    *,
    precision: int = 6,
) -> str:
    """Render the model-level result as a three-column Markdown table."""

    if precision < 0:
        raise ValueError(f"precision must be non-negative, got {precision}.")
    row = report.row
    return "\n".join(
        (
            "| Scope | Qdtype | Sparsity (%) |",
            "|---|---|---:|",
            f"| {row.scope} | {row.qdtype} | {row.sparsity_pct:.{precision}f} |",
        )
    )


def render_csv(
    report: CircleWeightSparsityReport,
    *,
    precision: int = 6,
) -> str:
    """Render the model-level result as a three-column CSV document."""

    if precision < 0:
        raise ValueError(f"precision must be non-negative, got {precision}.")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("Scope", "Qdtype", "Sparsity (%)"))
    writer.writerow(
        (
            report.row.scope,
            report.row.qdtype,
            f"{report.row.sparsity_pct:.{precision}f}",
        )
    )
    return output.getvalue()


def render_json(
    report: CircleWeightSparsityReport,
    *,
    precision: int = 6,
) -> str:
    """Render the model-level result as a one-row JSON array."""

    if precision < 0:
        raise ValueError(f"precision must be non-negative, got {precision}.")
    payload = [
        {
            "Scope": report.row.scope,
            "Qdtype": report.row.qdtype,
            "Sparsity (%)": round(report.row.sparsity_pct, precision),
        }
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render_report(
    report: CircleWeightSparsityReport,
    output_format: str,
    *,
    precision: int = 6,
) -> str:
    """Render a report in Markdown, CSV, or JSON format."""

    normalized = output_format.lower()
    if normalized in ("markdown", "md"):
        return render_markdown(report, precision=precision) + "\n"
    if normalized == "csv":
        return render_csv(report, precision=precision)
    if normalized == "json":
        return render_json(report, precision=precision)
    raise ValueError(f"Unsupported output format {output_format!r}.")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Circle weight sparsity analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure model-level semantic weight sparsity from Circle files. "
            "Affine integer zero is detected by qcode == zero_point."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Circle file paths or directories containing .circle files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively for .circle files.",
    )
    parser.add_argument(
        "--selection",
        choices=_SELECTION_MODES,
        default="auto",
        help=(
            "Weight selection policy. 'auto' uses known operator input roles and "
            "a conservative tensor-name fallback."
        ),
    )
    parser.add_argument(
        "--chunk-numel",
        type=int,
        default=_DEFAULT_CHUNK_NUMEL,
        help="Maximum logical elements decoded per temporary chunk.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail instead of skipping malformed or unsupported selected tensors.",
    )
    parser.add_argument(
        "--deduplicate-across-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deduplicate tensors with the same name, dtype, shape, and payload "
            "across multiple input files."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        dest="output_format",
        help="Output format for the single model-level result row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the report to this file instead of standard output.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places in the sparsity percentage.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print internal file, tensor, and element counters to standard error.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Circle weight sparsity command-line tool."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        report = measure_circle_weight_sparsity(
            args.inputs,
            recursive=args.recursive,
            selection=args.selection,
            chunk_numel=args.chunk_numel,
            strict=args.strict,
            deduplicate_across_files=args.deduplicate_across_files,
        )
        rendered = render_report(
            report,
            args.output_format,
            precision=args.precision,
        )
    except (CircleWeightSparsityError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if report.skipped_tensor_count:
        print(
            f"warning: skipped {report.skipped_tensor_count} selected tensor(s)",
            file=sys.stderr,
        )
        if args.verbose:
            for message in report.skipped_messages:
                print(f"  - {message}", file=sys.stderr)

    if args.verbose:
        print(
            "files={files} tensors={tensors} elements={elements} zeros={zeros} "
            "duplicates={duplicates} skipped={skipped}".format(
                files=report.source_count,
                tensors=report.tensor_count,
                elements=report.numel,
                zeros=report.zero_count,
                duplicates=report.duplicate_tensor_count,
                skipped=report.skipped_tensor_count,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
