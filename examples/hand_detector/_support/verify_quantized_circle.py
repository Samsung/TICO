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

"""Validate a quantized hand-detector Circle model without running inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from examples.hand_detector._support.tflite_flatbuffer import FlatBufferReader


ADD = 0
CONCATENATION = 2
CONV_2D = 3
DEPTHWISE_CONV_2D = 4
DEQUANTIZE = 6
MAX_POOL_2D = 17
RESHAPE = 22
RESIZE_BILINEAR = 23
PAD = 34
TRANSPOSE = 39
PRELU = 54
SLICE = 65
QUANTIZE = 114

PADDING_SAME = 0

TENSOR_FLOAT32 = 0
TENSOR_INT32 = 2
TENSOR_UINT8 = 3
TENSOR_INT64 = 4
TENSOR_INT16 = 7

TENSOR_TYPE_NAMES = {
    TENSOR_FLOAT32: "FLOAT32",
    TENSOR_INT32: "INT32",
    TENSOR_UINT8: "UINT8",
    TENSOR_INT64: "INT64",
    TENSOR_INT16: "INT16",
}

# This table captures target-backend constraints for data operators.
# MaxPool2D and ResizeBilinear intentionally allow distinct per-tensor
# affine qparams for their input and output tensors.
_DATA_OPERATOR_REQUIRES_SHARED_QPARAMS = {
    MAX_POOL_2D: False,
    RESHAPE: True,
    RESIZE_BILINEAR: False,
    PAD: True,
    TRANSPOSE: True,
    SLICE: True,
}

OPERATOR_NAMES = {
    ADD: "ADD",
    CONCATENATION: "CONCATENATION",
    CONV_2D: "CONV_2D",
    DEPTHWISE_CONV_2D: "DEPTHWISE_CONV_2D",
    DEQUANTIZE: "DEQUANTIZE",
    MAX_POOL_2D: "MAX_POOL_2D",
    RESHAPE: "RESHAPE",
    RESIZE_BILINEAR: "RESIZE_BILINEAR",
    PAD: "PAD",
    TRANSPOSE: "TRANSPOSE",
    PRELU: "PRELU",
    SLICE: "SLICE",
    QUANTIZE: "QUANTIZE",
}


@dataclass(frozen=True)
class QuantizationInfo:
    """Describe one tensor's affine quantization metadata."""

    scales: tuple[float, ...]
    zero_points: tuple[int, ...]
    quantized_dimension: int


@dataclass(frozen=True)
class CircleTensorInfo:
    """Describe the Circle tensor fields needed by the verifier."""

    shape: tuple[int, ...]
    tensor_type: int
    buffer_index: int
    name: str
    quantization: QuantizationInfo | None


@dataclass(frozen=True)
class CircleOperatorInfo:
    """Describe the Circle operator fields needed by the verifier."""

    builtin_code: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    options_table: int | None


def _parse_operator_codes(reader: FlatBufferReader, root: int) -> list[int]:
    """Decode builtin operator codes from a Circle model."""
    result: list[int] = []
    for table in reader.vector_tables(root, 1):
        deprecated = reader.scalar_i8(table, 0, 0)
        builtin = reader.scalar_i32(table, 3, deprecated)
        if builtin == 0 and deprecated != 0:
            builtin = deprecated
        result.append(builtin)
    return result


def _parse_quantization(
    reader: FlatBufferReader,
    tensor_table: int,
) -> QuantizationInfo | None:
    """Decode one optional Circle QuantizationParameters table."""
    table = reader.table(tensor_table, 4)
    if table is None:
        return None
    scales = tuple(reader.vector_f32(table, 2))
    zero_points = tuple(reader.vector_i64(table, 3))
    if not scales and not zero_points:
        return None
    return QuantizationInfo(
        scales=scales,
        zero_points=zero_points,
        quantized_dimension=reader.scalar_i32(table, 6, 0),
    )


def _parse_tensors(
    reader: FlatBufferReader,
    subgraph: int,
) -> list[CircleTensorInfo]:
    """Decode tensor shapes, types, names, buffers, and quantization metadata."""
    result: list[CircleTensorInfo] = []
    for table in reader.vector_tables(subgraph, 0):
        result.append(
            CircleTensorInfo(
                shape=tuple(reader.vector_i32(table, 0)),
                tensor_type=reader.scalar_i8(table, 1, 0),
                buffer_index=reader.scalar_u32(table, 2, 0),
                name=reader.string(table, 3) or "",
                quantization=_parse_quantization(reader, table),
            )
        )
    return result


def _parse_operators(
    reader: FlatBufferReader,
    subgraph: int,
    operator_codes: list[int],
) -> list[CircleOperatorInfo]:
    """Decode operator inputs, outputs, builtin codes, and option tables."""
    result: list[CircleOperatorInfo] = []
    for table in reader.vector_tables(subgraph, 3):
        opcode_index = reader.scalar_u32(table, 0, 0)
        result.append(
            CircleOperatorInfo(
                builtin_code=operator_codes[opcode_index],
                inputs=tuple(reader.vector_i32(table, 1)),
                outputs=tuple(reader.vector_i32(table, 2)),
                options_table=reader.table(table, 4),
            )
        )
    return result


def _expected_tensor_type(bit_width: int) -> int:
    """Return the Circle tensor type required for one configured bit width."""
    if bit_width == 8:
        return TENSOR_UINT8
    if bit_width == 16:
        return TENSOR_INT16
    raise ValueError(f"Unsupported bit width: {bit_width}")


def _expected_bias_type(bit_width: int) -> int:
    """Return the accumulator-backed Circle bias type for one bit width."""
    if bit_width == 8:
        return TENSOR_INT32
    if bit_width == 16:
        return TENSOR_INT64
    raise ValueError(f"Unsupported bit width: {bit_width}")


def _type_name(tensor_type: int) -> str:
    """Return a readable Circle tensor type name."""
    return TENSOR_TYPE_NAMES.get(tensor_type, str(tensor_type))


def _require_quantized_tensor(
    tensor: CircleTensorInfo,
    expected_type: int,
    *,
    context: str,
    expected_qparam_count: int | None = None,
    expected_axis: int | None = None,
) -> None:
    """Validate one tensor's integer type and affine quantization metadata."""
    if tensor.tensor_type != expected_type:
        raise RuntimeError(
            f"{context} must be {_type_name(expected_type)}, but "
            f"{tensor.name!r} is {_type_name(tensor.tensor_type)}."
        )
    quantization = tensor.quantization
    if quantization is None:
        raise RuntimeError(f"{context} {tensor.name!r} has no quantization metadata.")
    if not quantization.scales:
        raise RuntimeError(f"{context} {tensor.name!r} has no quantization scale.")
    if not quantization.zero_points:
        raise RuntimeError(f"{context} {tensor.name!r} has no zero point.")
    if len(quantization.scales) != len(quantization.zero_points):
        raise RuntimeError(
            f"{context} {tensor.name!r} has mismatched scale and zero-point "
            "vector lengths."
        )
    if expected_type == TENSOR_INT16 and any(
        zero_point != 0 for zero_point in quantization.zero_points
    ):
        raise RuntimeError(
            f"{context} {tensor.name!r} must use zero point 0 for symmetric INT16."
        )
    if expected_qparam_count is not None:
        if len(quantization.scales) != expected_qparam_count:
            raise RuntimeError(
                f"{context} {tensor.name!r} must contain "
                f"{expected_qparam_count} qparams, but contains "
                f"{len(quantization.scales)}."
            )
    if expected_axis is not None:
        if quantization.quantized_dimension != expected_axis:
            raise RuntimeError(
                f"{context} {tensor.name!r} must use quantized dimension "
                f"{expected_axis}, but uses {quantization.quantized_dimension}."
            )


def _require_same_qparams(
    tensors: Iterable[CircleTensorInfo],
    *,
    context: str,
) -> None:
    """Require every tensor to use the exact same affine quantization parameters."""
    values = list(tensors)
    if len(values) < 2:
        return
    reference = values[0].quantization
    if reference is None:
        raise RuntimeError(f"{context} reference tensor has no quantization metadata.")
    for tensor in values[1:]:
        if tensor.quantization != reference:
            raise RuntimeError(
                f"{context} requires identical scale and zero point, but "
                f"{values[0].name!r} and {tensor.name!r} differ."
            )


def _require_data_operator_qparams(
    builtin_code: int,
    tensors: Iterable[CircleTensorInfo],
    *,
    context: str,
) -> None:
    """Enforce shared qparams only when required by the target NPU."""
    if _DATA_OPERATOR_REQUIRES_SHARED_QPARAMS.get(builtin_code, False):
        _require_same_qparams(tensors, context=context)


def _normalize_axis(axis: int, rank: int, *, context: str) -> int:
    """Normalize one possibly negative axis."""
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise RuntimeError(f"{context} axis {axis} is invalid for rank {rank}.")
    return normalized


def _is_width_concatenation(
    reader: FlatBufferReader,
    tensors: list[CircleTensorInfo],
    operator: CircleOperatorInfo,
    *,
    context: str,
) -> bool:
    """Return whether one channels-last Concat joins width."""
    if operator.options_table is None:
        raise RuntimeError(f"{context} has no builtin options.")
    if not operator.outputs:
        raise RuntimeError(f"{context} has no output tensor.")
    output = tensors[operator.outputs[0]]
    rank = len(output.shape)
    if rank < 2:
        raise RuntimeError(f"{context} output rank {rank} has no width axis.")
    axis = reader.scalar_i32(operator.options_table, 0, 0)
    return _normalize_axis(axis, rank, context=context) == rank - 2


def _require_supported_quantize_transition(
    input_tensor: CircleTensorInfo,
    output_tensor: CircleTensorInfo,
    *,
    context: str,
) -> tuple[int, int]:
    """Require a backend-supported Q8/Q16 dtype transition."""
    allowed_types = (TENSOR_UINT8, TENSOR_INT16)
    for tensor in (input_tensor, output_tensor):
        if tensor.tensor_type not in allowed_types:
            raise RuntimeError(
                f"{context} supports only UINT8 and INT16, but "
                f"{tensor.name!r} is {_type_name(tensor.tensor_type)}."
            )
        _require_quantized_tensor(
            tensor,
            tensor.tensor_type,
            context=context,
            expected_qparam_count=1,
        )
    transition = (
        input_tensor.tensor_type,
        output_tensor.tensor_type,
    )
    allowed = {
        (TENSOR_UINT8, TENSOR_INT16),
        (TENSOR_INT16, TENSOR_UINT8),
    }
    if transition not in allowed:
        raise RuntimeError(
            f"{context} supports only Q8->Q16 and Q16->Q8, "
            f"found {_type_name(transition[0])}->"
            f"{_type_name(transition[1])}."
        )
    return transition


def _tensor_indices(indices: Iterable[int]) -> list[int]:
    """Drop optional negative tensor indices from an operator input list."""
    return [index for index in indices if index >= 0]


def _require_per_tensor_data(
    tensors: list[CircleTensorInfo],
    indices: Iterable[int],
    expected_type: int,
    *,
    context: str,
) -> list[CircleTensorInfo]:
    """Validate and return data tensors with one affine qparam each."""
    result = [tensors[index] for index in _tensor_indices(indices)]
    for tensor in result:
        _require_quantized_tensor(
            tensor,
            expected_type,
            context=context,
            expected_qparam_count=1,
        )
    return result


def verify_quantized_circle(
    path: str | Path,
    bit_width: int,
    *,
    expected_resize_count: int = 2,
    expected_same_padding_conv_count: int = 33,
    expected_pad_count: int = 3,
    expected_max_pool_count: int = 4,
    expected_concat_count: int = 2,
) -> dict[str, Any]:
    """Validate graph I/O, operator tensors, biases, and resize options."""
    circle_path = Path(path)
    data = circle_path.read_bytes()
    if len(data) < 8 or data[4:8] != b"CIR0":
        raise ValueError(f"{circle_path} does not contain a Circle CIR0 identifier")

    reader = FlatBufferReader(data)
    root = reader.root_table()
    operator_codes = _parse_operator_codes(reader, root)
    subgraphs = reader.vector_tables(root, 2)
    if len(subgraphs) != 1:
        raise RuntimeError(f"Expected one subgraph, found {len(subgraphs)}.")
    subgraph = subgraphs[0]
    tensors = _parse_tensors(reader, subgraph)
    operators = _parse_operators(reader, subgraph, operator_codes)
    inputs = tuple(reader.vector_i32(subgraph, 1))
    outputs = tuple(reader.vector_i32(subgraph, 2))

    expected_type = _expected_tensor_type(bit_width)
    expected_bias_type = _expected_bias_type(bit_width)
    _require_per_tensor_data(
        tensors,
        inputs,
        expected_type,
        context="Graph input",
    )
    _require_per_tensor_data(
        tensors,
        outputs,
        expected_type,
        context="Graph output",
    )

    counts = {name: 0 for name in OPERATOR_NAMES.values()}
    resize_options: list[tuple[bool, bool]] = []
    conv_weight_count = 0
    depthwise_weight_count = 0
    bias_count = 0
    prelu_slope_count = 0
    same_padding_conv_count = 0
    distinct_max_pool_qparam_count = 0

    for operator_index, operator in enumerate(operators):
        name = OPERATOR_NAMES.get(
            operator.builtin_code,
            f"BUILTIN_{operator.builtin_code}",
        )
        if name in counts:
            counts[name] += 1
        if operator.builtin_code == DEQUANTIZE:
            raise RuntimeError(
                f"Operator {operator_index} is DEQUANTIZE; the graph is not "
                "fully integer-quantized."
            )

        if operator.builtin_code in (CONV_2D, DEPTHWISE_CONV_2D):
            if len(operator.inputs) < 2 or not operator.outputs:
                raise RuntimeError(f"{name} has incomplete tensor connections.")
            if operator.options_table is None:
                raise RuntimeError(f"{name} does not contain builtin options.")
            # Padding.SAME is enum value 0 and is also the FlatBuffer field
            # default. FlatBuffers may therefore omit the field entirely when
            # the serialized value is SAME. Use the schema default instead of
            # a sentinel so an omitted field is decoded as SAME.
            padding = reader.scalar_i8(operator.options_table, 0, PADDING_SAME)
            if padding == PADDING_SAME:
                same_padding_conv_count += 1
            _require_per_tensor_data(
                tensors,
                [operator.inputs[0], operator.outputs[0]],
                expected_type,
                context=f"{name} activation",
            )

            weight = tensors[operator.inputs[1]]
            if operator.builtin_code == CONV_2D:
                expected_axis = 0
                expected_channels = weight.shape[0]
                conv_weight_count += 1
            else:
                expected_axis = 3
                expected_channels = weight.shape[3]
                depthwise_weight_count += 1
            _require_quantized_tensor(
                weight,
                expected_type,
                context=f"{name} weight",
                expected_qparam_count=expected_channels,
                expected_axis=expected_axis,
            )

            if len(operator.inputs) >= 3 and operator.inputs[2] >= 0:
                bias = tensors[operator.inputs[2]]
                _require_quantized_tensor(
                    bias,
                    expected_bias_type,
                    context=f"{name} bias",
                    expected_qparam_count=expected_channels,
                    expected_axis=0,
                )
                bias_count += 1
            continue

        if operator.builtin_code == PRELU:
            if len(operator.inputs) != 2 or not operator.outputs:
                raise RuntimeError(
                    "PRELU must contain input, slope, and output tensors."
                )
            activation_tensors = _require_per_tensor_data(
                tensors,
                [operator.inputs[0], operator.outputs[0]],
                expected_type,
                context="PRELU activation",
            )
            slope = tensors[operator.inputs[1]]
            if len(slope.shape) != 1:
                raise RuntimeError(
                    f"PRELU slope {slope.name!r} must be rank 1, but has "
                    f"shape {slope.shape}."
                )
            expected_channels = activation_tensors[0].shape[-1]
            if slope.shape[0] != expected_channels:
                raise RuntimeError(
                    f"PRELU slope {slope.name!r} contains {slope.shape[0]} "
                    f"values, but the channel-last input contains "
                    f"{expected_channels} channels."
                )
            _require_quantized_tensor(
                slope,
                expected_type,
                context="PRELU slope",
                expected_qparam_count=expected_channels,
                expected_axis=0,
            )
            prelu_slope_count += 1
            continue

        if operator.builtin_code == ADD:
            _require_per_tensor_data(
                tensors,
                [*operator.inputs, *operator.outputs],
                expected_type,
                context="ADD tensor",
            )
            continue

        if operator.builtin_code == CONCATENATION:
            connected = _require_per_tensor_data(
                tensors,
                [*operator.inputs, *operator.outputs],
                expected_type,
                context="CONCATENATION tensor",
            )
            if not _is_width_concatenation(
                reader,
                tensors,
                operator,
                context=f"CONCATENATION@{operator_index}",
            ):
                _require_same_qparams(
                    connected,
                    context="CONCATENATION",
                )
            continue

        if operator.builtin_code in _DATA_OPERATOR_REQUIRES_SHARED_QPARAMS:
            if not operator.inputs or not operator.outputs:
                raise RuntimeError(f"{name} has incomplete tensor connections.")
            connected = _require_per_tensor_data(
                tensors,
                [operator.inputs[0], operator.outputs[0]],
                expected_type,
                context=f"{name} data tensor",
            )
            _require_data_operator_qparams(
                operator.builtin_code,
                connected,
                context=name,
            )
            if (
                operator.builtin_code == MAX_POOL_2D
                and connected[0].quantization != connected[1].quantization
            ):
                distinct_max_pool_qparam_count += 1

            if operator.builtin_code == RESIZE_BILINEAR:
                if operator.options_table is None:
                    raise RuntimeError(
                        "RESIZE_BILINEAR does not contain builtin options."
                    )
                # The schema keeps deprecated new_height/new_width in slots 0/1.
                align_corners = reader.scalar_bool(
                    operator.options_table,
                    2,
                    False,
                )
                half_pixel_centers = reader.scalar_bool(
                    operator.options_table,
                    3,
                    False,
                )
                resize_options.append((align_corners, half_pixel_centers))
            continue

        if operator.builtin_code == QUANTIZE:
            if not operator.inputs or not operator.outputs:
                raise RuntimeError("QUANTIZE has incomplete tensor connections.")
            _require_supported_quantize_transition(
                tensors[operator.inputs[0]],
                tensors[operator.outputs[0]],
                context=f"QUANTIZE@{operator_index}",
            )

    if counts["CONV_2D"] == 0:
        raise RuntimeError("The Circle graph does not contain CONV_2D.")
    if counts["DEPTHWISE_CONV_2D"] == 0:
        raise RuntimeError("The Circle graph does not contain DEPTHWISE_CONV_2D.")
    if counts["PRELU"] == 0:
        raise RuntimeError("The Circle graph does not contain PRELU.")
    if same_padding_conv_count != expected_same_padding_conv_count:
        raise RuntimeError(
            f"Expected {expected_same_padding_conv_count} SAME-padded convolution "
            f"operators, found {same_padding_conv_count}."
        )
    if counts["PAD"] != expected_pad_count:
        raise RuntimeError(
            f"Expected {expected_pad_count} explicit PAD operators, "
            f"found {counts['PAD']}."
        )
    if counts["MAX_POOL_2D"] != expected_max_pool_count:
        raise RuntimeError(
            f"Expected {expected_max_pool_count} MAX_POOL_2D operators, "
            f"found {counts['MAX_POOL_2D']}."
        )
    if counts["CONCATENATION"] != expected_concat_count:
        raise RuntimeError(
            f"Expected {expected_concat_count} CONCATENATION operators, "
            f"found {counts['CONCATENATION']}."
        )
    if len(resize_options) != expected_resize_count:
        raise RuntimeError(
            f"Expected {expected_resize_count} RESIZE_BILINEAR operators, "
            f"found {len(resize_options)}."
        )
    expected_options = [(False, True)] * expected_resize_count
    if resize_options != expected_options:
        raise RuntimeError(
            f"Expected ResizeBilinear options {expected_options}, "
            f"found {resize_options}."
        )

    quantized_tensor_count = sum(
        tensor.tensor_type == expected_type and tensor.quantization is not None
        for tensor in tensors
    )
    return {
        "path": str(circle_path),
        "size_bytes": len(data),
        "bit_width": bit_width,
        "tensor_type": _type_name(expected_type),
        "graph_inputs": len(inputs),
        "graph_outputs": len(outputs),
        "quantized_tensors": quantized_tensor_count,
        "conv_weights": conv_weight_count,
        "depthwise_weights": depthwise_weight_count,
        "quantized_biases": bias_count,
        "prelu_slopes": prelu_slope_count,
        "same_padding_convolutions": same_padding_conv_count,
        "max_pool_distinct_qparams": distinct_max_pool_qparam_count,
        "operator_counts": counts,
        "resize_options": [list(value) for value in resize_options],
        "input_tensors": [asdict(tensors[index]) for index in inputs],
        "output_tensors": [asdict(tensors[index]) for index in outputs],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circle", type=Path)
    parser.add_argument("--bits", type=int, required=True, choices=[8, 16])
    parser.add_argument("--expected-resize-count", type=int, default=2)
    parser.add_argument("--expected-same-padding-conv-count", type=int, default=33)
    parser.add_argument("--expected-pad-count", type=int, default=3)
    parser.add_argument("--expected-max-pool-count", type=int, default=4)
    parser.add_argument("--expected-concat-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Validate one quantized Circle model and print its summary."""
    args = parse_args()
    summary = verify_quantized_circle(
        args.circle,
        args.bits,
        expected_resize_count=args.expected_resize_count,
        expected_same_padding_conv_count=args.expected_same_padding_conv_count,
        expected_pad_count=args.expected_pad_count,
        expected_max_pool_count=args.expected_max_pool_count,
        expected_concat_count=args.expected_concat_count,
    )
    print(
        f"Verified {summary['tensor_type']} Circle model with "
        f"{summary['quantized_tensors']} quantized tensors and "
        f"{summary['operator_counts']['RESIZE_BILINEAR']} "
        f"RESIZE_BILINEAR operators: {args.circle}"
    )
    print(
        "MAX_POOL_2D operators with distinct input/output qparams: "
        f"{summary['max_pool_distinct_qparams']}"
    )


if __name__ == "__main__":
    main()
