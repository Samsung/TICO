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

"""Validate the NHWC input ABI and removed Transpose round trips in Circle."""

from __future__ import annotations

import argparse
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from examples.hand_detector._support.tflite_flatbuffer import FlatBufferReader


ADD = 0
TRANSPOSE = 39
EXPECTED_INPUT_SHAPE = (1, 192, 192, 3)


@dataclass(frozen=True)
class OperatorInfo:
    """Describe the Circle operator fields needed by the layout verifier."""

    builtin_code: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class LayoutVerificationSummary:
    """Describe the observable layout properties of one Circle model."""

    path: str
    size_bytes: int
    input_shapes: tuple[tuple[int, ...], ...]
    transpose_count: int
    add_count: int
    consecutive_inverse_transpose_pairs: int
    transpose_add_round_trips: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the summary."""

        return asdict(self)


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


def _parse_tensor_shapes(
    reader: FlatBufferReader,
    subgraph: int,
) -> list[tuple[int, ...]]:
    """Decode every tensor shape in one Circle subgraph."""

    return [
        tuple(reader.vector_i32(table, 0))
        for table in reader.vector_tables(subgraph, 0)
    ]


def _parse_tensor_buffers(reader: FlatBufferReader, subgraph: int) -> list[int]:
    """Decode every tensor buffer index in one Circle subgraph."""

    return [
        reader.scalar_u32(table, 2, 0) for table in reader.vector_tables(subgraph, 0)
    ]


def _parse_operators(
    reader: FlatBufferReader,
    subgraph: int,
    operator_codes: list[int],
) -> list[OperatorInfo]:
    """Decode operator types and tensor connections from one subgraph."""

    result: list[OperatorInfo] = []
    for table in reader.vector_tables(subgraph, 3):
        opcode_index = reader.scalar_u32(table, 0, 0)
        result.append(
            OperatorInfo(
                builtin_code=operator_codes[opcode_index],
                inputs=tuple(reader.vector_i32(table, 1)),
                outputs=tuple(reader.vector_i32(table, 2)),
            )
        )
    return result


def _const_i32_data(
    reader: FlatBufferReader,
    root: int,
    tensor_buffers: list[int],
    tensor_index: int,
) -> tuple[int, ...] | None:
    """Decode one inline INT32 constant tensor."""

    if tensor_index < 0 or tensor_index >= len(tensor_buffers):
        return None
    buffer_index = tensor_buffers[tensor_index]
    buffers = reader.vector_tables(root, 4)
    if buffer_index <= 0 or buffer_index >= len(buffers):
        return None
    payload = reader.vector_u8(buffers[buffer_index], 0)
    if not payload or len(payload) % 4 != 0:
        return None
    return tuple(
        struct.unpack_from("<i", payload, offset)[0]
        for offset in range(0, len(payload), 4)
    )


def _is_inverse(first: Iterable[int], second: Iterable[int]) -> bool:
    """Return whether composing two permutations produces the identity."""

    first_values = tuple(first)
    second_values = tuple(second)
    if len(first_values) != len(second_values):
        return False
    if sorted(first_values) != list(range(len(first_values))):
        return False
    if sorted(second_values) != list(range(len(second_values))):
        return False
    return all(
        first_values[second_values[index]] == index
        for index in range(len(second_values))
    )


def _build_edges(
    operators: list[OperatorInfo],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Build tensor producer and consumer indexes."""

    producers: dict[int, int] = {}
    consumers: dict[int, list[int]] = {}
    for operator_index, operator in enumerate(operators):
        for tensor_index in operator.outputs:
            if tensor_index >= 0:
                producers[tensor_index] = operator_index
        for tensor_index in operator.inputs:
            if tensor_index >= 0:
                consumers.setdefault(tensor_index, []).append(operator_index)
    return producers, consumers


def _transpose_permutation(
    operator: OperatorInfo,
    *,
    reader: FlatBufferReader,
    root: int,
    tensor_buffers: list[int],
) -> tuple[int, ...] | None:
    """Return the constant permutation used by one Transpose operator."""

    if operator.builtin_code != TRANSPOSE or len(operator.inputs) < 2:
        return None
    return _const_i32_data(
        reader,
        root,
        tensor_buffers,
        operator.inputs[1],
    )


def _count_consecutive_inverse_pairs(
    operators: list[OperatorInfo],
    producers: dict[int, int],
    *,
    reader: FlatBufferReader,
    root: int,
    tensor_buffers: list[int],
) -> int:
    """Count consecutive inverse Transpose operator pairs."""

    count = 0
    for operator in operators:
        if operator.builtin_code != TRANSPOSE or not operator.inputs:
            continue
        producer_index = producers.get(operator.inputs[0])
        if producer_index is None:
            continue
        producer = operators[producer_index]
        first = _transpose_permutation(
            producer,
            reader=reader,
            root=root,
            tensor_buffers=tensor_buffers,
        )
        second = _transpose_permutation(
            operator,
            reader=reader,
            root=root,
            tensor_buffers=tensor_buffers,
        )
        if first is not None and second is not None and _is_inverse(first, second):
            count += 1
    return count


def _count_transpose_add_round_trips(
    operators: list[OperatorInfo],
    producers: dict[int, int],
    consumers: dict[int, list[int]],
    *,
    reader: FlatBufferReader,
    root: int,
    tensor_buffers: list[int],
) -> int:
    """Count inverse Transpose round trips whose middle operator is ADD."""

    count = 0
    for operator in operators:
        if operator.builtin_code != ADD:
            continue
        if len(operator.inputs) != 2 or len(operator.outputs) != 1:
            continue
        input_producers = [producers.get(index) for index in operator.inputs]
        if any(index is None for index in input_producers):
            continue
        input_transposes = [operators[int(index)] for index in input_producers]
        first = _transpose_permutation(
            input_transposes[0],
            reader=reader,
            root=root,
            tensor_buffers=tensor_buffers,
        )
        second = _transpose_permutation(
            input_transposes[1],
            reader=reader,
            root=root,
            tensor_buffers=tensor_buffers,
        )
        if first is None or second is None or first != second:
            continue
        output_consumers = consumers.get(operator.outputs[0], [])
        if len(output_consumers) != 1:
            continue
        output_transpose = operators[output_consumers[0]]
        inverse = _transpose_permutation(
            output_transpose,
            reader=reader,
            root=root,
            tensor_buffers=tensor_buffers,
        )
        if inverse is not None and _is_inverse(first, inverse):
            count += 1
    return count


def verify_circle_layout(
    path: str | Path,
    *,
    expected_transpose_count: int = 0,
) -> dict[str, object]:
    """Validate the hand detector's NHWC input and optimized layout graph."""

    circle_path = Path(path)
    data = circle_path.read_bytes()
    if len(data) < 8 or data[4:8] != b"CIR0":
        raise ValueError(f"{circle_path} does not contain a Circle CIR0 identifier")

    reader = FlatBufferReader(data)
    root = reader.root_table()
    subgraphs = reader.vector_tables(root, 2)
    if len(subgraphs) != 1:
        raise RuntimeError(f"Expected one subgraph, found {len(subgraphs)}.")
    subgraph = subgraphs[0]
    operator_codes = _parse_operator_codes(reader, root)
    tensor_shapes = _parse_tensor_shapes(reader, subgraph)
    tensor_buffers = _parse_tensor_buffers(reader, subgraph)
    operators = _parse_operators(reader, subgraph, operator_codes)
    input_indices = tuple(reader.vector_i32(subgraph, 1))
    input_shapes = tuple(tensor_shapes[index] for index in input_indices)
    if input_shapes != (EXPECTED_INPUT_SHAPE,):
        raise RuntimeError(
            "Expected one NHWC graph input with shape "
            f"{list(EXPECTED_INPUT_SHAPE)}, found {input_shapes}."
        )

    producers, consumers = _build_edges(operators)
    inverse_pairs = _count_consecutive_inverse_pairs(
        operators,
        producers,
        reader=reader,
        root=root,
        tensor_buffers=tensor_buffers,
    )
    add_round_trips = _count_transpose_add_round_trips(
        operators,
        producers,
        consumers,
        reader=reader,
        root=root,
        tensor_buffers=tensor_buffers,
    )
    if inverse_pairs:
        raise RuntimeError(
            f"Found {inverse_pairs} consecutive inverse Transpose pairs."
        )
    if add_round_trips:
        raise RuntimeError(
            f"Found {add_round_trips} Transpose-ADD-Transpose round trips."
        )
    transpose_count = sum(operator.builtin_code == TRANSPOSE for operator in operators)
    if transpose_count != expected_transpose_count:
        raise RuntimeError(
            f"Expected {expected_transpose_count} Transpose operators, "
            f"found {transpose_count}."
        )

    summary = LayoutVerificationSummary(
        path=str(circle_path),
        size_bytes=len(data),
        input_shapes=input_shapes,
        transpose_count=transpose_count,
        add_count=sum(operator.builtin_code == ADD for operator in operators),
        consecutive_inverse_transpose_pairs=inverse_pairs,
        transpose_add_round_trips=add_round_trips,
    )
    return summary.to_dict()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circle", type=Path)
    parser.add_argument(
        "--expected-transpose-count",
        type=int,
        default=0,
        help="Require this exact number of Circle Transpose operators.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Circle layout verifier and print its summary."""

    args = parse_args()
    summary = verify_circle_layout(
        args.circle,
        expected_transpose_count=args.expected_transpose_count,
    )
    print(f"Verified NHWC input shape {list(EXPECTED_INPUT_SHAPE)}.")
    print("Verified zero consecutive inverse Transpose pairs.")
    print("Verified zero Transpose-ADD-Transpose round trips.")
    print("Verified Circle Transpose operator count: " f"{summary['transpose_count']}.")


if __name__ == "__main__":
    main()
