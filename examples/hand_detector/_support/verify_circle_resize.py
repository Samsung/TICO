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

"""Count Circle RESIZE_BILINEAR operators in an exported model."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.hand_detector._support.tflite_flatbuffer import FlatBufferReader


RESIZE_BILINEAR_BUILTIN_CODE = 23


def parse_operator_codes(reader: FlatBufferReader, root: int) -> list[int]:
    """Decode builtin operator codes from a Circle model table."""
    result: list[int] = []
    for table in reader.vector_tables(root, 1):
        deprecated_code = reader.scalar_i8(table, 0, 0)
        builtin_code = reader.scalar_i32(table, 3, deprecated_code)
        if builtin_code == 0 and deprecated_code != 0:
            builtin_code = deprecated_code
        result.append(builtin_code)
    return result


def read_resize_bilinear_options(path: Path) -> list[tuple[bool, bool]]:
    """Return coordinate options of all first-subgraph RESIZE_BILINEAR nodes."""
    data = path.read_bytes()
    if len(data) < 8 or data[4:8] != b"CIR0":
        raise ValueError(f"{path} does not contain a Circle CIR0 identifier")
    reader = FlatBufferReader(data)
    root = reader.root_table()
    operator_codes = parse_operator_codes(reader, root)
    subgraphs = reader.vector_tables(root, 2)
    if not subgraphs:
        raise ValueError("The Circle model does not contain a subgraph")

    result: list[tuple[bool, bool]] = []
    for operator in reader.vector_tables(subgraphs[0], 3):
        opcode_index = reader.scalar_u32(operator, 0, 0)
        if operator_codes[opcode_index] != RESIZE_BILINEAR_BUILTIN_CODE:
            continue
        options = reader.table(operator, 4)
        if options is None:
            raise ValueError("RESIZE_BILINEAR does not contain builtin options")
        # ResizeBilinearOptions keeps deprecated new_height/new_width in slots 0/1.
        align_corners = reader.scalar_bool(options, 2, False)
        half_pixel_centers = reader.scalar_bool(options, 3, False)
        result.append((align_corners, half_pixel_centers))
    return result


def count_resize_bilinear(path: Path) -> int:
    """Return the number of RESIZE_BILINEAR operators in the first subgraph."""
    return len(read_resize_bilinear_options(path))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circle", type=Path)
    parser.add_argument("--expected-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Validate the expected number of Circle ResizeBilinear operators."""
    args = parse_args()
    options = read_resize_bilinear_options(args.circle)
    actual_count = len(options)
    if actual_count != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} RESIZE_BILINEAR operators, "
            f"found {actual_count}"
        )
    expected_options = [(False, True)] * args.expected_count
    if options != expected_options:
        raise RuntimeError(
            "Unexpected Circle ResizeBilinear options: "
            f"expected {expected_options}, found {options}"
        )
    print(
        f"Verified {actual_count} Circle RESIZE_BILINEAR operators with "
        f"alignCorners=False and halfPixelCenters=True in {args.circle}."
    )


if __name__ == "__main__":
    main()
