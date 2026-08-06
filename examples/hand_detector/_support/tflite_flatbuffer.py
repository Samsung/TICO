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

"""Minimal FlatBuffer reader for the TensorFlow Lite fields used by this model."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TENSOR_DTYPES: dict[int, np.dtype[Any]] = {
    0: np.dtype(np.float32),
    1: np.dtype(np.float16),
    2: np.dtype(np.int32),
    3: np.dtype(np.uint8),
    4: np.dtype(np.int64),
    6: np.dtype(np.bool_),
    7: np.dtype(np.int16),
    9: np.dtype(np.int8),
    10: np.dtype(np.float64),
    12: np.dtype(np.uint64),
    15: np.dtype(np.uint32),
    16: np.dtype(np.uint16),
}

BUILTIN_OPERATOR_NAMES = {
    0: "ADD",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    6: "DEQUANTIZE",
    17: "MAX_POOL_2D",
    22: "RESHAPE",
    23: "RESIZE_BILINEAR",
    34: "PAD",
    54: "PRELU",
}


class FlatBufferReader:
    """Read scalar, vector, string, and table fields from a FlatBuffer byte array."""

    def __init__(self, data: bytes) -> None:
        """Store the immutable FlatBuffer payload."""
        self.data = data

    def _unpack(self, fmt: str, offset: int) -> Any:
        """Unpack one scalar value at an absolute byte offset."""
        return struct.unpack_from(fmt, self.data, offset)[0]

    def u8(self, offset: int) -> int:
        """Read an unsigned 8-bit integer."""
        return self.data[offset]

    def i8(self, offset: int) -> int:
        """Read a signed 8-bit integer."""
        return int(self._unpack("<b", offset))

    def u16(self, offset: int) -> int:
        """Read an unsigned 16-bit integer."""
        return int(self._unpack("<H", offset))

    def i32(self, offset: int) -> int:
        """Read a signed 32-bit integer."""
        return int(self._unpack("<i", offset))

    def i64(self, offset: int) -> int:
        """Read a signed 64-bit integer."""
        return int(self._unpack("<q", offset))

    def u32(self, offset: int) -> int:
        """Read an unsigned 32-bit integer."""
        return int(self._unpack("<I", offset))

    def f32(self, offset: int) -> float:
        """Read a 32-bit floating-point value."""
        return float(self._unpack("<f", offset))

    def root_table(self) -> int:
        """Return the absolute position of the FlatBuffer root table."""
        return self.u32(0)

    def field(self, table: int, slot: int) -> int | None:
        """Return the absolute scalar field position for a table slot."""
        vtable = table - self.i32(table)
        vtable_size = self.u16(vtable)
        entry = vtable + 4 + 2 * slot
        if entry + 2 > vtable + vtable_size:
            return None
        relative = self.u16(entry)
        return None if relative == 0 else table + relative

    def indirect(self, offset: int) -> int:
        """Follow a FlatBuffer uoffset."""
        return offset + self.u32(offset)

    def table(self, parent: int, slot: int) -> int | None:
        """Return a nested table position."""
        field = self.field(parent, slot)
        return None if field is None else self.indirect(field)

    def scalar_i8(self, table: int, slot: int, default: int = 0) -> int:
        """Read an int8 table field with a default."""
        field = self.field(table, slot)
        return default if field is None else self.i8(field)

    def scalar_i32(self, table: int, slot: int, default: int = 0) -> int:
        """Read an int32 table field with a default."""
        field = self.field(table, slot)
        return default if field is None else self.i32(field)

    def scalar_u32(self, table: int, slot: int, default: int = 0) -> int:
        """Read a uint32 table field with a default."""
        field = self.field(table, slot)
        return default if field is None else self.u32(field)

    def scalar_bool(self, table: int, slot: int, default: bool = False) -> bool:
        """Read a boolean table field with a default."""
        field = self.field(table, slot)
        return default if field is None else bool(self.u8(field))

    def vector(self, table: int, slot: int) -> tuple[int, int] | None:
        """Return a vector data position and element count."""
        field = self.field(table, slot)
        if field is None:
            return None
        vector = self.indirect(field)
        return vector + 4, self.u32(vector)

    def vector_i32(self, table: int, slot: int) -> list[int]:
        """Read an int32 vector."""
        vector = self.vector(table, slot)
        if vector is None:
            return []
        data, size = vector
        return [self.i32(data + 4 * index) for index in range(size)]

    def vector_i64(self, table: int, slot: int) -> list[int]:
        """Read an int64 vector."""
        vector = self.vector(table, slot)
        if vector is None:
            return []
        data, size = vector
        return [self.i64(data + 8 * index) for index in range(size)]

    def vector_f32(self, table: int, slot: int) -> list[float]:
        """Read a float32 vector."""
        vector = self.vector(table, slot)
        if vector is None:
            return []
        data, size = vector
        return [self.f32(data + 4 * index) for index in range(size)]

    def vector_u8(self, table: int, slot: int) -> bytes:
        """Read a byte vector."""
        vector = self.vector(table, slot)
        if vector is None:
            return b""
        data, size = vector
        return self.data[data : data + size]

    def vector_tables(self, table: int, slot: int) -> list[int]:
        """Read a vector of nested tables."""
        vector = self.vector(table, slot)
        if vector is None:
            return []
        data, size = vector
        return [self.indirect(data + 4 * index) for index in range(size)]

    def string(self, table: int, slot: int) -> str | None:
        """Read a UTF-8 string field."""
        field = self.field(table, slot)
        if field is None:
            return None
        string = self.indirect(field)
        size = self.u32(string)
        return self.data[string + 4 : string + 4 + size].decode("utf-8", "replace")


@dataclass(frozen=True)
class TensorInfo:
    """Describe one TFLite tensor."""

    shape: tuple[int, ...]
    tensor_type: int
    buffer_index: int
    name: str


@dataclass(frozen=True)
class OperatorInfo:
    """Describe one TFLite operator and its decoded builtin options."""

    index: int
    name: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    options: dict[str, Any]


class TFLiteModel:
    """Parse the subset of the TFLite schema needed by the MediaPipe detector."""

    def __init__(self, path: str | Path) -> None:
        """Read and decode the supported portions of one TFLite model."""
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.reader = FlatBufferReader(self.data)
        if self.data[4:8] != b"TFL3":
            raise ValueError(f"{self.path} is not a TFLite FlatBuffer")
        self.root = self.reader.root_table()
        self.operator_codes = self._parse_operator_codes()
        self.buffers = self._parse_buffers()
        subgraphs = self.reader.vector_tables(self.root, 2)
        if len(subgraphs) != 1:
            raise ValueError(f"Expected one subgraph, found {len(subgraphs)}")
        self.subgraph = subgraphs[0]
        self.tensors = self._parse_tensors()
        self.inputs = tuple(self.reader.vector_i32(self.subgraph, 1))
        self.outputs = tuple(self.reader.vector_i32(self.subgraph, 2))
        self.operators = self._parse_operators()

    def _parse_operator_codes(self) -> list[int]:
        """Decode the builtin operator code table."""
        result: list[int] = []
        for table in self.reader.vector_tables(self.root, 1):
            deprecated = self.reader.scalar_i8(table, 0, 0)
            builtin = self.reader.scalar_i32(table, 3, deprecated)
            if builtin == 0 and deprecated != 0:
                builtin = deprecated
            result.append(builtin)
        return result

    def _parse_buffers(self) -> list[bytes]:
        """Decode all inline buffer payloads."""
        return [
            self.reader.vector_u8(table, 0)
            for table in self.reader.vector_tables(self.root, 4)
        ]

    def _parse_tensors(self) -> list[TensorInfo]:
        """Decode tensor metadata from the only subgraph."""
        result: list[TensorInfo] = []
        for table in self.reader.vector_tables(self.subgraph, 0):
            result.append(
                TensorInfo(
                    shape=tuple(self.reader.vector_i32(table, 0)),
                    tensor_type=self.reader.scalar_i8(table, 1, 0),
                    buffer_index=self.reader.scalar_u32(table, 2, 0),
                    name=self.reader.string(table, 3) or "",
                )
            )
        return result

    def _decode_options(self, name: str, table: int | None) -> dict[str, Any]:
        """Decode builtin options used by one supported operator."""
        if table is None:
            return {}
        reader = self.reader
        if name == "CONV_2D":
            return {
                "padding": reader.scalar_i8(table, 0, 0),
                "stride_w": reader.scalar_i32(table, 1, 1),
                "stride_h": reader.scalar_i32(table, 2, 1),
                "fused_activation": reader.scalar_i8(table, 3, 0),
                "dilation_w": reader.scalar_i32(table, 4, 1),
                "dilation_h": reader.scalar_i32(table, 5, 1),
            }
        if name == "DEPTHWISE_CONV_2D":
            return {
                "padding": reader.scalar_i8(table, 0, 0),
                "stride_w": reader.scalar_i32(table, 1, 1),
                "stride_h": reader.scalar_i32(table, 2, 1),
                "depth_multiplier": reader.scalar_i32(table, 3, 1),
                "fused_activation": reader.scalar_i8(table, 4, 0),
                "dilation_w": reader.scalar_i32(table, 5, 1),
                "dilation_h": reader.scalar_i32(table, 6, 1),
            }
        if name == "MAX_POOL_2D":
            return {
                "padding": reader.scalar_i8(table, 0, 0),
                "stride_w": reader.scalar_i32(table, 1, 1),
                "stride_h": reader.scalar_i32(table, 2, 1),
                "filter_w": reader.scalar_i32(table, 3, 1),
                "filter_h": reader.scalar_i32(table, 4, 1),
                "fused_activation": reader.scalar_i8(table, 5, 0),
            }
        if name == "ADD":
            return {"fused_activation": reader.scalar_i8(table, 0, 0)}
        if name == "CONCATENATION":
            return {
                "axis": reader.scalar_i32(table, 0, 0),
                "fused_activation": reader.scalar_i8(table, 1, 0),
            }
        if name == "RESHAPE":
            return {"new_shape": reader.vector_i32(table, 0)}
        if name == "RESIZE_BILINEAR":
            # TFLite keeps two deprecated fields before the active options:
            #   slot 0: new_height
            #   slot 1: new_width
            #   slot 2: align_corners
            #   slot 3: half_pixel_centers
            return {
                "align_corners": reader.scalar_bool(table, 2, False),
                "half_pixel_centers": reader.scalar_bool(table, 3, False),
            }
        return {}

    def _parse_operators(self) -> list[OperatorInfo]:
        """Decode operators in execution order."""
        result: list[OperatorInfo] = []
        for index, table in enumerate(self.reader.vector_tables(self.subgraph, 3)):
            opcode_index = self.reader.scalar_u32(table, 0, 0)
            builtin = self.operator_codes[opcode_index]
            name = BUILTIN_OPERATOR_NAMES.get(builtin)
            if name is None:
                raise NotImplementedError(
                    f"Builtin operator code {builtin} at index {index} is unsupported"
                )
            result.append(
                OperatorInfo(
                    index=index,
                    name=name,
                    inputs=tuple(self.reader.vector_i32(table, 1)),
                    outputs=tuple(self.reader.vector_i32(table, 2)),
                    options=self._decode_options(name, self.reader.table(table, 4)),
                )
            )
        return result

    def tensor_array(self, tensor_index: int) -> np.ndarray[Any, Any]:
        """Return a constant tensor as a NumPy array."""
        tensor = self.tensors[tensor_index]
        try:
            dtype = TENSOR_DTYPES[tensor.tensor_type]
        except KeyError as exc:
            raise NotImplementedError(
                f"Unsupported TFLite tensor type {tensor.tensor_type}"
            ) from exc
        payload = self.buffers[tensor.buffer_index]
        if not payload:
            raise ValueError(f"Tensor {tensor_index} does not contain constant data")
        return np.frombuffer(payload, dtype=dtype).reshape(tensor.shape).copy()
