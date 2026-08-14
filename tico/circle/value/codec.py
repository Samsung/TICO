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

from typing import Any, Iterable

import numpy as np

from tico.circle.errors import CircleValueError
from tico.circle.graph import as_list
from tico.circle.value.dtype import (
    default_tensor_type_registry,
    TensorTypeRegistry,
    TensorTypeSpec,
)
from tico.circle.value.tensor import TensorQuantization, TensorValue


class TensorValueCodec:
    """Encode and decode inline Circle constant buffers without implicit casting."""

    def __init__(self, registry: TensorTypeRegistry | None = None) -> None:
        """Create a codec with an explicit or schema-derived tensor type registry."""

        self.registry = registry or default_tensor_type_registry()

    def encode(self, value: TensorValue) -> bytes:
        """Encode a logical tensor value into Circle little-endian buffer bytes."""

        spec = self.registry.by_value(value.tensor_type)
        self._check_logical_dtype(value, spec)
        if spec.packed:
            return self._encode_four_bit(value, spec)

        storage_dtype = spec.storage_dtype.newbyteorder("<")
        array = np.asarray(value.data, dtype=storage_dtype)
        array = np.ascontiguousarray(array.reshape(-1))
        return bytes(array.view(np.uint8))

    def decode(
        self,
        payload: bytes | bytearray | memoryview | np.ndarray[Any, Any],
        *,
        tensor_type: int,
        shape: Iterable[int],
        quantization: TensorQuantization | None = None,
    ) -> TensorValue:
        """Decode Circle buffer bytes into an immutable logical tensor value."""

        normalized_shape = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in normalized_shape):
            raise CircleValueError(
                f"Cannot decode a tensor with a negative shape: {normalized_shape}."
            )
        spec = self.registry.by_value(tensor_type)
        raw = _buffer_bytes(payload)
        element_count = _element_count(normalized_shape)
        expected_size = spec.storage_size(element_count)
        if len(raw) != expected_size:
            raise CircleValueError(
                f"TensorType.{spec.name} with shape {normalized_shape} requires "
                f"{expected_size} bytes, but the buffer contains {len(raw)}."
            )

        if spec.packed:
            array = self._decode_four_bit(raw, element_count, spec)
        else:
            storage_dtype = spec.storage_dtype.newbyteorder("<")
            array = np.frombuffer(raw, dtype=storage_dtype, count=element_count)
            array = array.astype(spec.logical_dtype, copy=True)

        return TensorValue(
            tensor_type=int(tensor_type),
            shape=normalized_shape,
            data=array.reshape(normalized_shape),
            quantization=quantization,
        )

    def decode_tensor(
        self,
        model: Any,
        *,
        subgraph_index: int,
        tensor_index: int,
    ) -> TensorValue:
        """Decode one non-variable tensor backed by an inline model buffer."""

        subgraphs = as_list(getattr(model, "subgraphs", None))
        if subgraph_index < 0 or subgraph_index >= len(subgraphs):
            raise IndexError(
                f"Subgraph index {subgraph_index} is outside "
                f"0..{len(subgraphs) - 1}."
            )
        subgraph = subgraphs[subgraph_index]
        tensors = as_list(getattr(subgraph, "tensors", None))
        if tensor_index < 0 or tensor_index >= len(tensors):
            raise IndexError(
                f"Tensor index {tensor_index} is outside 0..{len(tensors) - 1}."
            )
        tensor = tensors[tensor_index]
        if bool(getattr(tensor, "isVariable", False)):
            raise CircleValueError("Variable tensors cannot be decoded as constants.")

        buffer_index = int(getattr(tensor, "buffer", 0) or 0)
        buffers = as_list(getattr(model, "buffers", None))
        if buffer_index <= 0 or buffer_index >= len(buffers):
            raise CircleValueError(
                f"Tensor {tensor_index} does not reference an inline constant buffer."
            )
        buffer = buffers[buffer_index]
        data = getattr(buffer, "data", None)
        if data is None:
            if int(getattr(buffer, "offset", 0) or 0) or int(
                getattr(buffer, "size", 0) or 0
            ):
                raise CircleValueError(
                    "External Circle buffers are not supported by TensorValueCodec."
                )
            raise CircleValueError(
                f"Tensor {tensor_index} references a buffer without inline data."
            )
        if int(getattr(buffer, "offset", 0) or 0) or int(
            getattr(buffer, "size", 0) or 0
        ):
            raise CircleValueError(
                "External Circle buffers are not supported by TensorValueCodec."
            )

        return self.decode(
            data,
            tensor_type=int(getattr(tensor, "type", -1)),
            shape=tuple(int(value) for value in as_list(tensor.shape)),
            quantization=TensorQuantization.from_object(
                getattr(tensor, "quantization", None)
            ),
        )

    def write_buffer(self, buffer: Any, value: TensorValue) -> None:
        """Replace one Object API buffer payload with encoded tensor bytes."""

        payload = self.encode(value)
        buffer.data = np.frombuffer(payload, dtype=np.uint8).copy()
        if hasattr(buffer, "offset"):
            buffer.offset = 0
        if hasattr(buffer, "size"):
            buffer.size = 0

    @staticmethod
    def _check_logical_dtype(value: TensorValue, spec: TensorTypeSpec) -> None:
        """Reject implicit NumPy dtype conversion during encoding."""

        actual = np.dtype(value.data.dtype).newbyteorder("=")
        expected = spec.logical_dtype.newbyteorder("=")
        if actual != expected:
            raise CircleValueError(
                f"TensorType.{spec.name} requires NumPy dtype {expected}, "
                f"but the tensor value uses {actual}."
            )

    @staticmethod
    def _encode_four_bit(value: TensorValue, spec: TensorTypeSpec) -> bytes:
        """Pack low-nibble-first signed or unsigned four-bit values."""

        flat = np.asarray(value.data).reshape(-1).astype(np.int16, copy=False)
        minimum, maximum = (-8, 7) if spec.signed else (0, 15)
        if np.any(flat < minimum) or np.any(flat > maximum):
            raise CircleValueError(
                f"TensorType.{spec.name} values must be in " f"[{minimum}, {maximum}]."
            )
        nibbles = np.bitwise_and(flat, 0x0F).astype(np.uint8, copy=False)
        packed = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
        packed[:] = nibbles[0::2]
        packed[: flat.size // 2] |= nibbles[1::2] << np.uint8(4)
        return bytes(packed)

    @staticmethod
    def _decode_four_bit(
        payload: bytes,
        element_count: int,
        spec: TensorTypeSpec,
    ) -> np.ndarray[Any, Any]:
        """Unpack low-nibble-first signed or unsigned four-bit values."""

        packed = np.frombuffer(payload, dtype=np.uint8)
        nibbles = np.empty(element_count, dtype=np.uint8)
        nibbles[0::2] = packed & np.uint8(0x0F)
        nibbles[1::2] = packed[: element_count // 2] >> np.uint8(4)
        if spec.signed:
            signed = nibbles.astype(np.int8)
            signed[signed >= 8] -= 16
            return signed
        return nibbles


def _buffer_bytes(value: Any) -> bytes:
    """Return raw bytes from generated vectors, NumPy arrays, or byte sequences."""

    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    array = np.asarray(value, dtype=np.uint8)
    return bytes(np.ascontiguousarray(array).reshape(-1))


def _element_count(shape: tuple[int, ...]) -> int:
    """Return the product of a concrete shape."""

    count = 1
    for dimension in shape:
        count *= dimension
    return count
