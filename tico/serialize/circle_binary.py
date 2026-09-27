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

"""Circle binary layout, independent of conversion policy and file I/O.

The FlatBuffers Python builder has a 2 GiB limit. The complete Circle file does not:
Buffer.offset/size can address constant data after the FlatBuffer. This module
chooses the representation automatically and always returns the *entire* model.
"""

from __future__ import annotations

import copy
import struct
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Sequence

import numpy as np

CIRCLE_FILE_IDENTIFIER = b"CIR0"

# Keep these implementation details private. Tests lower the budget instead of
# allocating multi-gigabyte tensors in the ordinary test suite.
_FLATBUFFER_LIMIT = (1 << 31) - 1
_METADATA_RESERVE = 16 << 20
_ALIGNMENT = 16
_UINT64_MAX = (1 << 64) - 1
# Buffer's second field (offset: ulong), including the two vtable header slots.
_BUFFER_OFFSET_VTABLE_SLOT = 6


class CircleSerializationError(ValueError):
    """A model cannot be represented as a self-contained Circle binary."""


class _FlatbufferTooLarge(CircleSerializationError):
    """The FlatBuffer portion, rather than the complete file, exceeds its budget."""


@dataclass(frozen=True)
class ExternalBufferRange:
    """An already validated file-relative constant-buffer range."""

    index: int
    offset: int
    size: int


def _buffer_size(data: Any) -> int:
    """Get the byte count without reading or copying a dense buffer."""

    if data is None:
        return 0
    if isinstance(data, np.ndarray):
        if data.ndim != 1 or data.dtype != np.dtype(np.uint8):
            raise CircleSerializationError("Circle Buffer.data must be a uint8 vector.")
        return int(data.nbytes)
    if isinstance(data, memoryview):
        if data.ndim != 1 or data.itemsize != 1 or data.format != "B":
            raise CircleSerializationError(
                "Circle Buffer.data memoryviews must be one-dimensional unsigned bytes."
            )
        return data.nbytes
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    # Generated Object API tables also accept Python sequences of byte values.
    return len(data)


def _byte_view(data: Any) -> memoryview:
    """Retain dense storage; only non-contiguous inputs/sequences need a copy."""

    if isinstance(data, np.ndarray):
        _buffer_size(data)
        return memoryview(np.ascontiguousarray(data)).cast("B")
    if isinstance(data, (bytes, bytearray, memoryview)):
        _buffer_size(data)
        view = memoryview(data)
        if not view.c_contiguous:
            view = memoryview(view.tobytes())
        return view.cast("B")
    # bytes() validates sequence elements instead of silently wrapping integers.
    return memoryview(bytes(data))


def _check_unresolved_offsets(model: Any) -> None:
    """Never repack file-relative references without their backing payloads."""

    for index, buffer in enumerate(getattr(model, "buffers", None) or ()):
        if getattr(buffer, "offset", 0) or getattr(buffer, "size", 0):
            raise CircleSerializationError(
                f"Buffer {index} contains unresolved external offsets. "
                "Load the complete binary with CircleDocument.from_bytes() or "
                "CircleDocument.load() before repacking it."
            )
    for graph in getattr(model, "subgraphs", None) or ():
        for operator in getattr(graph, "operators", None) or ():
            if getattr(operator, "largeCustomOptionsOffset", 0) or getattr(
                operator, "largeCustomOptionsSize", 0
            ):
                raise CircleSerializationError(
                    "External custom operator options are not supported. "
                    "Repacking them would lose their file-relative payload."
                )


def _load_flatbuffers() -> Any:
    """Import the existing FlatBuffers dependency only when packing is needed."""

    return import_module("flatbuffers")


def _pack_flatbuffer(model: Any, limit: int) -> bytearray:
    """Pack one bounded header; reject oversized vectors before copying them."""

    flatbuffers = _load_flatbuffers()
    builder_type: Any = flatbuffers.Builder

    class BoundedBuilder(builder_type):
        def Prep(self, size: int, additionalBytes: int) -> None:
            # Match Builder.Prep's alignment calculation. Checking before the
            # superclass runs also precedes CreateNumpyVector's tobytes() copy.
            offset = self.Offset()
            padding = (-(offset + additionalBytes)) & (size - 1)
            if offset + padding + size + additionalBytes > limit:
                raise _FlatbufferTooLarge(
                    f"Circle FlatBuffer exceeds its {limit}-byte budget."
                )
            super().Prep(size, additionalBytes)

    builder = BoundedBuilder(min(1024, limit))
    try:
        builder.Finish(model.Pack(builder), CIRCLE_FILE_IDENTIFIER)
    except flatbuffers.builder.BuilderSizeError as error:
        raise _FlatbufferTooLarge(
            "Circle FlatBuffer exceeds the builder limit."
        ) from error
    return builder.Output()


def _payload_layout(
    header_size: int, sizes: Sequence[int]
) -> tuple[tuple[int, ...], int]:
    """Compute aligned uint64 file offsets using Python's unbounded integers."""

    if header_size < 0:
        raise CircleSerializationError("A Circle header cannot have a negative size.")
    cursor = header_size
    offsets = []
    for size in sizes:
        if size <= 0:
            raise CircleSerializationError("External payloads must be non-empty.")
        cursor = (cursor + _ALIGNMENT - 1) & -_ALIGNMENT
        if cursor > _UINT64_MAX or size > _UINT64_MAX - cursor:
            raise CircleSerializationError("Circle payload offsets exceed uint64.")
        offsets.append(cursor)
        cursor += size
    if cursor > sys.maxsize:
        raise CircleSerializationError(
            "The complete Circle binary exceeds this Python process's address space."
        )
    return tuple(offsets), cursor


def _join_payloads(
    header: bytearray,
    payloads: Sequence[memoryview],
    offsets: Sequence[int],
) -> bytes:
    """Materialize the complete binary once, without per-payload byte copies."""

    if len(payloads) != len(offsets):
        raise CircleSerializationError("Every Circle payload must have one offset.")
    parts: list[bytes | bytearray | memoryview] = [header]
    cursor = len(header)
    for payload, offset in zip(payloads, offsets):
        if offset < cursor:
            raise CircleSerializationError(
                "Circle payloads overlap the preceding data."
            )
        parts.extend((b"\0" * (offset - cursor), payload))
        cursor = offset + len(payload)
    return b"".join(parts)


def serialize_circle_model(model: Any) -> bytes:
    """Serialize an Object API model, automatically relocating large payloads.

    Small models retain the ordinary inline layout. An aggregate payload check
    reserves headroom for metadata; an actual bounded pack also handles cases
    where metadata, padding, or many individually small constants reach the
    limit. Only a size-limit exception selects the alternative representation.

    No file is created here. The result is always complete bytes, not a header,
    a path, or a lazily materialized object. Callers must have enough RAM for the
    resulting binary in addition to the conversion graph and tensor storage.
    """

    if model is None or not hasattr(model, "Pack"):
        raise TypeError("Expected a Circle Object API model with a Pack method.")
    _check_unresolved_offsets(model)
    buffers = list(getattr(model, "buffers", None) or ())
    sizes = [_buffer_size(getattr(buffer, "data", None)) for buffer in buffers]
    if sizes and sizes[0]:
        raise CircleSerializationError("Circle buffer 0 must remain empty.")
    limit = _FLATBUFFER_LIMIT
    reserve = min(_METADATA_RESERVE, limit // 8)
    if sum(sizes) <= limit - reserve:
        try:
            return bytes(_pack_flatbuffer(model, limit))
        except _FlatbufferTooLarge:
            # The failed builder is out of scope before allocating the header.
            # Do not turn graph errors, invalid values, or MemoryError into a
            # different export path.
            pass

    # Keep empty and one-byte constants inline, including buffer 0. This also
    # avoids consumers that treat a one-byte external size as a placeholder.
    indices = [index for index, size in enumerate(sizes) if size > 1]
    if not indices:
        raise CircleSerializationError(
            "Circle metadata cannot fit in the FlatBuffer and there are no "
            "relocatable constant buffers."
        )

    # Shallow-copy only the affected schema tables, never the tensor payloads or
    # the graph. The caller's Object API model remains usable after success or
    # failure, and can be serialized repeatedly.
    projected = copy.copy(model)
    projected.buffers = list(buffers)
    payloads = []
    for index in indices:
        payload = _byte_view(buffers[index].data)
        payloads.append(payload)
        buffer = copy.copy(buffers[index])
        buffer.data = None
        # A real, non-default value keeps the offset field in the vtable. Its
        # final file position is patched in the finished header, not repacked.
        buffer.offset = _ALIGNMENT
        buffer.size = len(payload)
        projected.buffers[index] = buffer
    try:
        header = _pack_flatbuffer(projected, limit)
    except _FlatbufferTooLarge as error:
        raise CircleSerializationError(
            "Circle graph metadata or inline custom options exceed the FlatBuffer "
            "limit even after relocating constant buffers. Split the graph."
        ) from error

    offsets, _ = _payload_layout(len(header), [len(payload) for payload in payloads])
    from circle_schema import circle

    root = circle.Model.Model.GetRootAsModel(header, 0)
    for index, offset in zip(indices, offsets):
        table = root.Buffers(index)._tab
        field = table.Offset(_BUFFER_OFFSET_VTABLE_SLOT)
        if not field:
            raise CircleSerializationError("Circle schema omitted Buffer.offset.")
        struct.pack_into("<Q", header, table.Pos + field, offset)
    return _join_payloads(header, payloads, offsets)


def external_buffer_ranges(
    root: Any, binary_size: int
) -> tuple[ExternalBufferRange, ...]:
    """Validate external ranges before unpacking or reading any payload."""

    ranges = []
    for index in range(root.BuffersLength()):
        buffer = root.Buffers(index)
        offset, size = int(buffer.Offset()), int(buffer.Size())
        if offset == 0 and size == 0:
            continue
        if index == 0:
            raise CircleSerializationError(
                "Circle buffer 0 must not have external data."
            )
        if offset <= 1 or size <= 0:
            raise CircleSerializationError(
                f"Buffer {index} has incomplete external metadata."
            )
        if buffer.DataLength():
            raise CircleSerializationError(
                f"Buffer {index} has both inline and external data."
            )
        if offset < 8 or offset > binary_size or size > binary_size - offset:
            raise CircleSerializationError(
                f"Buffer {index} points outside the Circle binary."
            )
        ranges.append(ExternalBufferRange(index, offset, size))
    return tuple(ranges)


def restore_external_buffers(
    model: Any,
    data: bytes,
    ranges: Sequence[ExternalBufferRange],
) -> None:
    """Resolve payloads to read-only NumPy views and clear stale file positions.

    These views keep the source bytes alive. A transform that edits a payload
    in place must copy it first; assigning a replacement buffer remains valid.
    Repacking then recomputes the layout rather than preserving old offsets.
    """

    for region in ranges:
        buffer = model.buffers[region.index]
        buffer.data = np.frombuffer(
            data, dtype=np.uint8, count=region.size, offset=region.offset
        )
        buffer.offset = 0
        buffer.size = 0
    _check_unresolved_offsets(model)
