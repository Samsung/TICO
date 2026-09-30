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

import mmap
import os
import sys
import tempfile
import weakref
from pathlib import Path
from typing import Any, BinaryIO, Callable, TypeAlias

from tico.circle._schema import accessor_api_type, object_api_type
from tico.circle.errors import CircleIOError
from tico.serialize.circle_binary import (
    CircleBinaryLayout,
    external_buffer_ranges,
    plan_circle_layout,
    restore_external_buffers,
    serialize_circle_model,
)

PathLike: TypeAlias = str | os.PathLike[str]
BinarySource: TypeAlias = PathLike | BinaryIO
BinaryDestination: TypeAlias = PathLike | BinaryIO

CIRCLE_FILE_IDENTIFIER = b"CIR0"

# File saves stream the appended layout, so above this aggregate payload size they
# skip the inline FlatBuffer pack, whose builder growth and finished header hold
# several copies of every constant. Bytes-returning APIs keep the full FlatBuffer
# budget; see docs/large_circle_export.md for the resulting layout differences.
_STREAMING_INLINE_BUDGET = 1 << 30


def read_circle_bytes(source: BinarySource) -> bytes:
    """Read Circle binary data from a path, standard input, or binary stream."""

    try:
        if isinstance(source, (str, os.PathLike)):
            if os.fspath(source) == "-":
                data = sys.stdin.buffer.read()
            else:
                data = Path(source).read_bytes()
        else:
            data = source.read()
    except OSError as error:
        raise CircleIOError(f"Failed to read Circle model from {source!r}.") from error

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CircleIOError("Circle input must provide binary data.")
    result = bytes(data)
    if not result:
        raise CircleIOError("Circle input is empty.")
    return result


class CirclePayloadMapping:
    """Own one read-only mapping of a Circle file that payload views borrow from.

    ``release()`` drops this owner's reference. The operating-system mapping is
    unmapped immediately when no payload view is alive; otherwise it is unmapped
    when the last view is released, because Python refuses to close a mapping
    with exported buffers. The file descriptor opened for mapping is closed as
    soon as the mapping exists; the mapping keeps its own duplicate until it is
    unmapped.
    """

    def __init__(self, path: Path, mapping: mmap.mmap):
        self._path = path
        self._mapping: mmap.mmap | None = mapping
        # Observe the mapping after release without keeping it alive.
        self._mapping_ref = weakref.ref(mapping)

    @property
    def path(self) -> Path:
        """Return the mapped file path."""

        return self._path

    @property
    def size(self) -> int:
        """Return the mapped byte count."""

        if self._mapping is None:
            raise ValueError("The Circle payload mapping has been released.")
        return len(self._mapping)

    @property
    def released(self) -> bool:
        """Return whether this owner has released its reference."""

        return self._mapping is None

    @property
    def closed(self) -> bool:
        """Return whether the operating-system mapping itself is gone.

        After ``release()`` a surviving payload view still keeps the pages mapped;
        this reports ``False`` until that last view is dropped as well.
        """

        mapping = self._mapping_ref()
        return mapping is None or mapping.closed

    def buffer(self) -> mmap.mmap:
        """Return the mapping for parsing; callers must not keep it past release."""

        if self._mapping is None:
            raise ValueError("The Circle payload mapping has been released.")
        return self._mapping

    def same_file(self, path: PathLike) -> bool:
        """Return whether ``path`` currently names the mapped file."""

        try:
            return os.path.samefile(self._path, path)
        except OSError:
            return False

    def release(self) -> None:
        """Release this owner's reference; unmap now if no view remains."""

        mapping, self._mapping = self._mapping, None
        if mapping is None:
            return
        try:
            mapping.close()
        except BufferError:
            # Live payload views still export the mapping. They keep it alive and
            # it is unmapped when the last one is released.
            pass

    def __enter__(self) -> CirclePayloadMapping:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "released" if self._mapping is None else f"{len(self._mapping)} bytes"
        return f"CirclePayloadMapping(path={str(self._path)!r}, {state})"


def map_circle_file(path: PathLike) -> CirclePayloadMapping:
    """Map a regular file read-only without reading it into Python memory."""

    file_path = Path(path)
    try:
        with file_path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size == 0:
                raise CircleIOError("Circle input is empty.")
            mapping = mmap.mmap(stream.fileno(), size, access=mmap.ACCESS_READ)
    except OSError as error:
        raise CircleIOError(f"Failed to map Circle model {file_path}.") from error
    return CirclePayloadMapping(file_path, mapping)


def write_circle_bytes(
    data: bytes,
    destination: BinaryDestination,
    *,
    atomic: bool = True,
) -> None:
    """Write Circle binary data to a path, standard output, or binary stream."""

    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, received {type(data).__name__}.")
    _write_destination(destination, lambda stream: stream.write(data), atomic=atomic)


def _write_destination(
    destination: BinaryDestination,
    writer: Callable[[BinaryIO], Any],
    *,
    atomic: bool,
) -> None:
    """Route one writer over a stream, standard output, or an (atomic) file path."""

    if not isinstance(destination, (str, os.PathLike)):
        try:
            writer(destination)
            return
        except OSError as error:
            raise CircleIOError(
                "Failed to write Circle data to the output stream."
            ) from error

    path_text = os.fspath(destination)
    if path_text == "-":
        try:
            writer(sys.stdout.buffer)
            sys.stdout.buffer.flush()
            return
        except OSError as error:
            raise CircleIOError(
                "Failed to write Circle data to standard output."
            ) from error

    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        try:
            with path.open("wb") as stream:
                writer(stream)
            return
        except OSError as error:
            raise CircleIOError(f"Failed to write Circle model to {path}.") from error

    # Write beside the destination, then replace it, so a failure at any point
    # leaves the existing file untouched and removes the partial temporary file.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer(stream)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        _discard_temporary(temporary_path)
        raise CircleIOError(f"Failed to write Circle model to {path}.") from error
    except BaseException:
        _discard_temporary(temporary_path)
        raise


def _discard_temporary(temporary_path: Path | None) -> None:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)


def _model_from_buffer(data: Any) -> Any:
    """Unpack a complete Circle binary exposed through the buffer protocol.

    Only the FlatBuffer portion is copied into a writable ``bytearray`` so that
    metadata vectors stay editable. External payloads become read-only NumPy views
    of ``data`` itself, so a ``bytes`` object or a file mapping stays alive while
    those views exist. A file whose constants are all inline therefore copies the
    complete FlatBuffer, including the inline constant bytes.
    """

    size = len(data)
    if not size:
        raise CircleIOError("Circle input is empty.")
    if size < 8 or bytes(data[4:8]) != CIRCLE_FILE_IDENTIFIER:
        raise CircleIOError(
            "Circle input does not contain the expected CIR0 file identifier."
        )

    try:
        accessor_type = accessor_api_type("Model")
        root = accessor_type.GetRootAsModel(data, 0)
        ranges = external_buffer_ranges(root, size)
        # Only copy the FlatBuffer portion, not multi-gigabyte external data.
        # Metadata vectors remain writable as in the ordinary document API.
        header_end = min((region.offset for region in ranges), default=size)
        header = bytearray(memoryview(data)[:header_end])
        root = accessor_type.GetRootAsModel(header, 0)
        model_type = object_api_type("Model")
        if hasattr(model_type, "InitFromObj"):
            model = model_type.InitFromObj(root)
        elif hasattr(root, "UnPack"):
            model = root.UnPack()
        else:
            raise RuntimeError(
                "The Circle schema does not expose an Object API unpacker."
            )
        restore_external_buffers(model, data, ranges)
        return model
    except Exception as error:
        if isinstance(error, CircleIOError):
            raise
        raise CircleIOError(
            f"Failed to deserialize Circle binary data: {error}"
        ) from error


def model_from_bytes(data: bytes) -> Any:
    """Deserialize Circle binary data into the generated Object API model."""

    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, received {type(data).__name__}.")
    return _model_from_buffer(data)


def model_to_bytes(model: Any) -> bytes:
    """Serialize a generated Circle Object API model into binary data."""

    if model is None or not hasattr(model, "Pack"):
        raise TypeError("Expected a Circle Object API model with a Pack method.")

    try:
        return serialize_circle_model(model)
    except Exception as error:
        raise CircleIOError(f"Failed to serialize the Circle model: {error}") from error


def load_model(source: BinarySource) -> Any:
    """Load a Circle Object API model from a path or binary stream."""

    return model_from_bytes(read_circle_bytes(source))


def load_model_mapped(path: PathLike) -> tuple[Any, CirclePayloadMapping]:
    """Load a model from a regular file whose payloads borrow a read-only mapping.

    The returned mapping owns the file mapping; the model's external ``Buffer.data``
    arrays are views into it. Release the mapping only after the model, or any
    document derived from it without copying payloads, is no longer needed.
    """

    mapping = map_circle_file(path)
    try:
        model = _model_from_buffer(mapping.buffer())
    except BaseException:
        mapping.release()
        raise
    return model, mapping


def _plan_model_layout(model: Any) -> CircleBinaryLayout:
    """Prepare the file layout used by ``save_model``; errors become I/O errors."""

    if model is None or not hasattr(model, "Pack"):
        raise TypeError("Expected a Circle Object API model with a Pack method.")
    try:
        return plan_circle_layout(model, inline_budget=_STREAMING_INLINE_BUDGET)
    except Exception as error:
        raise CircleIOError(f"Failed to serialize the Circle model: {error}") from error


def save_model(
    model: Any,
    destination: BinaryDestination,
    *,
    atomic: bool = True,
) -> None:
    """Serialize and save a Circle Object API model.

    The header is packed and every payload offset is fixed before the first byte
    is written, then header, padding, and payload views are streamed in order.
    No buffer the size of the complete binary is allocated for the appended
    layout; the inline layout writes the header the packer already produced.
    """

    layout = _plan_model_layout(model)
    _write_destination(destination, layout.write_to, atomic=atomic)
