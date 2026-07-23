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

import os
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

from tico.circle._schema import accessor_api_type, object_api_type
from tico.circle.errors import CircleIOError

PathLike: TypeAlias = str | os.PathLike[str]
BinarySource: TypeAlias = PathLike | BinaryIO
BinaryDestination: TypeAlias = PathLike | BinaryIO

CIRCLE_FILE_IDENTIFIER = b"CIR0"


def _load_flatbuffers() -> Any:
    """Import the FlatBuffers runtime lazily."""

    return import_module("flatbuffers")


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


def write_circle_bytes(
    data: bytes,
    destination: BinaryDestination,
    *,
    atomic: bool = True,
) -> None:
    """Write Circle binary data to a path, standard output, or binary stream."""

    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, received {type(data).__name__}.")

    if not isinstance(destination, (str, os.PathLike)):
        try:
            destination.write(data)
            return
        except OSError as error:
            raise CircleIOError(
                "Failed to write Circle data to the output stream."
            ) from error

    path_text = os.fspath(destination)
    if path_text == "-":
        try:
            sys.stdout.buffer.write(data)
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
            path.write_bytes(data)
            return
        except OSError as error:
            raise CircleIOError(f"Failed to write Circle model to {path}.") from error

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
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise CircleIOError(f"Failed to write Circle model to {path}.") from error


def model_from_bytes(data: bytes) -> Any:
    """Deserialize Circle binary data into the generated Object API model."""

    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, received {type(data).__name__}.")
    if not data:
        raise CircleIOError("Circle input is empty.")
    if len(data) < 8 or data[4:8] != CIRCLE_FILE_IDENTIFIER:
        raise CircleIOError(
            "Circle input does not contain the expected CIR0 file identifier."
        )

    try:
        accessor_type = accessor_api_type("Model")
        root = accessor_type.GetRootAsModel(bytearray(data), 0)
        model_type = object_api_type("Model")
        if hasattr(model_type, "InitFromObj"):
            return model_type.InitFromObj(root)
        if hasattr(root, "UnPack"):
            return root.UnPack()
        raise RuntimeError("The Circle schema does not expose an Object API unpacker.")
    except Exception as error:
        if isinstance(error, CircleIOError):
            raise
        raise CircleIOError("Failed to deserialize Circle binary data.") from error


def model_to_bytes(model: Any) -> bytes:
    """Serialize a generated Circle Object API model into binary data."""

    if model is None or not hasattr(model, "Pack"):
        raise TypeError("Expected a Circle Object API model with a Pack method.")

    try:
        flatbuffers = _load_flatbuffers()
        builder = flatbuffers.Builder(1024)
        root_offset = model.Pack(builder)
        builder.Finish(root_offset, CIRCLE_FILE_IDENTIFIER)
        return bytes(builder.Output())
    except Exception as error:
        raise CircleIOError("Failed to serialize the Circle model.") from error


def load_model(source: BinarySource) -> Any:
    """Load a Circle Object API model from a path or binary stream."""

    return model_from_bytes(read_circle_bytes(source))


def save_model(
    model: Any,
    destination: BinaryDestination,
    *,
    atomic: bool = True,
) -> None:
    """Serialize and save a Circle Object API model."""

    write_circle_bytes(model_to_bytes(model), destination, atomic=atomic)
