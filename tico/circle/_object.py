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

import copy
import dataclasses
import struct
from collections.abc import Callable, Mapping, Sequence, Set
from typing import Any, Protocol, TypeAlias

import numpy as np

from tico.circle._schema import object_api_type
from tico.circle.errors import CircleValueError

FrozenValue: TypeAlias = Any


class ObjectFactory(Protocol):
    """Create a generated Circle Object API table by its schema name."""

    def __call__(self, table_name: str) -> Any:
        """Return a mutable table instance for the requested schema name."""

        ...


def create_object(
    table_name: str,
    factory: ObjectFactory | None = None,
) -> Any:
    """Create an Object API table through the generated schema or a test factory."""

    if not table_name:
        raise ValueError("table_name must not be empty.")
    if factory is not None:
        value = factory(table_name)
    else:
        value = object_api_type(table_name)()
    if value is None:
        raise CircleValueError(
            f"Object factory returned None for Circle table {table_name!r}."
        )
    return value


def clone_object(value: Any) -> Any:
    """Return a deep, independently mutable copy of an Object API value."""

    return copy.deepcopy(value)


def vector_as_tuple(value: Any) -> tuple[Any, ...]:
    """Normalize a generated vector field to an immutable Python tuple."""

    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return tuple(value.reshape(-1).tolist())
    try:
        return tuple(value)
    except TypeError as error:
        raise CircleValueError(
            f"Expected a generated vector field, but received {type(value).__name__}."
        ) from error


def freeze_object(value: Any) -> FrozenValue:
    """Convert an arbitrary Object API value into a stable hashable fingerprint.

    The function preserves floating-point bit patterns, NumPy dtype and shape
    information, mapping keys, dataclass fields, slots, and public object attributes.
    Cyclic object graphs are rejected because Circle Object API tables are expected to
    form trees at their value-bearing leaves.
    """

    return _freeze_object(value, active=set())


def _freeze_object(value: Any, *, active: set[int]) -> FrozenValue:
    """Implement recursive object freezing while detecting cycles."""

    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        return ("float64", struct.pack("<d", value))
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return bytes(value)
    if isinstance(value, np.generic):
        array = np.asarray(value)
        return (
            "numpy-scalar",
            array.dtype.str,
            bytes(array.reshape(-1).view(np.uint8)),
        )
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        shape = tuple(int(dimension) for dimension in array.shape)
        if array.dtype.hasobject:
            identity = id(value)
            if identity in active:
                raise CircleValueError(
                    "Cyclic Object API values cannot be fingerprinted."
                )
            active.add(identity)
            try:
                return (
                    "numpy-object-array",
                    array.dtype.str,
                    shape,
                    *tuple(
                        _freeze_object(item, active=active)
                        for item in array.reshape(-1)
                    ),
                )
            finally:
                active.remove(identity)
        return (
            "numpy-array",
            array.dtype.str,
            shape,
            bytes(array.reshape(-1).view(np.uint8)),
        )

    identity = id(value)
    if identity in active:
        raise CircleValueError("Cyclic Object API values cannot be fingerprinted.")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            frozen_items = [
                (
                    _freeze_object(key, active=active),
                    _freeze_object(item, active=active),
                )
                for key, item in value.items()
            ]
            frozen_items.sort(key=repr)
            return ("mapping", *tuple(frozen_items))
        if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
            frozen_items = [_freeze_object(item, active=active) for item in value]
            frozen_items.sort(key=repr)
            return ("set", *tuple(frozen_items))
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray, memoryview),
        ):
            return (
                "sequence",
                *tuple(_freeze_object(item, active=active) for item in value),
            )
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = tuple(
                (
                    field.name,
                    _freeze_object(getattr(value, field.name), active=active),
                )
                for field in dataclasses.fields(value)
            )
            return ("dataclass", type(value).__qualname__, *fields)

        attributes = _public_attributes(value)
        if attributes:
            fields = tuple(
                (
                    name,
                    _freeze_object(item, active=active),
                )
                for name, item in attributes
            )
            return ("object", type(value).__qualname__, *fields)
    finally:
        active.remove(identity)

    raise CircleValueError(
        f"Unsupported Object API fingerprint value: {type(value).__name__}."
    )


def _public_attributes(value: Any) -> tuple[tuple[str, Any], ...]:
    """Return deterministic public data attributes from a generated object."""

    names: set[str] = set()
    try:
        names.update(vars(value))
    except TypeError:
        pass

    for owner in type(value).__mro__:
        slots = getattr(owner, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.update(str(name) for name in slots)

    if not names:
        names.update(name for name in dir(value) if not name.startswith("_"))

    attributes: list[tuple[str, Any]] = []
    for name in sorted(names):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if isinstance(item, Callable) or callable(item):
            continue
        attributes.append((name, item))
    return tuple(attributes)
