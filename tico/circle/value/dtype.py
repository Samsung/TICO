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

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from tico.circle._schema import circle_schema
from tico.circle.errors import CircleValueError


@dataclass(frozen=True)
class TensorTypeSpec:
    """Describe the logical and serialized representation of one Circle tensor type."""

    name: str
    tensor_type: int
    logical_dtype: np.dtype[Any]
    storage_dtype: np.dtype[Any]
    bits_per_element: int
    signed: bool | None = None
    packed: bool = False

    def __post_init__(self) -> None:
        """Normalize dtypes and validate the representation contract."""

        logical_dtype = np.dtype(self.logical_dtype)
        storage_dtype = np.dtype(self.storage_dtype)
        object.__setattr__(self, "tensor_type", int(self.tensor_type))
        object.__setattr__(self, "logical_dtype", logical_dtype)
        object.__setattr__(self, "storage_dtype", storage_dtype)
        if not self.name:
            raise ValueError("Tensor type names must not be empty.")
        if self.bits_per_element <= 0:
            raise ValueError("bits_per_element must be positive.")
        if self.packed:
            if self.bits_per_element != 4:
                raise ValueError(
                    "The common packed codec currently supports four-bit types only."
                )
            if storage_dtype != np.dtype(np.uint8):
                raise ValueError("Packed four-bit tensor types require uint8 storage.")
        else:
            if self.bits_per_element != storage_dtype.itemsize * 8:
                raise ValueError(
                    "Dense bits_per_element must match the storage dtype width."
                )
            if logical_dtype.newbyteorder("=") != storage_dtype.newbyteorder("="):
                raise ValueError("Dense logical and storage dtypes must match exactly.")

    @property
    def byte_width(self) -> int | None:
        """Return bytes per logical element for byte-aligned tensor types."""

        if self.packed or self.bits_per_element % 8 != 0:
            return None
        return self.bits_per_element // 8

    def storage_size(self, element_count: int) -> int:
        """Return the serialized byte count for a logical element count."""

        if element_count < 0:
            raise ValueError("element_count must not be negative.")
        return (element_count * self.bits_per_element + 7) // 8


class TensorTypeRegistry:
    """Map Circle tensor enum values and names to explicit storage specifications."""

    def __init__(self, specs: Iterable[TensorTypeSpec] = ()) -> None:
        """Create a registry and reject duplicate names or enum values."""

        self._by_value: dict[int, TensorTypeSpec] = {}
        self._by_name: dict[str, TensorTypeSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: TensorTypeSpec) -> None:
        """Register one tensor type specification."""

        value = int(spec.tensor_type)
        name = spec.name.upper()
        if value in self._by_value:
            raise ValueError(f"Duplicate Circle tensor type value: {value}.")
        if name in self._by_name:
            raise ValueError(f"Duplicate Circle tensor type name: {name!r}.")
        self._by_value[value] = spec
        self._by_name[name] = spec

    def by_value(self, tensor_type: int) -> TensorTypeSpec:
        """Return a specification by Circle tensor enum value."""

        value = int(tensor_type)
        try:
            return self._by_value[value]
        except KeyError as error:
            raise CircleValueError(
                f"Circle tensor type {value} has no registered value codec."
            ) from error

    def by_name(self, name: str) -> TensorTypeSpec:
        """Return a specification by symbolic Circle tensor type name."""

        normalized = name.upper()
        try:
            return self._by_name[normalized]
        except KeyError as error:
            raise CircleValueError(
                f"Circle tensor type {name!r} has no registered value codec."
            ) from error

    def get(self, tensor_type: int) -> TensorTypeSpec | None:
        """Return a specification when one is registered for the enum value."""

        return self._by_value.get(int(tensor_type))

    @property
    def specs(self) -> tuple[TensorTypeSpec, ...]:
        """Return all registered specifications ordered by enum value."""

        return tuple(self._by_value[value] for value in sorted(self._by_value))


def circle_tensor_type_value(name: str) -> int:
    """Return a generated TensorType enum value by symbolic name."""

    schema = circle_schema()
    enum_module = getattr(schema, "TensorType", None)
    enum_type = (
        getattr(enum_module, "TensorType", None) if enum_module is not None else None
    )
    if enum_type is None:
        enum_type = getattr(schema, "TensorType", None)
    if enum_type is None or not hasattr(enum_type, name):
        raise CircleValueError(f"Circle schema does not provide TensorType.{name}.")
    return int(getattr(enum_type, name))


@lru_cache(maxsize=1)
def default_tensor_type_registry() -> TensorTypeRegistry:
    """Return codecs for the dense and four-bit tensor types used by TICO."""

    registry = TensorTypeRegistry()
    dense_specs = (
        ("FLOAT32", np.float32, 32, None),
        ("FLOAT16", np.float16, 16, None),
        ("FLOAT64", np.float64, 64, None),
        ("INT8", np.int8, 8, True),
        ("UINT8", np.uint8, 8, False),
        ("INT16", np.int16, 16, True),
        ("UINT16", np.uint16, 16, False),
        ("INT32", np.int32, 32, True),
        ("UINT32", np.uint32, 32, False),
        ("INT64", np.int64, 64, True),
        ("UINT64", np.uint64, 64, False),
        ("BOOL", np.bool_, 8, None),
        ("COMPLEX64", np.complex64, 64, None),
        ("COMPLEX128", np.complex128, 128, None),
        ("BFLOAT16", np.uint16, 16, None),
    )
    for name, dtype, bits, signed in dense_specs:
        try:
            value = circle_tensor_type_value(name)
        except CircleValueError:
            continue
        registry.register(
            TensorTypeSpec(
                name=name,
                tensor_type=value,
                logical_dtype=np.dtype(dtype),
                storage_dtype=np.dtype(dtype),
                bits_per_element=bits,
                signed=signed,
            )
        )

    packed_specs = (
        ("INT4", np.int8, True),
        ("UINT4", np.uint8, False),
    )
    for name, logical_dtype, signed in packed_specs:
        try:
            value = circle_tensor_type_value(name)
        except CircleValueError:
            continue
        registry.register(
            TensorTypeSpec(
                name=name,
                tensor_type=value,
                logical_dtype=np.dtype(logical_dtype),
                storage_dtype=np.dtype(np.uint8),
                bits_per_element=4,
                signed=signed,
                packed=True,
            )
        )
    return registry
