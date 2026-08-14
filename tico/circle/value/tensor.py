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

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from tico.circle._object import (
    clone_object,
    create_object,
    freeze_object,
    FrozenValue,
    ObjectFactory,
    vector_as_tuple,
)
from tico.circle.errors import CircleValueError


@dataclass(frozen=True)
class TensorQuantization:
    """Represent a serializable Circle quantization record as immutable values."""

    scale: tuple[float, ...] = ()
    zero_point: tuple[int, ...] = ()
    minimum: tuple[float, ...] = ()
    maximum: tuple[float, ...] = ()
    quantized_dimension: int = 0
    details_type: int = 0
    details: Any = field(default=None, repr=False, compare=False, hash=False)
    _details_fingerprint: FrozenValue = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize vectors and own an independent copy of union details."""

        object.__setattr__(self, "scale", tuple(float(value) for value in self.scale))
        object.__setattr__(
            self,
            "zero_point",
            tuple(int(value) for value in self.zero_point),
        )
        object.__setattr__(
            self,
            "minimum",
            tuple(float(value) for value in self.minimum),
        )
        object.__setattr__(
            self,
            "maximum",
            tuple(float(value) for value in self.maximum),
        )
        object.__setattr__(
            self,
            "quantized_dimension",
            int(self.quantized_dimension),
        )
        object.__setattr__(self, "details_type", int(self.details_type))
        owned_details = clone_object(self.details)
        object.__setattr__(self, "details", owned_details)
        object.__setattr__(
            self,
            "_details_fingerprint",
            freeze_object(owned_details),
        )
        self._validate()

    @property
    def details_fingerprint(self) -> FrozenValue:
        """Return a stable comparison key for optional quantization details."""

        return self._details_fingerprint

    @classmethod
    def from_object(cls, value: Any) -> TensorQuantization | None:
        """Create an immutable quantization record from an Object API table."""

        if value is None:
            return None
        return cls(
            scale=tuple(float(item) for item in vector_as_tuple(value.scale)),
            zero_point=tuple(int(item) for item in vector_as_tuple(value.zeroPoint)),
            minimum=tuple(float(item) for item in vector_as_tuple(value.min)),
            maximum=tuple(float(item) for item in vector_as_tuple(value.max)),
            quantized_dimension=int(getattr(value, "quantizedDimension", 0) or 0),
            details_type=int(getattr(value, "detailsType", 0) or 0),
            details=clone_object(getattr(value, "details", None)),
        )

    def to_object(self, factory: ObjectFactory | None = None) -> Any:
        """Create a mutable QuantizationParameters table from this record."""

        value = create_object("QuantizationParameters", factory)
        value.scale = list(self.scale)
        value.zeroPoint = list(self.zero_point)
        value.min = list(self.minimum)
        value.max = list(self.maximum)
        value.quantizedDimension = self.quantized_dimension
        value.detailsType = self.details_type
        value.details = clone_object(self.details)
        return value

    def _validate(self) -> None:
        """Reject incomplete affine quantization and inconsistent ranges."""

        if bool(self.scale) != bool(self.zero_point):
            raise CircleValueError(
                "Quantization scale and zero-point vectors must both be present."
            )
        if self.scale and len(self.scale) != len(self.zero_point):
            raise CircleValueError(
                "Quantization scale and zero-point vectors must have equal length."
            )
        if any(scale < 0.0 for scale in self.scale):
            raise CircleValueError("Quantization scales must not be negative.")
        if bool(self.minimum) != bool(self.maximum):
            raise CircleValueError(
                "Quantization minimum and maximum vectors must both be present."
            )
        if self.minimum and len(self.minimum) != len(self.maximum):
            raise CircleValueError(
                "Quantization minimum and maximum vectors must have equal length."
            )
        if self.quantized_dimension < 0:
            raise CircleValueError("quantized_dimension must not be negative.")
        if self.details_type == 0 and self.details is not None:
            raise CircleValueError(
                "Quantization details require a non-zero details_type."
            )


@dataclass(frozen=True)
class TensorValue:
    """Own a concrete immutable tensor value and its Circle storage type."""

    tensor_type: int
    shape: tuple[int, ...]
    data: np.ndarray[Any, Any] = field(repr=False, compare=False, hash=False)
    quantization: TensorQuantization | None = None
    _data_fingerprint: tuple[str, tuple[int, ...], bytes] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Normalize shape and copy data into a read-only contiguous array."""

        tensor_type = int(self.tensor_type)
        shape = tuple(int(dimension) for dimension in self.shape)
        if any(dimension < 0 for dimension in shape):
            raise CircleValueError(
                f"Concrete tensor shapes must not contain negative values: {shape}."
            )

        array = np.asarray(self.data)
        if array.dtype.hasobject or array.dtype.kind in {"S", "U"}:
            raise CircleValueError(
                "TensorValue does not support object or string storage."
            )
        expected_elements = _element_count(shape)
        if array.size != expected_elements:
            raise CircleValueError(
                f"Tensor shape {shape} requires {expected_elements} elements, "
                f"but data provides {array.size}."
            )
        array = np.ascontiguousarray(array.reshape(shape)).copy()
        array.setflags(write=False)

        fingerprint = (
            array.dtype.str,
            shape,
            bytes(array.reshape(-1).view(np.uint8)),
        )
        object.__setattr__(self, "tensor_type", tensor_type)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "data", array)
        object.__setattr__(self, "_data_fingerprint", fingerprint)

    @classmethod
    def from_values(
        cls,
        tensor_type: int,
        values: Any,
        *,
        shape: Iterable[int] | None = None,
        dtype: np.dtype[Any] | type[Any] | None = None,
        quantization: TensorQuantization | None = None,
    ) -> TensorValue:
        """Create a tensor value from array-like values with optional reshaping."""

        array = np.asarray(values, dtype=dtype)
        resolved_shape = (
            tuple(int(dimension) for dimension in array.shape)
            if shape is None
            else tuple(int(dimension) for dimension in shape)
        )
        return cls(
            tensor_type=int(tensor_type),
            shape=resolved_shape,
            data=array,
            quantization=quantization,
        )

    @property
    def element_count(self) -> int:
        """Return the number of logical elements in the tensor."""

        return _element_count(self.shape)

    @property
    def nbytes(self) -> int:
        """Return the in-memory byte count of the logical NumPy representation."""

        return int(self.data.nbytes)

    def mutable_copy(self) -> np.ndarray[Any, Any]:
        """Return a writable copy of the logical tensor data."""

        return self.data.copy()


def _element_count(shape: tuple[int, ...]) -> int:
    """Return the product of a concrete shape, including scalar and empty tensors."""

    count = 1
    for dimension in shape:
        count *= dimension
    return count
