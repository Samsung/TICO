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

from tico.circle._object import (
    clone_object,
    create_object,
    freeze_object,
    FrozenValue,
    ObjectFactory,
    vector_as_tuple,
)
from tico.circle.errors import CircleValueError
from tico.circle.value.tensor import TensorQuantization, TensorValue


@dataclass(frozen=True)
class TensorContract:
    """Describe externally observable metadata for one Circle tensor."""

    tensor_type: int
    shape: tuple[int, ...]
    shape_signature: tuple[int, ...] | None = None
    quantization: TensorQuantization | None = None
    is_variable: bool = False
    sparsity: Any = field(default=None, repr=False, compare=False, hash=False)
    has_rank: bool | None = None
    variant_tensors: Any = field(
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )
    _sparsity_fingerprint: FrozenValue = field(init=False, repr=False)
    _variant_fingerprint: FrozenValue = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize metadata, own nested objects, and validate shape semantics."""

        shape = tuple(int(dimension) for dimension in self.shape)
        signature = (
            None
            if self.shape_signature is None
            else tuple(int(dimension) for dimension in self.shape_signature)
        )
        owned_sparsity = clone_object(self.sparsity)
        owned_variants = clone_object(self.variant_tensors)

        object.__setattr__(self, "tensor_type", int(self.tensor_type))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "shape_signature", signature)
        object.__setattr__(self, "is_variable", bool(self.is_variable))
        object.__setattr__(self, "sparsity", owned_sparsity)
        object.__setattr__(self, "variant_tensors", owned_variants)
        object.__setattr__(
            self,
            "_sparsity_fingerprint",
            freeze_object(owned_sparsity),
        )
        object.__setattr__(
            self,
            "_variant_fingerprint",
            freeze_object(owned_variants),
        )
        if self.has_rank is False:
            object.__setattr__(self, "has_rank", None)
        elif self.has_rank is not None:
            object.__setattr__(self, "has_rank", True)
        self._validate()

    @property
    def sparsity_fingerprint(self) -> FrozenValue:
        """Return a stable comparison key for optional sparsity metadata."""

        return self._sparsity_fingerprint

    @property
    def variant_fingerprint(self) -> FrozenValue:
        """Return a stable comparison key for optional variant tensor metadata."""

        return self._variant_fingerprint

    @property
    def rank(self) -> int:
        """Return the concrete tensor rank."""

        return len(self.shape)

    @property
    def element_count(self) -> int:
        """Return the number of elements in the concrete placeholder shape."""

        count = 1
        for dimension in self.shape:
            count *= dimension
        return count

    @classmethod
    def from_tensor(cls, tensor: Any) -> TensorContract:
        """Capture semantic tensor metadata without its name or buffer."""

        raw_signature = getattr(tensor, "shapeSignature", None)
        signature_values = vector_as_tuple(raw_signature)
        signature = (
            tuple(int(value) for value in signature_values)
            if signature_values
            else None
        )
        has_rank = (
            True
            if hasattr(tensor, "hasRank") and bool(getattr(tensor, "hasRank"))
            else None
        )
        raw_variants = getattr(tensor, "variantTensors", None)
        variant_tensors = (
            clone_object(raw_variants) if vector_as_tuple(raw_variants) else None
        )
        return cls(
            tensor_type=int(getattr(tensor, "type", -1)),
            shape=tuple(int(value) for value in vector_as_tuple(tensor.shape)),
            shape_signature=signature,
            quantization=TensorQuantization.from_object(
                getattr(tensor, "quantization", None)
            ),
            is_variable=bool(getattr(tensor, "isVariable", False)),
            sparsity=clone_object(getattr(tensor, "sparsity", None)),
            has_rank=has_rank,
            variant_tensors=variant_tensors,
        )

    @classmethod
    def from_value(
        cls,
        value: TensorValue,
        *,
        shape_signature: Iterable[int] | None = None,
        is_variable: bool = False,
        sparsity: Any = None,
        has_rank: bool | None = None,
        variant_tensors: Any = None,
    ) -> TensorContract:
        """Create a tensor contract matching a concrete tensor value."""

        return cls(
            tensor_type=value.tensor_type,
            shape=value.shape,
            shape_signature=(
                None
                if shape_signature is None
                else tuple(int(dimension) for dimension in shape_signature)
            ),
            quantization=value.quantization,
            is_variable=is_variable,
            sparsity=sparsity,
            has_rank=has_rank,
            variant_tensors=variant_tensors,
        )

    def apply_to_tensor(
        self,
        tensor: Any,
        *,
        factory: ObjectFactory | None = None,
    ) -> None:
        """Overwrite semantic metadata while preserving name and buffer fields."""

        tensor.type = self.tensor_type
        tensor.shape = list(self.shape)
        tensor.shapeSignature = (
            None if self.shape_signature is None else list(self.shape_signature)
        )
        tensor.quantization = (
            None if self.quantization is None else self.quantization.to_object(factory)
        )
        tensor.isVariable = self.is_variable
        if hasattr(tensor, "sparsity") or self.sparsity is not None:
            tensor.sparsity = clone_object(self.sparsity)
        if self.has_rank is not None and hasattr(tensor, "hasRank"):
            tensor.hasRank = self.has_rank
        if hasattr(tensor, "variantTensors") or self.variant_tensors is not None:
            tensor.variantTensors = clone_object(self.variant_tensors)

    def make_tensor(
        self,
        *,
        name: str,
        buffer_index: int = 0,
        factory: ObjectFactory | None = None,
    ) -> Any:
        """Create a generated Tensor table carrying this contract."""

        if not name:
            raise ValueError("Circle tensor names must not be empty.")
        if buffer_index < 0:
            raise ValueError("buffer_index must not be negative.")
        tensor = create_object("Tensor", factory)
        tensor.name = name
        tensor.buffer = int(buffer_index)
        self.apply_to_tensor(tensor, factory=factory)
        return tensor

    def matches_tensor(self, tensor: Any) -> bool:
        """Return whether a generated tensor carries this exact semantic contract."""

        return self == TensorContract.from_tensor(tensor)

    def _validate(self) -> None:
        """Reject malformed concrete shapes, signatures, and quantization axes."""

        if any(dimension < 0 for dimension in self.shape):
            raise CircleValueError(
                f"Concrete tensor shapes must not contain negative values: "
                f"{self.shape}."
            )
        signature = self.shape_signature
        if signature is not None:
            if not signature:
                raise CircleValueError(
                    "shape_signature must use None instead of an empty tuple."
                )
            if len(signature) != len(self.shape):
                raise CircleValueError(
                    "shape and shape_signature must have the same rank."
                )
            for concrete, symbolic in zip(self.shape, signature):
                if symbolic < -1:
                    raise CircleValueError(
                        f"Invalid shape signature dimension: {symbolic}."
                    )
                if symbolic == -1:
                    if concrete != 1:
                        raise CircleValueError(
                            "Dynamic shape signature dimensions require concrete "
                            "placeholder value 1."
                        )
                elif concrete != symbolic:
                    raise CircleValueError(
                        "Static shape signature dimensions must match shape values."
                    )
        if self.quantization is not None and self.quantization.scale:
            if self.rank == 0 and len(self.quantization.scale) > 1:
                raise CircleValueError(
                    "Scalar tensors cannot use per-axis quantization."
                )
            if (
                len(self.quantization.scale) > 1
                and self.quantization.quantized_dimension >= self.rank
            ):
                raise CircleValueError(
                    "Per-axis quantized_dimension must be within the tensor rank."
                )
