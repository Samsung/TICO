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

"""Immutable, decoded view of one Circle subgraph used by the reference runtime.

The program is built once from the serialized ``.circle`` bytes. It resolves
operator codes, decodes inline constants through :class:`TensorValueCodec`, and
captures the serialized tensor contracts. Nothing here depends on the PyTorch
graph that produced the model; execution reads only what is stored in the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from tico.circle._schema import decode_text, enum_name
from tico.circle.analysis import effective_builtin_code, TensorContract
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleError, CircleValueError
from tico.circle.graph import as_indices, as_list, is_constant_tensor
from tico.circle.runtime.errors import (
    CircleRuntimeValidationError,
    UnsupportedCircleOperatorError,
)
from tico.circle.value import (
    default_tensor_type_registry,
    TensorQuantization,
    TensorTypeSpec,
    TensorValueCodec,
)

OPTIONAL_INDEX = -1


def tensor_type_spec(tensor_type: int) -> TensorTypeSpec:
    """Return the registered storage specification of a Circle tensor type."""

    return default_tensor_type_registry().by_value(int(tensor_type))


def tensor_type_name(tensor_type: int) -> str:
    """Return the symbolic Circle ``TensorType`` name for diagnostics."""

    return enum_name("TensorType", int(tensor_type))


@lru_cache(maxsize=None)
def builtin_operator_name(builtin_code: int) -> str:
    """Return the symbolic ``BuiltinOperator`` name for diagnostics."""

    return enum_name("BuiltinOperator", int(builtin_code))


@dataclass(frozen=True)
class RuntimeTensor:
    """Describe one serialized tensor together with its decoded constant payload."""

    index: int
    name: str
    contract: TensorContract
    dtype: np.dtype
    constant: np.ndarray | None

    @property
    def tensor_type(self) -> int:
        """Return the serialized Circle tensor type."""

        return self.contract.tensor_type

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the serialized placeholder shape."""

        return self.contract.shape

    @property
    def shape_signature(self) -> tuple[int, ...] | None:
        """Return the serialized shape signature, if any."""

        return self.contract.shape_signature

    @property
    def is_constant(self) -> bool:
        """Return whether the tensor carries decoded constant data."""

        return self.constant is not None

    @property
    def is_variable(self) -> bool:
        """Return whether the tensor is a stateful variable."""

        return self.contract.is_variable

    @property
    def is_dynamic(self) -> bool:
        """Return whether any dimension is marked dynamic in the shape signature."""

        signature = self.contract.shape_signature
        return signature is not None and any(dim == -1 for dim in signature)

    @property
    def quantization(self) -> TensorQuantization | None:
        """Return the serialized quantization record, if any."""

        return self.contract.quantization

    @property
    def is_quantized(self) -> bool:
        """Return whether the tensor carries affine quantization parameters."""

        quantization = self.contract.quantization
        return quantization is not None and len(quantization.scale) > 0

    @property
    def is_quantized_integer(self) -> bool:
        """Return whether the tensor is an integer tensor with affine qparams."""

        return self.is_quantized and self.dtype.kind in {"i", "u"}

    def describe(self) -> str:
        """Return a compact human-readable description used in diagnostics."""

        signature = self.contract.shape_signature
        shape = (
            list(self.contract.shape)
            if signature is None
            else [
                dim if sig != -1 else -1
                for dim, sig in zip(self.contract.shape, signature)
            ]
        )
        role = "constant" if self.is_constant else "activation"
        quant = ""
        if self.is_quantized:
            assert self.contract.quantization is not None
            scales = self.contract.quantization.scale
            quant = f", per-{'tensor' if len(scales) == 1 else 'channel'} quantized"
        return (
            f"tensors[{self.index}] {self.name!r} "
            f"{tensor_type_name(self.tensor_type)}{shape} ({role}{quant})"
        )


@dataclass(frozen=True)
class RuntimeOperator:
    """Describe one serialized operator with its resolved builtin code."""

    index: int
    builtin_code: int
    version: int
    custom_code: str | None
    options_type: int
    options: Any
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    @property
    def name(self) -> str:
        """Return the builtin operator name or the custom operator code."""

        if self.custom_code:
            return f"CUSTOM({self.custom_code})"
        return builtin_operator_name(self.builtin_code)


class CircleProgram:
    """Decoded single-subgraph Circle model ready for repeated execution."""

    def __init__(
        self,
        document: CircleDocument,
        *,
        subgraph_index: int = 0,
        codec: TensorValueCodec | None = None,
    ) -> None:
        """Decode tensors, constants, and operators of one subgraph."""

        self.document = document
        self.subgraph_index = int(subgraph_index)
        self._codec = codec or TensorValueCodec()
        model = document.model
        subgraphs = as_list(getattr(model, "subgraphs", None))
        if not subgraphs:
            raise CircleRuntimeValidationError("Circle model has no subgraph.")
        if self.subgraph_index < 0 or self.subgraph_index >= len(subgraphs):
            raise CircleRuntimeValidationError(
                f"Subgraph index {self.subgraph_index} is outside "
                f"0..{len(subgraphs) - 1}."
            )
        subgraph = subgraphs[self.subgraph_index]
        self.subgraph = subgraph
        self.tensors: tuple[RuntimeTensor, ...] = tuple(
            self._decode_tensor(model, subgraph, index, tensor)
            for index, tensor in enumerate(as_list(getattr(subgraph, "tensors", None)))
        )
        self.operators: tuple[RuntimeOperator, ...] = tuple(
            self._decode_operator(model, index, operator)
            for index, operator in enumerate(
                as_list(getattr(subgraph, "operators", None))
            )
        )
        self.inputs: tuple[int, ...] = tuple(
            as_indices(getattr(subgraph, "inputs", None))
        )
        self.outputs: tuple[int, ...] = tuple(
            as_indices(getattr(subgraph, "outputs", None))
        )
        self._validate_interface()

    @classmethod
    def from_bytes(cls, data: bytes, *, subgraph_index: int = 0) -> CircleProgram:
        """Parse serialized Circle bytes and decode the selected subgraph."""

        return cls(CircleDocument.from_bytes(data), subgraph_index=subgraph_index)

    @property
    def subgraph_count(self) -> int:
        """Return the number of subgraphs in the underlying model."""

        return self.document.subgraph_count

    def tensor(self, index: int) -> RuntimeTensor:
        """Return a tensor by index with a descriptive bounds check."""

        if index < 0 or index >= len(self.tensors):
            raise CircleRuntimeValidationError(
                f"Tensor index {index} is outside 0..{len(self.tensors) - 1} "
                f"in subgraphs[{self.subgraph_index}]."
            )
        return self.tensors[index]

    def describe_operator(self, operator: RuntimeOperator) -> str:
        """Return a diagnostic description including operand types and shapes."""

        def describe_index(index: int) -> str:
            if index == OPTIONAL_INDEX:
                return "<absent>"
            if index < 0 or index >= len(self.tensors):
                return f"tensors[{index}] <invalid index>"
            return self.tensors[index].describe()

        inputs = ", ".join(describe_index(index) for index in operator.inputs)
        outputs = ", ".join(describe_index(index) for index in operator.outputs)
        return (
            f"subgraphs[{self.subgraph_index}].operators[{operator.index}] "
            f"{operator.name} (builtin code {operator.builtin_code}, version "
            f"{operator.version}) inputs=[{inputs}] outputs=[{outputs}]"
        )

    def _decode_tensor(
        self,
        model: Any,
        subgraph: Any,
        index: int,
        tensor: Any,
    ) -> RuntimeTensor:
        path = f"subgraphs[{self.subgraph_index}].tensors[{index}]"
        try:
            contract = TensorContract.from_tensor(tensor)
        except CircleValueError as error:
            raise CircleRuntimeValidationError(
                f"{path} has an invalid tensor contract: {error}"
            ) from error
        try:
            spec = tensor_type_spec(contract.tensor_type)
        except CircleValueError as error:
            raise UnsupportedCircleOperatorError(
                f"{path} uses Circle tensor type {contract.tensor_type}, which has "
                "no value codec in the reference runtime."
            ) from error
        name = decode_text(getattr(tensor, "name", ""))

        constant: np.ndarray | None = None
        buffer_index = int(getattr(tensor, "buffer", 0) or 0)
        buffers = as_list(getattr(model, "buffers", None))
        if 0 < buffer_index < len(buffers):
            buffer = buffers[buffer_index]
            if int(getattr(buffer, "offset", 0) or 0) or int(
                getattr(buffer, "size", 0) or 0
            ):
                raise UnsupportedCircleOperatorError(
                    f"{path} references external buffer storage (offset/size), "
                    "which the reference runtime does not read."
                )
        if is_constant_tensor(model, subgraph, index):
            if contract.shape_signature is not None and any(
                dim == -1 for dim in contract.shape_signature
            ):
                raise CircleRuntimeValidationError(
                    f"{path} is a constant with a dynamic shape signature "
                    f"{contract.shape_signature}."
                )
            try:
                value = self._codec.decode_tensor(
                    model,
                    subgraph_index=self.subgraph_index,
                    tensor_index=index,
                )
            except CircleError as error:
                raise CircleRuntimeValidationError(
                    f"{path} constant payload cannot be decoded: {error}"
                ) from error
            constant = value.data
        return RuntimeTensor(
            index=index,
            name=name,
            contract=contract,
            dtype=np.dtype(spec.logical_dtype),
            constant=constant,
        )

    def _decode_operator(
        self,
        model: Any,
        index: int,
        operator: Any,
    ) -> RuntimeOperator:
        path = f"subgraphs[{self.subgraph_index}].operators[{index}]"
        operator_codes = as_list(getattr(model, "operatorCodes", None))
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            raise CircleRuntimeValidationError(
                f"{path} references invalid operator code index {opcode_index}."
            )
        operator_code = operator_codes[opcode_index]
        builtin_code = effective_builtin_code(model, operator)
        custom_code = decode_text(getattr(operator_code, "customCode", None)) or None
        inputs = tuple(as_indices(getattr(operator, "inputs", None)))
        outputs = tuple(as_indices(getattr(operator, "outputs", None)))
        for role, indices in (("input", inputs), ("output", outputs)):
            for position, tensor_index in enumerate(indices):
                if tensor_index == OPTIONAL_INDEX:
                    continue
                if tensor_index < 0 or tensor_index >= len(self.tensors):
                    raise CircleRuntimeValidationError(
                        f"{path} {role} {position} references invalid tensor "
                        f"{tensor_index}."
                    )
        return RuntimeOperator(
            index=index,
            builtin_code=int(builtin_code),
            version=int(getattr(operator_code, "version", 1) or 1),
            custom_code=custom_code,
            options_type=int(getattr(operator, "builtinOptionsType", 0) or 0),
            options=getattr(operator, "builtinOptions", None),
            inputs=inputs,
            outputs=outputs,
        )

    def _validate_interface(self) -> None:
        for role, indices in (("inputs", self.inputs), ("outputs", self.outputs)):
            for position, tensor_index in enumerate(indices):
                if tensor_index < 0 or tensor_index >= len(self.tensors):
                    raise CircleRuntimeValidationError(
                        f"subgraphs[{self.subgraph_index}].{role}[{position}] "
                        f"references invalid tensor {tensor_index}."
                    )
        for tensor_index in self.inputs:
            tensor = self.tensors[tensor_index]
            if tensor.is_constant:
                raise CircleRuntimeValidationError(
                    f"Graph input {tensor.describe()} must not carry constant data."
                )
        producers: dict[int, int] = {}
        for operator in self.operators:
            for tensor_index in operator.outputs:
                if tensor_index == OPTIONAL_INDEX:
                    continue
                previous = producers.get(tensor_index)
                if previous is not None:
                    raise CircleRuntimeValidationError(
                        f"{self.tensors[tensor_index].describe()} is produced by "
                        f"operators {previous} and {operator.index}."
                    )
                producers[tensor_index] = operator.index
                tensor = self.tensors[tensor_index]
                if tensor.is_constant:
                    raise CircleRuntimeValidationError(
                        f"{tensor.describe()} is produced by operator "
                        f"{operator.index} but also carries constant data."
                    )
                if tensor_index in self.inputs:
                    raise CircleRuntimeValidationError(
                        f"{tensor.describe()} is a graph input but is also produced "
                        f"by operator {operator.index}."
                    )


__all__ = [
    "OPTIONAL_INDEX",
    "CircleProgram",
    "RuntimeOperator",
    "RuntimeTensor",
    "builtin_operator_name",
    "tensor_type_name",
    "tensor_type_spec",
]
