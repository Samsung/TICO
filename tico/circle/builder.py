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
from typing import Any, Iterable, Sequence

import numpy as np

from tico.circle._object import create_object, ObjectFactory
from tico.circle._schema import decode_text
from tico.circle.analysis import TensorContract
from tico.circle.errors import CircleRewriteError, CircleValueError
from tico.circle.graph import as_indices, as_list, OPTIONAL_TENSOR_INDEX
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True)
class ConstantKey:
    """Identify a constant by semantic tensor contract and exact storage bytes."""

    contract: TensorContract
    payload: bytes


class ConstantPool:
    """Deduplicate inline buffers globally and constant tensors per subgraph."""

    def __init__(
        self,
        model: Any,
        *,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Index existing inline constants without changing the supplied model."""

        self.model = model
        self.codec = codec or TensorValueCodec()
        self.object_factory = object_factory
        self._buffers: dict[bytes, int] = {}
        self._tensors: dict[tuple[int, ConstantKey], int] = {}
        self._index_existing_objects()

    def add_buffer(self, payload: bytes, *, deduplicate: bool = True) -> int:
        """Add inline bytes and optionally reuse an existing exact payload."""

        normalized = bytes(payload)
        existing = self._buffers.get(normalized)
        if deduplicate and existing is not None:
            return existing

        buffer = create_object("Buffer", self.object_factory)
        buffer.data = np.frombuffer(normalized, dtype=np.uint8).copy()
        if hasattr(buffer, "offset"):
            buffer.offset = 0
        if hasattr(buffer, "size"):
            buffer.size = 0
        buffers = _mutable_list(self.model, "buffers")
        buffers.append(buffer)
        index = len(buffers) - 1
        self._buffers.setdefault(normalized, index)
        return index

    def intern_buffer(self, payload: bytes) -> int:
        """Return a shared inline buffer index for exact payload bytes."""

        return self.add_buffer(payload, deduplicate=True)

    def intern_constant(
        self,
        *,
        subgraph_index: int,
        name: str,
        value: TensorValue,
        contract: TensorContract | None = None,
    ) -> int:
        """Return a constant tensor index, creating storage and metadata if needed."""

        subgraph = _subgraph(self.model, subgraph_index)
        resolved_contract = contract or TensorContract.from_value(value)
        _validate_constant_contract(value, resolved_contract)
        payload = self.codec.encode(value)
        key = ConstantKey(resolved_contract, payload)
        existing = self._tensors.get((subgraph_index, key))
        if existing is not None:
            return existing

        buffer_index = self.intern_buffer(payload)
        tensor_name = _unique_tensor_name(subgraph, name)
        tensor = resolved_contract.make_tensor(
            name=tensor_name,
            buffer_index=buffer_index,
            factory=self.object_factory,
        )
        tensors = _mutable_list(subgraph, "tensors")
        tensors.append(tensor)
        tensor_index = len(tensors) - 1
        self._tensors[(subgraph_index, key)] = tensor_index
        return tensor_index

    def _index_existing_objects(self) -> None:
        """Populate buffer and tensor keys from existing model contents."""

        buffers = as_list(getattr(self.model, "buffers", None))
        if not buffers:
            buffer_zero = create_object("Buffer", self.object_factory)
            buffer_zero.data = None
            self.model.buffers = [buffer_zero]
            buffers = [buffer_zero]

        for buffer_index, buffer in enumerate(buffers):
            if buffer_index == 0:
                continue
            payload = _inline_buffer_payload(buffer)
            if payload is not None:
                self._buffers.setdefault(payload, buffer_index)

        for subgraph_index, subgraph in enumerate(
            as_list(getattr(self.model, "subgraphs", None))
        ):
            for tensor_index, tensor in enumerate(
                as_list(getattr(subgraph, "tensors", None))
            ):
                if bool(getattr(tensor, "isVariable", False)):
                    continue
                buffer_index = int(getattr(tensor, "buffer", 0) or 0)
                if buffer_index <= 0 or buffer_index >= len(buffers):
                    continue
                payload = _inline_buffer_payload(buffers[buffer_index])
                if payload is None:
                    continue
                try:
                    contract = TensorContract.from_tensor(tensor)
                except CircleValueError:
                    continue
                key = ConstantKey(contract, payload)
                self._tensors.setdefault((subgraph_index, key), tensor_index)


class CircleBuilder:
    """Create reusable Circle buffers, tensors, operator codes, and operators safely."""

    def __init__(
        self,
        document_or_model: Any,
        *,
        subgraph_index: int = 0,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
        constant_pool: ConstantPool | None = None,
    ) -> None:
        """Bind a builder to one subgraph while retaining model-global pools."""

        self.model = getattr(document_or_model, "model", document_or_model)
        self.subgraph_index = int(subgraph_index)
        self.subgraph = _subgraph(self.model, self.subgraph_index)
        if constant_pool is not None and constant_pool.model is not self.model:
            raise ValueError("constant_pool must belong to the same Circle model.")
        if constant_pool is not None and codec is not None:
            if codec is not constant_pool.codec:
                raise ValueError(
                    "codec must match the supplied constant_pool codec instance."
                )
        if constant_pool is not None and object_factory is not None:
            if object_factory is not constant_pool.object_factory:
                raise ValueError(
                    "object_factory must match the supplied constant_pool factory."
                )
        self.codec = (
            constant_pool.codec
            if constant_pool is not None
            else codec or TensorValueCodec()
        )
        self.object_factory = (
            constant_pool.object_factory
            if constant_pool is not None
            else object_factory
        )
        self.constant_pool = constant_pool or ConstantPool(
            self.model,
            codec=self.codec,
            object_factory=self.object_factory,
        )

    def add_buffer(self, payload: bytes, *, deduplicate: bool = True) -> int:
        """Add an inline buffer or return an existing exact payload match."""

        return self.constant_pool.add_buffer(
            payload,
            deduplicate=deduplicate,
        )

    def add_tensor(
        self,
        name: str,
        contract: TensorContract,
        *,
        buffer_index: int = 0,
    ) -> int:
        """Append one tensor with a unique name and return its local index."""

        self._validate_buffer_index(buffer_index)
        tensor = contract.make_tensor(
            name=_unique_tensor_name(self.subgraph, name),
            buffer_index=buffer_index,
            factory=self.object_factory,
        )
        tensors = _mutable_list(self.subgraph, "tensors")
        tensors.append(tensor)
        return len(tensors) - 1

    def add_constant(
        self,
        name: str,
        value: TensorValue,
        *,
        contract: TensorContract | None = None,
    ) -> int:
        """Intern one immutable tensor value as a subgraph-local constant."""

        return self.constant_pool.intern_constant(
            subgraph_index=self.subgraph_index,
            name=name,
            value=value,
            contract=contract,
        )

    def find_or_add_operator_code(
        self,
        builtin_code: int,
        *,
        version: int = 1,
        custom_code: str | bytes | None = None,
        deprecated_builtin_code: int | None = None,
    ) -> int:
        """Return an operator-code index matching all serialized identity fields."""

        builtin_code = int(builtin_code)
        version = int(version)
        if builtin_code < 0:
            raise ValueError("builtin_code must not be negative.")
        if version <= 0:
            raise ValueError("Operator code versions must be positive.")
        normalized_custom_code = _normalize_custom_code(custom_code)

        operator_codes = _mutable_list(self.model, "operatorCodes")
        for index, operator_code in enumerate(operator_codes):
            if int(getattr(operator_code, "builtinCode", -1)) != builtin_code:
                continue
            if int(getattr(operator_code, "version", 1) or 1) != version:
                continue
            existing_custom_code = _normalize_custom_code(
                getattr(operator_code, "customCode", None)
            )
            if existing_custom_code == normalized_custom_code:
                return index

        operator_code = create_object("OperatorCode", self.object_factory)
        operator_code.builtinCode = builtin_code
        operator_code.version = version
        operator_code.customCode = normalized_custom_code
        operator_code.deprecatedBuiltinCode = (
            min(127, builtin_code)
            if deprecated_builtin_code is None
            else int(deprecated_builtin_code)
        )
        operator_codes.append(operator_code)
        return len(operator_codes) - 1

    def make_operator(
        self,
        builtin_code: int,
        *,
        inputs: Sequence[int],
        outputs: Sequence[int],
        version: int = 1,
        custom_code: str | bytes | None = None,
        builtin_options_type: int | None = None,
        builtin_options: Any = None,
        builtin_options2_type: int | None = None,
        builtin_options2: Any = None,
        intermediates: Sequence[int] = (),
        mutating_variable_inputs: Sequence[bool] = (),
    ) -> Any:
        """Create a validated operator table without inserting it into the graph."""

        self._validate_tensor_indices(inputs, optional=True, field_name="inputs")
        self._validate_tensor_indices(outputs, optional=False, field_name="outputs")
        self._validate_tensor_indices(
            intermediates,
            optional=True,
            field_name="intermediates",
        )
        if mutating_variable_inputs and len(mutating_variable_inputs) != len(inputs):
            raise ValueError(
                "mutating_variable_inputs must be empty or match the input count."
            )

        operator = create_object("Operator", self.object_factory)
        operator.opcodeIndex = self.find_or_add_operator_code(
            builtin_code,
            version=version,
            custom_code=custom_code,
        )
        operator.inputs = [int(index) for index in inputs]
        operator.outputs = [int(index) for index in outputs]
        operator.intermediates = [int(index) for index in intermediates]
        if hasattr(operator, "mutatingVariableInputs"):
            operator.mutatingVariableInputs = [
                bool(value) for value in mutating_variable_inputs
            ]
        if builtin_options_type is not None or hasattr(
            operator,
            "builtinOptionsType",
        ):
            operator.builtinOptionsType = int(builtin_options_type or 0)
        if builtin_options is not None or hasattr(operator, "builtinOptions"):
            operator.builtinOptions = builtin_options
        if builtin_options2_type is not None or hasattr(
            operator,
            "builtinOptions2Type",
        ):
            operator.builtinOptions2Type = int(builtin_options2_type or 0)
        if builtin_options2 is not None or hasattr(operator, "builtinOptions2"):
            operator.builtinOptions2 = builtin_options2
        return operator

    def add_operator_with_outputs(
        self,
        builtin_code: int,
        *,
        inputs: Sequence[int],
        outputs: Sequence[int],
        version: int = 1,
        custom_code: str | bytes | None = None,
        builtin_options_type: int | None = None,
        builtin_options: Any = None,
        builtin_options2_type: int | None = None,
        builtin_options2: Any = None,
        intermediates: Sequence[int] = (),
        mutating_variable_inputs: Sequence[bool] = (),
    ) -> int:
        """Append an operator that writes already-created output tensors."""

        operator = self.make_operator(
            builtin_code,
            inputs=inputs,
            outputs=outputs,
            version=version,
            custom_code=custom_code,
            builtin_options_type=builtin_options_type,
            builtin_options=builtin_options,
            builtin_options2_type=builtin_options2_type,
            builtin_options2=builtin_options2,
            intermediates=intermediates,
            mutating_variable_inputs=mutating_variable_inputs,
        )
        operators = _mutable_list(self.subgraph, "operators")
        operators.append(operator)
        return len(operators) - 1

    def add_operator(
        self,
        builtin_code: int,
        *,
        inputs: Sequence[int],
        output_contracts: Sequence[TensorContract],
        output_names: Sequence[str] | None = None,
        version: int = 1,
        custom_code: str | bytes | None = None,
        builtin_options_type: int | None = None,
        builtin_options: Any = None,
        builtin_options2_type: int | None = None,
        builtin_options2: Any = None,
        intermediates: Sequence[int] = (),
        mutating_variable_inputs: Sequence[bool] = (),
    ) -> tuple[int, ...]:
        """Append an operator and create all output tensors atomically."""

        contracts = tuple(output_contracts)
        names = (
            tuple(output_names)
            if output_names is not None
            else tuple(
                f"operator_{len(as_list(self.subgraph.operators))}_output_{position}"
                for position in range(len(contracts))
            )
        )
        if len(names) != len(contracts):
            raise ValueError("output_names must match output_contracts in length.")
        if any(not name for name in names):
            raise ValueError("Output tensor names must not be empty.")
        self._validate_tensor_indices(inputs, optional=True, field_name="inputs")
        self._validate_tensor_indices(
            intermediates,
            optional=True,
            field_name="intermediates",
        )

        tensors = _mutable_list(self.subgraph, "tensors")
        original_tensor_count = len(tensors)
        output_indices: list[int] = []
        reserved_names: set[str] = set()
        try:
            for name, contract in zip(names, contracts):
                unique_name = _unique_tensor_name(
                    self.subgraph,
                    name,
                    additionally_reserved=reserved_names,
                )
                reserved_names.add(unique_name)
                tensor = contract.make_tensor(
                    name=unique_name,
                    buffer_index=0,
                    factory=self.object_factory,
                )
                tensors.append(tensor)
                output_indices.append(len(tensors) - 1)

            self.add_operator_with_outputs(
                builtin_code,
                inputs=inputs,
                outputs=output_indices,
                version=version,
                custom_code=custom_code,
                builtin_options_type=builtin_options_type,
                builtin_options=builtin_options,
                builtin_options2_type=builtin_options2_type,
                builtin_options2=builtin_options2,
                intermediates=intermediates,
                mutating_variable_inputs=mutating_variable_inputs,
            )
        except Exception:
            del tensors[original_tensor_count:]
            raise
        return tuple(output_indices)

    def replace_operator(
        self,
        operator_index: int,
        replacement: Any,
        *,
        require_same_outputs: bool = True,
    ) -> Any:
        """Replace one operator after validating references and outputs."""

        operators = _mutable_list(self.subgraph, "operators")
        if operator_index < 0 or operator_index >= len(operators):
            raise IndexError(
                f"Operator index {operator_index} is outside 0..{len(operators) - 1}."
            )
        self._validate_operator(replacement)
        previous = operators[operator_index]
        if require_same_outputs and as_indices(previous.outputs) != as_indices(
            replacement.outputs
        ):
            raise CircleRewriteError(
                "Replacement operators must preserve output tensor indices."
            )
        operators[operator_index] = replacement
        return previous

    def _validate_operator(self, operator: Any) -> None:
        """Check one operator's opcode and tensor index references."""

        operator_codes = as_list(getattr(self.model, "operatorCodes", None))
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            raise CircleRewriteError(
                f"Operator references invalid opcode index {opcode_index}."
            )
        self._validate_tensor_indices(
            as_indices(getattr(operator, "inputs", None)),
            optional=True,
            field_name="inputs",
        )
        self._validate_tensor_indices(
            as_indices(getattr(operator, "outputs", None)),
            optional=False,
            field_name="outputs",
        )
        self._validate_tensor_indices(
            as_indices(getattr(operator, "intermediates", None)),
            optional=True,
            field_name="intermediates",
        )

    def _validate_tensor_indices(
        self,
        indices: Iterable[int],
        *,
        optional: bool,
        field_name: str,
    ) -> None:
        """Validate a sequence of subgraph-local tensor references."""

        tensor_count = len(as_list(getattr(self.subgraph, "tensors", None)))
        for position, raw_index in enumerate(indices):
            index = int(raw_index)
            if optional and index == OPTIONAL_TENSOR_INDEX:
                continue
            if index < 0 or index >= tensor_count:
                raise CircleRewriteError(
                    f"Operator {field_name}[{position}] references tensor {index}, "
                    f"but the valid range is 0..{tensor_count - 1}."
                )

    def _validate_buffer_index(self, buffer_index: int) -> None:
        """Validate one model-global buffer reference."""

        buffers = as_list(getattr(self.model, "buffers", None))
        if buffer_index < 0 or buffer_index >= len(buffers):
            raise CircleRewriteError(
                f"Buffer index {buffer_index} is outside 0..{len(buffers) - 1}."
            )


def _subgraph(model: Any, index: int) -> Any:
    """Return one subgraph with a descriptive bounds check."""

    subgraphs = as_list(getattr(model, "subgraphs", None))
    if index < 0 or index >= len(subgraphs):
        raise IndexError(f"Subgraph index {index} is outside 0..{len(subgraphs) - 1}.")
    return subgraphs[index]


def _mutable_list(owner: Any, field_name: str) -> list[Any]:
    """Return a mutable list field, normalizing generated vectors when necessary."""

    value = getattr(owner, field_name, None)
    if isinstance(value, list):
        return value
    normalized = as_list(value)
    setattr(owner, field_name, normalized)
    return normalized


def _inline_buffer_payload(buffer: Any) -> bytes | None:
    """Return inline bytes or None for unresolved external or absent payloads."""

    if int(getattr(buffer, "offset", 0) or 0) or int(getattr(buffer, "size", 0) or 0):
        return None
    data = getattr(buffer, "data", None)
    if data is None:
        return None
    try:
        array = np.asarray(data, dtype=np.uint8)
    except (TypeError, ValueError):
        return None
    return bytes(np.ascontiguousarray(array).reshape(-1))


def _unique_tensor_name(
    subgraph: Any,
    requested: str,
    *,
    additionally_reserved: set[str] | None = None,
) -> str:
    """Return a stable unique tensor name derived from a requested prefix."""

    if not requested:
        raise ValueError("Circle tensor names must not be empty.")
    existing = {
        decode_text(getattr(tensor, "name", ""))
        for tensor in as_list(getattr(subgraph, "tensors", None))
    }
    if additionally_reserved:
        existing.update(additionally_reserved)
    if requested not in existing:
        return requested
    suffix = 0
    while f"{requested}_{suffix}" in existing:
        suffix += 1
    return f"{requested}_{suffix}"


def _normalize_custom_code(value: str | bytes | None) -> str | None:
    """Normalize absent, byte, and string custom operator codes for comparison."""

    if value is None:
        return None
    normalized = decode_text(value)
    return normalized or None


def _validate_constant_contract(
    value: TensorValue,
    contract: TensorContract,
) -> None:
    """Require a constant contract to agree with the logical value exactly."""

    if contract.is_variable:
        raise CircleValueError("Constant tensors cannot be marked as variable.")
    if contract.tensor_type != value.tensor_type:
        raise CircleValueError(
            "Constant tensor type must match the TensorValue tensor type."
        )
    if contract.shape != value.shape:
        raise CircleValueError(
            "Constant tensor shape must match the TensorValue shape."
        )
    if contract.quantization != value.quantization:
        raise CircleValueError(
            "Constant tensor quantization must match the TensorValue quantization."
        )
