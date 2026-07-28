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
from typing import Any, Callable

import numpy as np
from circle_schema import circle

from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list, OPTIONAL_TENSOR_INDEX


@dataclass(frozen=True)
class CircleEvaluationResult:
    """Store graph outputs and all tensor values produced during evaluation."""

    outputs: tuple[np.ndarray, ...]
    tensor_values: dict[int, np.ndarray]


def _enum_value(enum_module_name: str, member_name: str) -> int:
    """Return a generated Circle enum value with a descriptive failure."""

    enum_module = getattr(circle, enum_module_name, None)
    enum_type = (
        getattr(enum_module, enum_module_name, None)
        if enum_module is not None
        else None
    )
    if enum_type is None or not hasattr(enum_type, member_name):
        raise RuntimeError(
            f"Circle schema does not provide {enum_module_name}.{member_name}."
        )
    return int(getattr(enum_type, member_name))


_TENSOR_TYPE_TO_DTYPE: dict[int, np.dtype[Any]] = {
    _enum_value("TensorType", "FLOAT32"): np.dtype("<f4"),
    _enum_value("TensorType", "FLOAT16"): np.dtype("<f2"),
    _enum_value("TensorType", "INT32"): np.dtype("<i4"),
    _enum_value("TensorType", "UINT8"): np.dtype("u1"),
    _enum_value("TensorType", "INT64"): np.dtype("<i8"),
    _enum_value("TensorType", "BOOL"): np.dtype("?"),
    _enum_value("TensorType", "INT16"): np.dtype("<i2"),
    _enum_value("TensorType", "INT8"): np.dtype("i1"),
    _enum_value("TensorType", "FLOAT64"): np.dtype("<f8"),
    _enum_value("TensorType", "UINT64"): np.dtype("<u8"),
    _enum_value("TensorType", "UINT32"): np.dtype("<u4"),
    _enum_value("TensorType", "UINT16"): np.dtype("<u2"),
}
_DTYPE_TO_TENSOR_TYPE: dict[np.dtype[Any], int] = {
    dtype.newbyteorder("="): tensor_type
    for tensor_type, dtype in _TENSOR_TYPE_TO_DTYPE.items()
}


def numpy_dtype_from_circle_tensor_type(tensor_type: int) -> np.dtype[Any]:
    """Return the NumPy dtype corresponding to a supported Circle tensor type."""

    try:
        return _TENSOR_TYPE_TO_DTYPE[int(tensor_type)]
    except KeyError as error:
        raise NotImplementedError(
            "CircleReferenceEvaluator does not support tensor type "
            f"{int(tensor_type)}."
        ) from error


def circle_tensor_type_from_numpy_dtype(dtype: np.dtype[Any] | type[Any]) -> int:
    """Return the Circle tensor type corresponding to a supported NumPy dtype."""

    normalized = np.dtype(dtype).newbyteorder("=")
    try:
        return _DTYPE_TO_TENSOR_TYPE[normalized]
    except KeyError as error:
        raise NotImplementedError(
            f"Circle value-test fixtures do not support NumPy dtype {normalized}."
        ) from error


def _buffer_bytes(data: Any) -> bytes:
    """Convert generated buffer payload data into immutable bytes."""

    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.uint8).tobytes()
    try:
        return bytes(data)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Unsupported Circle buffer payload type: {type(data).__name__}."
        ) from error


def _shape_tuple(tensor: Any) -> tuple[int, ...]:
    """Return a tensor shape as a tuple of Python integers."""

    return tuple(
        int(dimension) for dimension in as_list(getattr(tensor, "shape", None))
    )


def _builtin_operator_code(operator_code: Any) -> int:
    """Resolve the builtin code stored in an OperatorCode object."""

    builtin_code = int(getattr(operator_code, "builtinCode", 0) or 0)
    deprecated_code = int(
        getattr(operator_code, "deprecatedBuiltinCode", builtin_code) or 0
    )

    placeholder = getattr(
        getattr(circle.BuiltinOperator, "BuiltinOperator", object()),
        "PLACEHOLDER_FOR_GREATER_OP_CODES",
        127,
    )
    if builtin_code == 0 and deprecated_code != 0:
        return deprecated_code
    if deprecated_code != int(placeholder) and builtin_code == int(placeholder):
        return deprecated_code
    return builtin_code


class CircleReferenceEvaluator:
    """Evaluate a deliberately small subset of Circle operators with NumPy.

    The evaluator is test-only infrastructure. It is intentionally not a general
    Circle runtime and rejects unsupported data types, operators, fused
    activations, external buffers, and optional inputs instead of guessing their
    semantics.
    """

    def __init__(self) -> None:
        """Create an evaluator with handlers for value-test operators."""

        self._handlers: dict[
            int,
            Callable[[Any, tuple[np.ndarray, ...]], tuple[np.ndarray, ...]],
        ] = {
            _enum_value("BuiltinOperator", "ADD"): self._evaluate_add,
            _enum_value("BuiltinOperator", "SUB"): self._evaluate_sub,
            _enum_value("BuiltinOperator", "MUL"): self._evaluate_mul,
            _enum_value("BuiltinOperator", "RESHAPE"): self._evaluate_reshape,
            _enum_value("BuiltinOperator", "TRANSPOSE"): self._evaluate_transpose,
        }

    def evaluate(
        self,
        document: CircleDocument,
        inputs: tuple[np.ndarray, ...],
        *,
        subgraph_index: int = 0,
    ) -> CircleEvaluationResult:
        """Evaluate one subgraph and return outputs plus intermediate values."""

        subgraph = document.subgraph(subgraph_index)
        tensors = as_list(getattr(subgraph, "tensors", None))
        operators = as_list(getattr(subgraph, "operators", None))
        input_indices = as_indices(getattr(subgraph, "inputs", None))
        output_indices = as_indices(getattr(subgraph, "outputs", None))

        if len(inputs) != len(input_indices):
            raise ValueError(
                "Circle input count mismatch: "
                f"expected {len(input_indices)}, received {len(inputs)}."
            )

        tensor_values: dict[int, np.ndarray] = {}
        for tensor_index, input_value in zip(input_indices, inputs):
            tensor_values[tensor_index] = self._validate_tensor_value(
                tensors[tensor_index],
                np.asarray(input_value),
                path=f"subgraphs[{subgraph_index}].inputs[{tensor_index}]",
            )

        for tensor_index, tensor in enumerate(tensors):
            if tensor_index in tensor_values:
                continue
            constant = self._decode_constant(document, tensor, tensor_index)
            if constant is not None:
                tensor_values[tensor_index] = constant

        operator_codes = as_list(getattr(document.model, "operatorCodes", None))
        for operator_index, operator in enumerate(operators):
            opcode_index = int(getattr(operator, "opcodeIndex", -1))
            if opcode_index < 0 or opcode_index >= len(operator_codes):
                raise ValueError(
                    f"Operator {operator_index} references invalid opcode index "
                    f"{opcode_index}."
                )

            builtin_code = _builtin_operator_code(operator_codes[opcode_index])
            try:
                handler = self._handlers[builtin_code]
            except KeyError as error:
                raise NotImplementedError(
                    "CircleReferenceEvaluator does not support builtin operator "
                    f"{builtin_code} at operator {operator_index}."
                ) from error

            operator_inputs = self._resolve_operator_inputs(
                operator,
                tensor_values,
                operator_index=operator_index,
            )
            output_values = handler(operator, operator_inputs)
            output_indices_for_operator = as_indices(getattr(operator, "outputs", None))
            if len(output_values) != len(output_indices_for_operator):
                raise ValueError(
                    f"Operator {operator_index} produced {len(output_values)} values "
                    f"for {len(output_indices_for_operator)} output tensors."
                )

            for tensor_index, output_value in zip(
                output_indices_for_operator, output_values
            ):
                if tensor_index < 0 or tensor_index >= len(tensors):
                    raise ValueError(
                        f"Operator {operator_index} references invalid output tensor "
                        f"{tensor_index}."
                    )
                tensor_values[tensor_index] = self._validate_tensor_value(
                    tensors[tensor_index],
                    np.asarray(output_value),
                    path=(
                        f"subgraphs[{subgraph_index}].operators[{operator_index}]"
                        f".outputs[{tensor_index}]"
                    ),
                )

        missing_outputs = [
            tensor_index
            for tensor_index in output_indices
            if tensor_index not in tensor_values
        ]
        if missing_outputs:
            raise ValueError(
                f"Circle graph outputs were not evaluated: {missing_outputs}."
            )

        outputs = tuple(
            np.array(tensor_values[tensor_index], copy=True)
            for tensor_index in output_indices
        )
        copied_values = {
            tensor_index: np.array(value, copy=True)
            for tensor_index, value in tensor_values.items()
        }
        return CircleEvaluationResult(outputs=outputs, tensor_values=copied_values)

    def _decode_constant(
        self,
        document: CircleDocument,
        tensor: Any,
        tensor_index: int,
    ) -> np.ndarray | None:
        """Decode one inline constant tensor or return None for activations."""

        if bool(getattr(tensor, "isVariable", False)):
            return None

        buffer_index = int(getattr(tensor, "buffer", 0) or 0)
        if buffer_index == 0:
            return None

        buffers = as_list(getattr(document.model, "buffers", None))
        if buffer_index < 0 or buffer_index >= len(buffers):
            raise ValueError(
                f"Tensor {tensor_index} references invalid buffer {buffer_index}."
            )

        buffer = buffers[buffer_index]
        offset = int(getattr(buffer, "offset", 0) or 0)
        size = int(getattr(buffer, "size", 0) or 0)
        if offset or size:
            raise NotImplementedError(
                "CircleReferenceEvaluator does not support external buffers."
            )

        data = getattr(buffer, "data", None)
        if data is None:
            return None
        raw = _buffer_bytes(data)
        if not raw:
            return None

        dtype = numpy_dtype_from_circle_tensor_type(int(getattr(tensor, "type")))
        shape = _shape_tuple(tensor)
        element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        expected_size = element_count * dtype.itemsize
        if len(raw) != expected_size:
            raise ValueError(
                f"Tensor {tensor_index} expects {expected_size} buffer bytes for "
                f"shape {shape} and dtype {dtype}, but buffer {buffer_index} "
                f"contains {len(raw)} bytes."
            )

        value = np.frombuffer(raw, dtype=dtype, count=element_count).copy()
        return value.reshape(shape if shape else ())

    def _resolve_operator_inputs(
        self,
        operator: Any,
        tensor_values: dict[int, np.ndarray],
        *,
        operator_index: int,
    ) -> tuple[np.ndarray, ...]:
        """Resolve operator inputs from the evaluated tensor map."""

        values: list[np.ndarray] = []
        for position, tensor_index in enumerate(
            as_indices(getattr(operator, "inputs", None))
        ):
            if tensor_index == OPTIONAL_TENSOR_INDEX:
                raise NotImplementedError(
                    "CircleReferenceEvaluator does not support optional inputs."
                )
            try:
                values.append(tensor_values[tensor_index])
            except KeyError as error:
                raise ValueError(
                    f"Operator {operator_index} input {position} references tensor "
                    f"{tensor_index}, which has not been evaluated."
                ) from error
        return tuple(values)

    def _validate_tensor_value(
        self,
        tensor: Any,
        value: np.ndarray,
        *,
        path: str,
    ) -> np.ndarray:
        """Validate and copy a value against a Circle tensor contract."""

        expected_shape = _shape_tuple(tensor)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{path} has shape {tuple(value.shape)}, expected {expected_shape}."
            )

        expected_dtype = numpy_dtype_from_circle_tensor_type(
            int(getattr(tensor, "type"))
        ).newbyteorder("=")
        actual_dtype = value.dtype.newbyteorder("=")
        if actual_dtype != expected_dtype:
            raise TypeError(
                f"{path} has dtype {actual_dtype}, expected {expected_dtype}."
            )
        return np.array(value, copy=True, order="C")

    def _require_no_fused_activation(self, operator: Any) -> None:
        """Reject arithmetic operators with a non-NONE fused activation."""

        options = getattr(operator, "builtinOptions", None)
        activation = int(getattr(options, "fusedActivationFunction", 0) or 0)
        none_value = _enum_value("ActivationFunctionType", "NONE")
        if activation != none_value:
            raise NotImplementedError(
                "CircleReferenceEvaluator supports only fused activation NONE."
            )

    def _evaluate_add(
        self,
        operator: Any,
        inputs: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate an ADD operator."""

        self._require_no_fused_activation(operator)
        self._require_input_count("ADD", inputs, 2)
        return (np.add(inputs[0], inputs[1]),)

    def _evaluate_sub(
        self,
        operator: Any,
        inputs: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a SUB operator."""

        self._require_no_fused_activation(operator)
        self._require_input_count("SUB", inputs, 2)
        return (np.subtract(inputs[0], inputs[1]),)

    def _evaluate_mul(
        self,
        operator: Any,
        inputs: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a MUL operator."""

        self._require_no_fused_activation(operator)
        self._require_input_count("MUL", inputs, 2)
        return (np.multiply(inputs[0], inputs[1]),)

    def _evaluate_reshape(
        self,
        operator: Any,
        inputs: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a RESHAPE operator."""

        if len(inputs) == 2:
            target_shape = tuple(int(value) for value in inputs[1].reshape(-1))
        elif len(inputs) == 1:
            options = getattr(operator, "builtinOptions", None)
            target_shape = tuple(
                int(value) for value in as_list(getattr(options, "newShape", None))
            )
        else:
            raise ValueError(
                f"RESHAPE expects one or two inputs, received {len(inputs)}."
            )
        return (np.reshape(inputs[0], target_shape),)

    def _evaluate_transpose(
        self,
        operator: Any,
        inputs: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        """Evaluate a TRANSPOSE operator."""

        del operator
        self._require_input_count("TRANSPOSE", inputs, 2)
        permutation = tuple(int(value) for value in inputs[1].reshape(-1))
        rank = inputs[0].ndim
        if len(permutation) != rank or sorted(permutation) != list(range(rank)):
            raise ValueError(
                f"Invalid TRANSPOSE permutation {permutation} for rank {rank}."
            )
        return (np.transpose(inputs[0], axes=permutation),)

    @staticmethod
    def _require_input_count(
        operator_name: str,
        inputs: tuple[np.ndarray, ...],
        expected: int,
    ) -> None:
        """Require an exact operator input count."""

        if len(inputs) != expected:
            raise ValueError(
                f"{operator_name} expects {expected} inputs, received {len(inputs)}."
            )
