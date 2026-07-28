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

from collections.abc import Sequence
from typing import Any

import numpy as np
from circle_schema import circle

from tico.circle.document import CircleDocument

from test.support.circle.evaluator import (
    circle_tensor_type_from_numpy_dtype,
    numpy_dtype_from_circle_tensor_type,
)


class CircleModelBuilder:
    """Build small, executable Circle Object API fixtures for value tests."""

    def __init__(
        self,
        *,
        description: str = "circle-value-test",
        subgraph_name: str = "main",
    ) -> None:
        """Create an empty single-subgraph Circle model."""

        self.model = circle.Model.ModelT()
        self.model.version = 0
        self.model.description = description
        self.model.operatorCodes = []
        self.model.buffers = [circle.Buffer.BufferT()]
        self.model.metadataBuffer = []
        self.model.metadata = []
        self.model.signatureDefs = []

        self.subgraph = circle.SubGraph.SubGraphT()
        self.subgraph.name = subgraph_name
        self.subgraph.tensors = []
        self.subgraph.inputs = []
        self.subgraph.outputs = []
        self.subgraph.operators = []
        self.model.subgraphs = [self.subgraph]

        self._tensor_names: dict[str, int] = {}
        self._operator_codes: dict[int, int] = {}

    def input(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: np.dtype[Any] | type[Any] = np.float32,
        shape_signature: Sequence[int] | None = None,
    ) -> int:
        """Add a graph input tensor and return its tensor index."""

        tensor_index = self._add_tensor(
            name,
            shape,
            dtype=dtype,
            buffer_index=0,
            shape_signature=shape_signature,
        )
        self.subgraph.inputs.append(tensor_index)
        return tensor_index

    def constant(
        self,
        name: str,
        value: Any,
        *,
        dtype: np.dtype[Any] | type[Any] | None = None,
    ) -> int:
        """Add an inline constant tensor and return its tensor index."""

        array = np.asarray(value, dtype=dtype)
        array = np.ascontiguousarray(array)
        circle_type = circle_tensor_type_from_numpy_dtype(array.dtype)
        storage_dtype = numpy_dtype_from_circle_tensor_type(circle_type)
        array = np.ascontiguousarray(array.astype(storage_dtype, copy=False))

        buffer = circle.Buffer.BufferT()
        buffer.data = array.reshape(-1).view(np.uint8)
        self.model.buffers.append(buffer)
        return self._add_tensor(
            name,
            array.shape,
            dtype=array.dtype,
            buffer_index=len(self.model.buffers) - 1,
        )

    def const_f32(self, name: str, value: Any) -> int:
        """Add a FLOAT32 constant tensor."""

        return self.constant(name, value, dtype=np.float32)

    def const_i32(self, name: str, value: Any) -> int:
        """Add an INT32 constant tensor."""

        return self.constant(name, value, dtype=np.int32)

    def add(self, lhs: int, rhs: int, *, name: str) -> int:
        """Add an ADD operator and return its output tensor index."""

        options = circle.AddOptions.AddOptionsT()
        options.fusedActivationFunction = self._activation_none()
        options.potScaleInt16 = False
        return self._binary_operator(
            self._builtin_operator("ADD"),
            lhs,
            rhs,
            name=name,
            options_type=self._builtin_options("AddOptions"),
            options=options,
        )

    def sub(self, lhs: int, rhs: int, *, name: str) -> int:
        """Add a SUB operator and return its output tensor index."""

        options = circle.SubOptions.SubOptionsT()
        options.fusedActivationFunction = self._activation_none()
        options.potScaleInt16 = False
        return self._binary_operator(
            self._builtin_operator("SUB"),
            lhs,
            rhs,
            name=name,
            options_type=self._builtin_options("SubOptions"),
            options=options,
        )

    def mul(self, lhs: int, rhs: int, *, name: str) -> int:
        """Add a MUL operator and return its output tensor index."""

        options = circle.MulOptions.MulOptionsT()
        options.fusedActivationFunction = self._activation_none()
        return self._binary_operator(
            self._builtin_operator("MUL"),
            lhs,
            rhs,
            name=name,
            options_type=self._builtin_options("MulOptions"),
            options=options,
        )

    def reshape(
        self,
        tensor_index: int,
        new_shape: Sequence[int],
        *,
        name: str,
    ) -> int:
        """Add a RESHAPE operator with an inline INT32 shape tensor."""

        input_tensor = self._tensor(tensor_index)
        input_shape = tuple(int(value) for value in input_tensor.shape)
        resolved_shape = self._resolve_reshape_shape(input_shape, new_shape)
        shape_tensor_index = self.const_i32(f"{name}_shape", list(new_shape))
        output_index = self._add_tensor(
            name,
            resolved_shape,
            dtype=numpy_dtype_from_circle_tensor_type(int(input_tensor.type)),
            buffer_index=0,
        )

        options = circle.ReshapeOptions.ReshapeOptionsT()
        options.newShape = [int(value) for value in new_shape]
        self._append_operator(
            self._builtin_operator("RESHAPE"),
            inputs=[tensor_index, shape_tensor_index],
            outputs=[output_index],
            options_type=self._builtin_options("ReshapeOptions"),
            options=options,
        )
        return output_index

    def transpose(
        self,
        tensor_index: int,
        permutation: Sequence[int],
        *,
        name: str,
    ) -> int:
        """Add a TRANSPOSE operator with an inline INT32 permutation tensor."""

        input_tensor = self._tensor(tensor_index)
        input_shape = tuple(int(value) for value in input_tensor.shape)
        normalized_permutation = tuple(int(value) for value in permutation)
        if len(normalized_permutation) != len(input_shape) or sorted(
            normalized_permutation
        ) != list(range(len(input_shape))):
            raise ValueError(
                f"Invalid permutation {normalized_permutation} for shape {input_shape}."
            )

        permutation_tensor_index = self.const_i32(
            f"{name}_permutation",
            normalized_permutation,
        )
        output_shape = tuple(input_shape[axis] for axis in normalized_permutation)
        output_index = self._add_tensor(
            name,
            output_shape,
            dtype=numpy_dtype_from_circle_tensor_type(int(input_tensor.type)),
            buffer_index=0,
        )

        options = circle.TransposeOptions.TransposeOptionsT()
        self._append_operator(
            self._builtin_operator("TRANSPOSE"),
            inputs=[tensor_index, permutation_tensor_index],
            outputs=[output_index],
            options_type=self._builtin_options("TransposeOptions"),
            options=options,
        )
        return output_index

    def set_outputs(self, *tensor_indices: int) -> None:
        """Set graph outputs in the supplied order."""

        if not tensor_indices:
            raise ValueError("At least one graph output is required.")
        for tensor_index in tensor_indices:
            self._tensor(tensor_index)
        self.subgraph.outputs = [int(index) for index in tensor_indices]

    def tensor_index(self, name: str) -> int:
        """Return the tensor index registered for a name."""

        try:
            return self._tensor_names[name]
        except KeyError as error:
            raise KeyError(f"Unknown Circle tensor name: {name!r}.") from error

    def operator_count(self) -> int:
        """Return the number of operators currently in the subgraph."""

        return len(self.subgraph.operators)

    def build(
        self,
        *,
        signature_key: str | None = "serving_default",
    ) -> CircleDocument:
        """Finalize the fixture and return it as a CircleDocument."""

        if not self.subgraph.outputs:
            raise ValueError("Circle fixture does not define any graph outputs.")
        if signature_key is None:
            self.model.signatureDefs = []
        else:
            self.model.signatureDefs = [self._make_signature(signature_key)]
        return CircleDocument(self.model)

    def _add_tensor(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: np.dtype[Any] | type[Any],
        buffer_index: int,
        shape_signature: Sequence[int] | None = None,
    ) -> int:
        """Create a tensor and return its subgraph-local index."""

        if not name:
            raise ValueError("Circle tensor names must not be empty.")
        if name in self._tensor_names:
            raise ValueError(f"Duplicate Circle tensor name: {name!r}.")

        normalized_shape = [int(dimension) for dimension in shape]
        normalized_signature = (
            list(normalized_shape)
            if shape_signature is None
            else [int(dimension) for dimension in shape_signature]
        )
        if len(normalized_shape) != len(normalized_signature):
            raise ValueError("Shape and shape signature must have the same rank.")

        tensor = circle.Tensor.TensorT()
        tensor.name = name
        tensor.shape = normalized_shape
        tensor.shapeSignature = normalized_signature
        tensor.type = circle_tensor_type_from_numpy_dtype(dtype)
        tensor.buffer = int(buffer_index)
        tensor.isVariable = False

        tensor_index = len(self.subgraph.tensors)
        self.subgraph.tensors.append(tensor)
        self._tensor_names[name] = tensor_index
        return tensor_index

    def _binary_operator(
        self,
        builtin_code: int,
        lhs: int,
        rhs: int,
        *,
        name: str,
        options_type: int,
        options: Any,
    ) -> int:
        """Add a broadcastable binary arithmetic operator."""

        lhs_tensor = self._tensor(lhs)
        rhs_tensor = self._tensor(rhs)
        if int(lhs_tensor.type) != int(rhs_tensor.type):
            raise TypeError(
                f"Binary fixture inputs must have the same type: "
                f"{lhs_tensor.type} != {rhs_tensor.type}."
            )

        output_shape = np.broadcast_shapes(
            tuple(int(value) for value in lhs_tensor.shape),
            tuple(int(value) for value in rhs_tensor.shape),
        )
        output_index = self._add_tensor(
            name,
            output_shape,
            dtype=numpy_dtype_from_circle_tensor_type(int(lhs_tensor.type)),
            buffer_index=0,
        )
        self._append_operator(
            builtin_code,
            inputs=[lhs, rhs],
            outputs=[output_index],
            options_type=options_type,
            options=options,
        )
        return output_index

    def _append_operator(
        self,
        builtin_code: int,
        *,
        inputs: Sequence[int],
        outputs: Sequence[int],
        options_type: int,
        options: Any,
    ) -> None:
        """Append one builtin operator to the graph."""

        operator = circle.Operator.OperatorT()
        operator.opcodeIndex = self._operator_code_index(builtin_code)
        operator.inputs = [int(index) for index in inputs]
        operator.outputs = [int(index) for index in outputs]
        operator.intermediates = []
        operator.mutatingVariableInputs = []
        operator.builtinOptionsType = int(options_type)
        operator.builtinOptions = options
        self.subgraph.operators.append(operator)

    def _operator_code_index(self, builtin_code: int) -> int:
        """Return an existing operator-code index or create one."""

        existing = self._operator_codes.get(int(builtin_code))
        if existing is not None:
            return existing

        operator_code = circle.OperatorCode.OperatorCodeT()
        operator_code.builtinCode = int(builtin_code)
        operator_code.deprecatedBuiltinCode = min(127, int(builtin_code))
        operator_code.version = 1
        operator_code.customCode = None

        index = len(self.model.operatorCodes)
        self.model.operatorCodes.append(operator_code)
        self._operator_codes[int(builtin_code)] = index
        return index

    def _make_signature(self, signature_key: str) -> Any:
        """Create a signature matching the current graph interface."""

        signature = circle.SignatureDef.SignatureDefT()
        signature.signatureKey = signature_key
        signature.subgraphIndex = 0
        signature.inputs = [
            self._make_tensor_map(self._tensor(index).name, index)
            for index in self.subgraph.inputs
        ]
        signature.outputs = [
            self._make_tensor_map(self._tensor(index).name, index)
            for index in self.subgraph.outputs
        ]
        return signature

    @staticmethod
    def _make_tensor_map(name: str, tensor_index: int) -> Any:
        """Create one signature tensor mapping."""

        tensor_map = circle.TensorMap.TensorMapT()
        tensor_map.name = name
        tensor_map.tensorIndex = int(tensor_index)
        return tensor_map

    def _tensor(self, tensor_index: int) -> Any:
        """Return a tensor after validating its index."""

        index = int(tensor_index)
        if index < 0 or index >= len(self.subgraph.tensors):
            raise IndexError(
                f"Tensor index {index} is outside 0..{len(self.subgraph.tensors) - 1}."
            )
        return self.subgraph.tensors[index]

    @staticmethod
    def _resolve_reshape_shape(
        input_shape: Sequence[int],
        requested_shape: Sequence[int],
    ) -> tuple[int, ...]:
        """Resolve one optional inferred dimension in a reshape target."""

        requested = [int(dimension) for dimension in requested_shape]
        inferred_positions = [
            position for position, dimension in enumerate(requested) if dimension == -1
        ]
        if len(inferred_positions) > 1:
            raise ValueError("RESHAPE supports at most one inferred dimension.")
        if any(dimension < -1 for dimension in requested):
            raise ValueError(f"Invalid reshape target: {requested}.")

        input_elements = int(np.prod(tuple(input_shape), dtype=np.int64))
        known_elements = int(
            np.prod(
                [dimension for dimension in requested if dimension != -1],
                dtype=np.int64,
            )
        )
        if inferred_positions:
            if known_elements == 0 or input_elements % known_elements != 0:
                raise ValueError(
                    f"Cannot infer reshape target {requested} from shape {input_shape}."
                )
            requested[inferred_positions[0]] = input_elements // known_elements
        elif known_elements != input_elements:
            raise ValueError(
                f"Reshape target {requested} changes element count from "
                f"{input_elements} to {known_elements}."
            )
        return tuple(requested)

    @staticmethod
    def _builtin_operator(name: str) -> int:
        """Return a BuiltinOperator enum value."""

        return int(getattr(circle.BuiltinOperator.BuiltinOperator, name))

    @staticmethod
    def _builtin_options(name: str) -> int:
        """Return a BuiltinOptions enum value."""

        return int(getattr(circle.BuiltinOptions.BuiltinOptions, name))

    @staticmethod
    def _activation_none() -> int:
        """Return the NONE fused-activation enum value."""

        return int(circle.ActivationFunctionType.ActivationFunctionType.NONE)
