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

"""Graph executor of the Circle reference runtime.

The executor walks the serialized operator list in order, dispatches each
operator to a kernel, and validates every produced value against the tensor
contract stored in the ``.circle`` file: the dtype must match exactly and every
static dimension must match, while dimensions marked ``-1`` in the shape
signature may take any size. Results are never reshaped or cast to satisfy the
metadata; a mismatch is reported as an error that names the operator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tico.circle.document import CircleDocument
from tico.circle.runtime.errors import (
    CircleRuntimeError,
    CircleRuntimeValidationError,
    UnsupportedCircleOperatorError,
)
from tico.circle.runtime.kernels import (
    conversion_builtin_codes,
    default_kernel_registry,
    ExecutionMode,
    index_builtin_codes,
    KernelContext,
    KernelRegistry,
    value_preserving_builtin_codes,
)
from tico.circle.runtime.program import (
    CircleProgram,
    OPTIONAL_INDEX,
    RuntimeOperator,
    RuntimeTensor,
    tensor_type_name,
)
from tico.circle.runtime.quantization import dequantize_array, fake_quantize_array


@dataclass(frozen=True)
class ExecutionResult:
    """Return graph outputs and, optionally, every traced tensor value."""

    outputs: tuple[np.ndarray, ...]
    tensor_values: dict[int, np.ndarray] | None = None


class CircleReferenceRuntime:
    """Execute one serialized Circle subgraph with NumPy/PyTorch reference kernels.

    The runtime parses the model once. Repeated :meth:`run` calls reuse decoded
    constants and never mutate the caller's input arrays. Only the bytes of the
    ``.circle`` model are consulted; no exporter metadata is required.
    """

    def __init__(
        self,
        circle: bytes | CircleDocument | CircleProgram,
        *,
        subgraph_index: int = 0,
        registry: KernelRegistry | None = None,
        verify: bool = True,
    ) -> None:
        """Parse and prepare a Circle model for execution."""

        if isinstance(circle, CircleProgram):
            program = circle
        else:
            document = (
                circle
                if isinstance(circle, CircleDocument)
                else CircleDocument.from_bytes(circle)
            )
            if verify:
                document.verify(raise_on_error=True)
            program = CircleProgram(document, subgraph_index=subgraph_index)
        self.program = program
        self.registry = registry or default_kernel_registry()
        self._dequantized_constants: dict[int, np.ndarray] = {}
        self._last_use = self._compute_last_use()
        self.prepare()

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        """Check that every operator can be executed before running any data.

        The check rejects unsupported operators, control-flow references, and
        integer quantized arithmetic that has no reference definition, and it
        reports each problem with the operator index and operand contracts.
        """

        if self.program.subgraph_count != 1:
            raise UnsupportedCircleOperatorError(
                f"The reference runtime executes single-subgraph models; the model "
                f"has {self.program.subgraph_count} subgraphs."
            )
        for operator in self.program.operators:
            if operator.custom_code:
                raise UnsupportedCircleOperatorError(
                    "Custom operators are not supported by the reference runtime.\n"
                    f"  at {self.program.describe_operator(operator)}"
                )
            if self.registry.get(operator.builtin_code) is None:
                raise UnsupportedCircleOperatorError(
                    f"No reference kernel is registered for builtin operator "
                    f"{operator.name}.\n  at {self.program.describe_operator(operator)}"
                )
            if _references_subgraph(operator.options):
                raise UnsupportedCircleOperatorError(
                    "Control-flow operators referencing other subgraphs are not "
                    f"supported.\n  at {self.program.describe_operator(operator)}"
                )
            for tensor_index in operator.inputs + operator.outputs:
                if tensor_index == OPTIONAL_INDEX:
                    continue
                if self.program.tensor(tensor_index).is_variable:
                    raise UnsupportedCircleOperatorError(
                        "Variable (stateful) tensors are not supported.\n"
                        f"  at {self.program.describe_operator(operator)}"
                    )

    def _compute_last_use(self) -> dict[int, int]:
        last_use: dict[int, int] = {}
        for operator in self.program.operators:
            for tensor_index in operator.inputs:
                if tensor_index != OPTIONAL_INDEX:
                    last_use[tensor_index] = operator.index
        return last_use

    # -------------------------------------------------------------- contracts
    @property
    def input_tensors(self) -> tuple[RuntimeTensor, ...]:
        """Return graph input tensors in interface order."""

        return tuple(self.program.tensor(index) for index in self.program.inputs)

    @property
    def output_tensors(self) -> tuple[RuntimeTensor, ...]:
        """Return graph output tensors in interface order."""

        return tuple(self.program.tensor(index) for index in self.program.outputs)

    def effective_dtype(self, tensor: RuntimeTensor, mode: ExecutionMode) -> np.dtype:
        """Return the dtype a value of ``tensor`` has during execution in ``mode``."""

        if mode is ExecutionMode.FAKE_QUANTIZE and tensor.is_quantized_integer:
            return np.dtype(np.float32)
        return tensor.dtype

    def has_quantized_activations(self) -> bool:
        """Return whether any non-constant tensor carries integer quantization."""

        return any(
            tensor.is_quantized_integer and not tensor.is_constant
            for tensor in self.program.tensors
        )

    def probe_static_contracts(self) -> ExecutionResult:
        """Execute a fully static model on zero-valued inputs to check its contracts.

        Every operator result is validated against the serialized shape and
        dtype exactly as in a normal run, so the probe detects operators whose
        declared output metadata disagrees with their computed result without
        needing golden data. Models with dynamic input dimensions are rejected;
        their contracts are validated during a real :meth:`run`.
        """

        dynamic = [tensor for tensor in self.input_tensors if tensor.is_dynamic]
        if dynamic:
            raise CircleRuntimeValidationError(
                "Static contract probing requires static graph inputs; dynamic "
                f"inputs: {', '.join(tensor.describe() for tensor in dynamic)}."
            )
        mode = (
            ExecutionMode.FAKE_QUANTIZE
            if self.has_quantized_activations()
            else ExecutionMode.NATIVE
        )
        probes = [
            np.zeros(tensor.shape, dtype=self.effective_dtype(tensor, mode))
            for tensor in self.input_tensors
        ]
        return self.run(probes, mode=mode)

    # ---------------------------------------------------------------- running
    def run(
        self,
        inputs: Sequence[Any],
        *,
        mode: ExecutionMode = ExecutionMode.NATIVE,
        trace: bool = False,
    ) -> ExecutionResult:
        """Execute the subgraph on ``inputs`` and return its outputs.

        ``inputs`` are bound positionally to the graph inputs and must already
        have the serialized dtype (FLOAT32 for quantized inputs in fake-quantize
        mode). ``trace=True`` additionally keeps every intermediate tensor value
        in ``tensor_values``; by default values are released once their last
        consumer has executed.
        """

        program = self.program
        if len(inputs) != len(program.inputs):
            raise CircleRuntimeValidationError(
                f"Circle model expects {len(program.inputs)} inputs, received "
                f"{len(inputs)}."
            )
        values: dict[int, np.ndarray] = {}
        for position, (tensor_index, raw) in enumerate(zip(program.inputs, inputs)):
            tensor = program.tensor(tensor_index)
            array = self._as_readonly_array(raw, tensor, position)
            self._validate_value(
                array,
                tensor,
                mode,
                path=f"graph input {position}",
            )
            if mode is ExecutionMode.FAKE_QUANTIZE and tensor.is_quantized_integer:
                array = fake_quantize_array(array, tensor)
            values[tensor_index] = array

        for tensor in program.tensors:
            if tensor.constant is None:
                continue
            values[tensor.index] = self._constant_value(tensor, mode)

        output_set = set(program.outputs)
        for operator in program.operators:
            self._execute_operator(operator, values, mode)
            if trace:
                continue
            for tensor_index in operator.inputs:
                if tensor_index == OPTIONAL_INDEX or tensor_index in output_set:
                    continue
                if self._last_use.get(tensor_index) == operator.index:
                    tensor = program.tensor(tensor_index)
                    if not tensor.is_constant and tensor_index not in program.inputs:
                        values.pop(tensor_index, None)

        outputs = []
        for position, tensor_index in enumerate(program.outputs):
            if tensor_index not in values:
                raise CircleRuntimeValidationError(
                    f"Graph output {position} ({program.tensor(tensor_index).describe()}) "
                    "was never produced, is not a graph input, and is not a constant."
                )
            outputs.append(np.array(values[tensor_index], copy=True))
        traced = (
            {index: np.array(value, copy=True) for index, value in values.items()}
            if trace
            else None
        )
        return ExecutionResult(outputs=tuple(outputs), tensor_values=traced)

    def _constant_value(self, tensor: RuntimeTensor, mode: ExecutionMode) -> np.ndarray:
        assert tensor.constant is not None
        if mode is ExecutionMode.FAKE_QUANTIZE and tensor.is_quantized_integer:
            cached = self._dequantized_constants.get(tensor.index)
            if cached is None:
                cached = dequantize_array(tensor.constant, tensor)
                cached.setflags(write=False)
                self._dequantized_constants[tensor.index] = cached
            return cached
        return tensor.constant

    def _execute_operator(
        self,
        operator: RuntimeOperator,
        values: dict[int, np.ndarray],
        mode: ExecutionMode,
    ) -> None:
        program = self.program
        kernel = self.registry.get(operator.builtin_code)
        if kernel is None:
            raise UnsupportedCircleOperatorError(
                f"No reference kernel is registered for builtin operator "
                f"{operator.name}.\n  at {program.describe_operator(operator)}"
            )
        operands: list[np.ndarray | None] = []
        for position, tensor_index in enumerate(operator.inputs):
            if tensor_index == OPTIONAL_INDEX:
                operands.append(None)
                continue
            value = values.get(tensor_index)
            if value is None:
                raise CircleRuntimeValidationError(
                    f"Operator {operator.index} ({operator.name}) input {position} "
                    f"references {program.tensor(tensor_index).describe()}, which has "
                    "no value: it is not a graph input, not a constant, and not "
                    "produced by an earlier operator."
                )
            operands.append(value)
        self._check_quantized_operands(operator, mode)

        output_tensors = tuple(
            program.tensor(index)
            for index in operator.outputs
            if index != OPTIONAL_INDEX
        )
        context = KernelContext(
            program=program,
            operator=operator,
            inputs=tuple(operands),
            output_dtypes=tuple(
                self.effective_dtype(tensor, mode) for tensor in output_tensors
            ),
            mode=mode,
        )
        try:
            with np.errstate(all="ignore"):
                results = kernel(context)
        except CircleRuntimeError:
            raise
        except (
            ValueError,
            TypeError,
            IndexError,
            ZeroDivisionError,
            RuntimeError,
        ) as error:
            raise CircleRuntimeValidationError(
                f"{operator.name} kernel failed: {error}\n"
                f"  at {program.describe_operator(operator)}"
            ) from error

        if len(results) != len(output_tensors):
            raise CircleRuntimeValidationError(
                f"{operator.name} kernel produced {len(results)} values for "
                f"{len(output_tensors)} outputs.\n  at {program.describe_operator(operator)}"
            )
        requantize = (
            mode is ExecutionMode.FAKE_QUANTIZE
            and operator.builtin_code not in value_preserving_builtin_codes()
            and operator.builtin_code not in conversion_builtin_codes()
        )
        for position, (tensor, result) in enumerate(zip(output_tensors, results)):
            array = np.asarray(result)
            self._validate_value(
                array,
                tensor,
                mode,
                path=f"operator {operator.index} ({operator.name}) output {position}",
                operator=operator,
            )
            if requantize and tensor.is_quantized_integer:
                array = fake_quantize_array(array, tensor)
            # np.ascontiguousarray would promote 0-d arrays to rank 1; keep the rank.
            if not array.flags.c_contiguous or not array.flags.owndata:
                array = np.array(array, order="C", copy=True)
            array.setflags(write=False)
            values[tensor.index] = array

    def _check_quantized_operands(
        self, operator: RuntimeOperator, mode: ExecutionMode
    ) -> None:
        if mode is not ExecutionMode.NATIVE:
            return
        code = operator.builtin_code
        if (
            code in value_preserving_builtin_codes()
            or code in conversion_builtin_codes()
        ):
            return
        if code in index_builtin_codes():
            return
        quantized = [
            self.program.tensor(index)
            for index in operator.inputs + operator.outputs
            if index != OPTIONAL_INDEX
            and self.program.tensor(index).is_quantized_integer
        ]
        if quantized:
            raise UnsupportedCircleOperatorError(
                f"{operator.name} on integer quantized tensors has backend-defined "
                "integer arithmetic and is not executed in NATIVE mode; use "
                "ExecutionMode.FAKE_QUANTIZE for the dequantized reference semantics.\n"
                f"  quantized operands: {', '.join(t.describe() for t in quantized)}\n"
                f"  at {self.program.describe_operator(operator)}"
            )

    def _as_readonly_array(
        self, raw: Any, tensor: RuntimeTensor, position: int
    ) -> np.ndarray:
        if hasattr(raw, "detach") and hasattr(raw, "numpy"):
            # torch.Tensor without importing torch at module scope.
            raw = raw.detach().cpu().numpy()
        if not isinstance(raw, np.ndarray):
            raise CircleRuntimeValidationError(
                f"Graph input {position} ({tensor.describe()}) must be a NumPy array "
                f"or torch.Tensor, received {type(raw).__name__}."
            )
        view = raw.view()
        view.setflags(write=False)
        return view

    def _validate_value(
        self,
        value: np.ndarray,
        tensor: RuntimeTensor,
        mode: ExecutionMode,
        *,
        path: str,
        operator: RuntimeOperator | None = None,
    ) -> None:
        expected_dtype = self.effective_dtype(tensor, mode)
        location = (
            ""
            if operator is None
            else f"\n  at {self.program.describe_operator(operator)}"
        )
        if value.dtype.newbyteorder("=") != expected_dtype.newbyteorder("="):
            expected_name = tensor_type_name(tensor.tensor_type)
            if expected_dtype != tensor.dtype:
                expected_name = f"FLOAT32 (dequantized {expected_name})"
            raise CircleRuntimeValidationError(
                f"{path} has dtype {value.dtype}, but {tensor.describe()} requires "
                f"{expected_name}.{location}"
            )
        expected_shape = tensor.shape
        signature = tensor.shape_signature
        if value.ndim != len(expected_shape):
            raise CircleRuntimeValidationError(
                f"{path} has shape {list(value.shape)}, but {tensor.describe()} requires "
                f"rank {len(expected_shape)}.{location}"
            )
        for axis, actual in enumerate(value.shape):
            dynamic = signature is not None and signature[axis] == -1
            if dynamic:
                continue
            if actual != expected_shape[axis]:
                raise CircleRuntimeValidationError(
                    f"{path} has shape {list(value.shape)}, but {tensor.describe()} "
                    f"requires dimension {axis} to be {expected_shape[axis]}.{location}"
                )


def _references_subgraph(options: Any) -> bool:
    """Return whether builtin options reference another subgraph."""

    if options is None:
        return False
    for field_name in dir(options):
        if field_name.startswith("_"):
            continue
        normalized = field_name.lower()
        if not (
            normalized.endswith("subgraphindex")
            or normalized.endswith("subgraphindices")
            or normalized == "subgraph"
        ):
            continue
        value = getattr(options, field_name, None)
        if callable(value):
            continue
        if value is None:
            continue
        try:
            return len(value) > 0
        except TypeError:
            return True
    return False


def run_circle(
    circle: bytes | CircleDocument,
    inputs: Sequence[Any],
    *,
    mode: ExecutionMode = ExecutionMode.NATIVE,
    trace: bool = False,
) -> ExecutionResult:
    """Execute serialized Circle bytes or a document once."""

    return CircleReferenceRuntime(circle).run(inputs, mode=mode, trace=trace)


__all__ = ["CircleReferenceRuntime", "ExecutionResult", "run_circle"]
