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

"""Kernel protocol, execution context, and registry for the reference runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from tico.circle._schema import circle_schema
from tico.circle.runtime.errors import (
    CircleRuntimeValidationError,
    UnsupportedCircleOperatorError,
)
from tico.circle.runtime.program import (
    CircleProgram,
    OPTIONAL_INDEX,
    RuntimeOperator,
    RuntimeTensor,
    tensor_type_name,
)


class ExecutionMode(Enum):
    """Select how tensors carrying affine quantization parameters are executed.

    ``NATIVE`` executes every tensor with its serialized dtype. Integer tensors
    without quantization parameters use ordinary integer arithmetic. Operators
    that would need backend-defined integer arithmetic on quantized tensors are
    rejected instead of being approximated.

    ``FAKE_QUANTIZE`` mirrors the semantics of ONE's
    ``onecc quantize --fake_quantize`` conversion: quantized constants are
    dequantized, arithmetic operators compute in FLOAT32, and every quantized
    activation is passed through a quantize/dequantize round trip with its
    serialized parameters. Quantized graph inputs and outputs are FLOAT32.
    """

    NATIVE = "native"
    FAKE_QUANTIZE = "fake_quantize"


def builtin_code(name: str) -> int:
    """Return one generated ``BuiltinOperator`` value by symbolic name."""

    schema = circle_schema()
    enum_module = getattr(schema, "BuiltinOperator", None)
    enum_type = (
        getattr(enum_module, "BuiltinOperator", None)
        if enum_module is not None
        else None
    )
    if enum_type is None or not hasattr(enum_type, name):
        raise RuntimeError(f"Circle schema does not provide BuiltinOperator.{name}.")
    return int(getattr(enum_type, name))


def optional_builtin_code(name: str) -> int | None:
    """Return a ``BuiltinOperator`` value or None when the schema lacks it."""

    try:
        return builtin_code(name)
    except RuntimeError:
        return None


def tensor_type_value(name: str) -> int:
    """Return one generated ``TensorType`` value by symbolic name."""

    schema = circle_schema()
    enum_module = getattr(schema, "TensorType", None)
    enum_type = (
        getattr(enum_module, "TensorType", None) if enum_module is not None else None
    )
    if enum_type is None or not hasattr(enum_type, name):
        raise RuntimeError(f"Circle schema does not provide TensorType.{name}.")
    return int(getattr(enum_type, name))


@dataclass
class KernelContext:
    """Provide one operator's operands, contracts, and options to a kernel."""

    program: CircleProgram
    operator: RuntimeOperator
    inputs: tuple[np.ndarray | None, ...]
    output_dtypes: tuple[np.dtype, ...]
    mode: ExecutionMode

    @property
    def options(self) -> Any:
        """Return the builtin options table, or None when absent."""

        return self.operator.options

    @property
    def name(self) -> str:
        """Return the operator name for diagnostics."""

        return self.operator.name

    def option(self, field: str, default: Any = None) -> Any:
        """Read one builtin option field with a default for absent tables."""

        options = self.operator.options
        if options is None:
            return default
        value = getattr(options, field, default)
        return default if value is None else value

    def describe(self) -> str:
        """Return the full operator description used in error messages."""

        return self.program.describe_operator(self.operator)

    def fail(self, message: str) -> UnsupportedCircleOperatorError:
        """Create an unsupported-operator error carrying the operator description."""

        return UnsupportedCircleOperatorError(f"{message}\n  at {self.describe()}")

    def invalid(self, message: str) -> CircleRuntimeValidationError:
        """Create a validation error carrying the operator description."""

        return CircleRuntimeValidationError(f"{message}\n  at {self.describe()}")

    def input_tensor(self, position: int) -> RuntimeTensor:
        """Return the serialized tensor metadata of one present input."""

        if position >= len(self.operator.inputs):
            raise self.invalid(
                f"{self.name} requires input {position}, but only "
                f"{len(self.operator.inputs)} inputs are present."
            )
        index = self.operator.inputs[position]
        if index == OPTIONAL_INDEX:
            raise self.invalid(f"{self.name} input {position} must not be absent.")
        return self.program.tensor(index)

    def output_tensor(self, position: int = 0) -> RuntimeTensor:
        """Return the serialized tensor metadata of one output."""

        if position >= len(self.operator.outputs):
            raise self.invalid(
                f"{self.name} requires output {position}, but only "
                f"{len(self.operator.outputs)} outputs are present."
            )
        return self.program.tensor(self.operator.outputs[position])

    def has_input(self, position: int) -> bool:
        """Return whether an optional input is present."""

        return (
            position < len(self.inputs)
            and self.operator.inputs[position] != OPTIONAL_INDEX
        )

    def input(self, position: int) -> np.ndarray:
        """Return one required input value."""

        if position >= len(self.inputs):
            raise self.invalid(
                f"{self.name} requires input {position}, but only "
                f"{len(self.inputs)} inputs are present."
            )
        value = self.inputs[position]
        if value is None:
            raise self.invalid(f"{self.name} input {position} must not be absent.")
        return value

    def optional_input(self, position: int) -> np.ndarray | None:
        """Return one optional input value, or None when absent."""

        if position >= len(self.inputs):
            return None
        return self.inputs[position]

    def require_inputs(self, *counts: int) -> None:
        """Require the serialized input count to be one of ``counts``."""

        if len(self.operator.inputs) not in counts:
            allowed = " or ".join(str(count) for count in counts)
            raise self.invalid(
                f"{self.name} expects {allowed} inputs, found "
                f"{len(self.operator.inputs)}."
            )

    def require_outputs(self, count: int) -> None:
        """Require an exact serialized output count."""

        if len(self.operator.outputs) != count:
            raise self.invalid(
                f"{self.name} expects {count} outputs, found "
                f"{len(self.operator.outputs)}."
            )

    def output_dtype(self, position: int = 0) -> np.dtype:
        """Return the effective output dtype for the current execution mode."""

        return self.output_dtypes[position]

    def require_float(self, *positions: int) -> None:
        """Require the listed inputs to be floating-point arrays."""

        for position in positions:
            value = self.input(position)
            if value.dtype.kind != "f":
                raise self.fail(
                    f"{self.name} input {position} must be a floating-point tensor, "
                    f"found {tensor_type_name(self.input_tensor(position).tensor_type)}."
                )

    def require_same_dtype(self, *positions: int) -> np.dtype:
        """Require the listed inputs to share one dtype and return it."""

        dtypes = {self.input(position).dtype for position in positions}
        if len(dtypes) != 1:
            names = ", ".join(str(self.input(position).dtype) for position in positions)
            raise self.fail(
                f"{self.name} requires inputs {positions} to share one dtype, "
                f"found ({names})."
            )
        return dtypes.pop()

    def index_vector(self, position: int, *, name: str) -> tuple[int, ...]:
        """Read an integer input as a flat tuple of Python integers."""

        value = self.input(position)
        if value.dtype.kind not in {"i", "u"}:
            raise self.fail(
                f"{self.name} {name} (input {position}) must be an integer tensor, "
                f"found {value.dtype}."
            )
        if value.ndim > 1:
            raise self.invalid(
                f"{self.name} {name} (input {position}) must be a scalar or a "
                f"rank-1 tensor, found shape {value.shape}."
            )
        return tuple(int(item) for item in value.reshape(-1).tolist())

    def index_scalar(self, position: int, *, name: str) -> int:
        """Read an integer input holding exactly one element."""

        values = self.index_vector(position, name=name)
        if len(values) != 1:
            raise self.invalid(
                f"{self.name} {name} (input {position}) must hold one element, "
                f"found {len(values)}."
            )
        return values[0]


Kernel = Callable[[KernelContext], tuple[np.ndarray, ...]]


class KernelRegistry:
    """Map Circle builtin operator codes to reference kernels."""

    def __init__(self, entries: Iterable[tuple[int, Kernel]] = ()) -> None:
        """Create a registry and reject duplicate builtin codes."""

        self._kernels: dict[int, Kernel] = {}
        for code, kernel in entries:
            self.register(code, kernel)

    def register(self, code: int, kernel: Kernel) -> None:
        """Register one kernel for a builtin operator code."""

        code = int(code)
        if code in self._kernels:
            raise ValueError(f"A kernel is already registered for builtin code {code}.")
        self._kernels[code] = kernel

    def register_named(self, name: str, kernel: Kernel) -> None:
        """Register a kernel by ``BuiltinOperator`` name when the schema has it."""

        code = optional_builtin_code(name)
        if code is None:
            return
        self.register(code, kernel)

    def get(self, code: int) -> Kernel | None:
        """Return the kernel for a builtin code, if registered."""

        return self._kernels.get(int(code))

    def supports(self, code: int) -> bool:
        """Return whether a builtin code has a kernel."""

        return int(code) in self._kernels

    @property
    def builtin_codes(self) -> tuple[int, ...]:
        """Return the registered builtin codes in ascending order."""

        return tuple(sorted(self._kernels))

    def copy(self) -> KernelRegistry:
        """Return an independently mutable copy."""

        return KernelRegistry(self._kernels.items())


def normalize_axis(axis: int, rank: int, ctx: KernelContext, *, name: str) -> int:
    """Resolve a possibly negative axis against a rank with a descriptive error."""

    if rank == 0 and axis in (0, -1):
        return 0
    if axis < -rank or axis >= rank:
        raise ctx.invalid(
            f"{ctx.name} {name} {axis} is outside the valid range for rank {rank}."
        )
    return axis + rank if axis < 0 else axis


ACTIVATION_NONE = 0
ACTIVATION_RELU = 1
ACTIVATION_RELU_N1_TO_1 = 2
ACTIVATION_RELU6 = 3
ACTIVATION_TANH = 4
ACTIVATION_SIGN_BIT = 5


def apply_fused_activation(value: np.ndarray, ctx: KernelContext) -> np.ndarray:
    """Apply the ``fusedActivationFunction`` option while preserving dtype."""

    activation = int(ctx.option("fusedActivationFunction", ACTIVATION_NONE))
    if activation == ACTIVATION_NONE:
        return value
    if value.dtype.kind not in {"f", "i", "u"}:
        raise ctx.fail(
            f"{ctx.name} fused activation {activation} is not defined for dtype "
            f"{value.dtype}."
        )
    dtype = value.dtype
    if activation == ACTIVATION_RELU:
        return np.maximum(value, dtype.type(0)).astype(dtype, copy=False)
    if activation == ACTIVATION_RELU_N1_TO_1:
        return np.clip(value, dtype.type(-1), dtype.type(1)).astype(dtype, copy=False)
    if activation == ACTIVATION_RELU6:
        return np.clip(value, dtype.type(0), dtype.type(6)).astype(dtype, copy=False)
    if activation == ACTIVATION_TANH:
        if dtype.kind != "f":
            raise ctx.fail(
                f"{ctx.name} fused TANH activation requires a float tensor, found "
                f"{dtype}."
            )
        return np.tanh(value).astype(dtype, copy=False)
    raise ctx.fail(
        f"{ctx.name} fused activation function {activation} is not supported."
    )


def require_no_fused_activation(ctx: KernelContext) -> None:
    """Reject a non-NONE fused activation for operators that never fuse."""

    activation = int(ctx.option("fusedActivationFunction", ACTIVATION_NONE))
    if activation != ACTIVATION_NONE:
        raise ctx.fail(
            f"{ctx.name} does not support fused activation function {activation}."
        )


def broadcast_binary(
    ctx: KernelContext,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    same_dtype: bool = True,
) -> np.ndarray:
    """Evaluate a two-input broadcasting operation with a shape check."""

    ctx.require_inputs(2)
    lhs = ctx.input(0)
    rhs = ctx.input(1)
    if same_dtype and lhs.dtype != rhs.dtype:
        raise ctx.fail(
            f"{ctx.name} requires both inputs to share one dtype, found "
            f"{lhs.dtype} and {rhs.dtype}."
        )
    try:
        np.broadcast_shapes(lhs.shape, rhs.shape)
    except ValueError as error:
        raise ctx.invalid(
            f"{ctx.name} inputs with shapes {lhs.shape} and {rhs.shape} are not "
            "broadcast-compatible."
        ) from error
    return np.asarray(operation(lhs, rhs))


__all__ = [
    "ACTIVATION_NONE",
    "ExecutionMode",
    "Kernel",
    "KernelContext",
    "KernelRegistry",
    "apply_fused_activation",
    "broadcast_binary",
    "builtin_code",
    "normalize_axis",
    "optional_builtin_code",
    "require_no_fused_activation",
    "tensor_type_value",
]
