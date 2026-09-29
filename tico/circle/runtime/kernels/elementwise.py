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

"""Element-wise arithmetic, comparison, logical, and activation kernels."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from tico.circle.runtime.kernels.base import (
    apply_fused_activation,
    broadcast_binary,
    Kernel,
    KernelContext,
    KernelRegistry,
    require_no_fused_activation,
)


def _same_dtype(result: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Return ``result`` in ``dtype`` without changing values."""

    return result.astype(dtype, copy=False)


def _arithmetic(operation: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Kernel:
    """Build a broadcasting arithmetic kernel with fused activation support."""

    def kernel(ctx: KernelContext) -> tuple[np.ndarray, ...]:
        ctx.require_inputs(2)
        dtype = ctx.require_same_dtype(0, 1)
        if dtype.kind not in {"f", "i", "u"}:
            raise ctx.fail(f"{ctx.name} is not defined for dtype {dtype}.")
        result = broadcast_binary(ctx, operation)
        result = _same_dtype(result, dtype)
        return (apply_fused_activation(result, ctx),)

    return kernel


def _divide(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Divide like C: floats divide exactly, integers truncate toward zero."""

    if lhs.dtype.kind == "f":
        return np.divide(lhs, rhs)
    quotient = np.trunc(np.divide(lhs.astype(np.float64), rhs.astype(np.float64)))
    return quotient.astype(lhs.dtype)


def _pow(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    dtype = ctx.require_same_dtype(0, 1)
    if dtype.kind not in {"f", "i"}:
        raise ctx.fail(f"POW is not defined for dtype {dtype}.")
    require_no_fused_activation(ctx)
    result = broadcast_binary(ctx, np.power)
    return (_same_dtype(result, dtype),)


def _minmax(operation: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Kernel:
    def kernel(ctx: KernelContext) -> tuple[np.ndarray, ...]:
        ctx.require_inputs(2)
        dtype = ctx.require_same_dtype(0, 1)
        if dtype.kind not in {"f", "i", "u"}:
            raise ctx.fail(f"{ctx.name} is not defined for dtype {dtype}.")
        result = broadcast_binary(ctx, operation)
        return (_same_dtype(result, dtype),)

    return kernel


def _comparison(operation: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Kernel:
    def kernel(ctx: KernelContext) -> tuple[np.ndarray, ...]:
        ctx.require_inputs(2)
        ctx.require_same_dtype(0, 1)
        result = broadcast_binary(ctx, operation)
        return (result.astype(np.bool_, copy=False),)

    return kernel


def _logical_and(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    dtype = ctx.require_same_dtype(0, 1)
    if dtype != np.dtype(np.bool_):
        raise ctx.fail(f"LOGICAL_AND requires BOOL inputs, found {dtype}.")
    return (broadcast_binary(ctx, np.logical_and),)


def _logical_not(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    if value.dtype != np.dtype(np.bool_):
        raise ctx.fail(f"LOGICAL_NOT requires a BOOL input, found {value.dtype}.")
    return (np.logical_not(value),)


def _unary_float(
    operation: Callable[[np.ndarray], np.ndarray],
    *,
    allow_integer: bool = False,
) -> Kernel:
    """Build a unary kernel that preserves the input dtype."""

    def kernel(ctx: KernelContext) -> tuple[np.ndarray, ...]:
        ctx.require_inputs(1)
        value = ctx.input(0)
        kinds = {"f", "i", "u"} if allow_integer else {"f"}
        if value.dtype.kind not in kinds:
            raise ctx.fail(f"{ctx.name} is not defined for dtype {value.dtype}.")
        result = np.asarray(operation(value))
        return (_same_dtype(result, value.dtype),)

    return kernel


def _logistic(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _relu(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, value.dtype.type(0))


def _relu6(value: np.ndarray) -> np.ndarray:
    return np.clip(value, value.dtype.type(0), value.dtype.type(6))


def _relu_n1_to_1(value: np.ndarray) -> np.ndarray:
    return np.clip(value, value.dtype.type(-1), value.dtype.type(1))


def _elu(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0, value, np.exp(value) - 1)


def _leaky_relu(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    if value.dtype.kind != "f":
        raise ctx.fail(f"LEAKY_RELU is not defined for dtype {value.dtype}.")
    alpha = value.dtype.type(float(ctx.option("alpha", 0.0)))
    return (np.where(value >= 0, value, value * alpha).astype(value.dtype),)


def _gelu(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    if value.dtype.kind != "f":
        raise ctx.fail(f"GELU is not defined for dtype {value.dtype}.")
    x = value.astype(np.float64)
    if bool(ctx.option("approximate", False)):
        # TFLite reference: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))
        result = 0.5 * x * (1.0 + np.tanh(inner))
    else:
        # TFLite reference: 0.5 * x * (1 + erf(x / sqrt(2)))
        erf = np.vectorize(math.erf, otypes=[np.float64])
        result = 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))
    return (result.astype(value.dtype),)


def _prelu(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    dtype = ctx.require_same_dtype(0, 1)
    if dtype.kind != "f":
        raise ctx.fail(f"PRELU is not defined for dtype {dtype}.")
    value = ctx.input(0)
    alpha = ctx.input(1)
    try:
        broadcast_shape = np.broadcast_shapes(value.shape, alpha.shape)
    except ValueError as error:
        raise ctx.invalid(
            f"PRELU alpha shape {alpha.shape} does not broadcast against input "
            f"shape {value.shape}."
        ) from error
    if tuple(broadcast_shape) != tuple(value.shape):
        raise ctx.invalid(
            f"PRELU alpha shape {alpha.shape} must broadcast to the input shape "
            f"{value.shape}, not {tuple(broadcast_shape)}."
        )
    result = np.where(value >= 0, value, value * alpha)
    return (_same_dtype(result, dtype),)


def _select_v2(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(3)
    condition = ctx.input(0)
    if condition.dtype != np.dtype(np.bool_):
        raise ctx.fail(
            f"{ctx.name} condition must be a BOOL tensor, found {condition.dtype}."
        )
    dtype = ctx.require_same_dtype(1, 2)
    on_true = ctx.input(1)
    on_false = ctx.input(2)
    try:
        np.broadcast_shapes(condition.shape, on_true.shape, on_false.shape)
    except ValueError as error:
        raise ctx.invalid(
            f"{ctx.name} inputs with shapes {condition.shape}, {on_true.shape}, "
            f"and {on_false.shape} are not broadcast-compatible."
        ) from error
    return (np.where(condition, on_true, on_false).astype(dtype, copy=False),)


def register_elementwise_kernels(registry: KernelRegistry) -> None:
    """Register arithmetic, comparison, logical, and activation kernels."""

    registry.register_named("ADD", _arithmetic(np.add))
    registry.register_named("SUB", _arithmetic(np.subtract))
    registry.register_named("MUL", _arithmetic(np.multiply))
    registry.register_named("DIV", _arithmetic(_divide))
    registry.register_named("POW", _pow)
    registry.register_named("MAXIMUM", _minmax(np.maximum))
    registry.register_named("MINIMUM", _minmax(np.minimum))
    registry.register_named("EQUAL", _comparison(np.equal))
    registry.register_named("NOT_EQUAL", _comparison(np.not_equal))
    registry.register_named("GREATER", _comparison(np.greater))
    registry.register_named("GREATER_EQUAL", _comparison(np.greater_equal))
    registry.register_named("LESS", _comparison(np.less))
    registry.register_named("LESS_EQUAL", _comparison(np.less_equal))
    registry.register_named("LOGICAL_AND", _logical_and)
    registry.register_named("LOGICAL_NOT", _logical_not)
    registry.register_named("ABS", _unary_float(np.abs, allow_integer=True))
    registry.register_named("NEG", _unary_float(np.negative, allow_integer=True))
    registry.register_named("EXP", _unary_float(np.exp))
    registry.register_named("LOG", _unary_float(np.log))
    registry.register_named("SIN", _unary_float(np.sin))
    registry.register_named("COS", _unary_float(np.cos))
    registry.register_named("SQRT", _unary_float(np.sqrt))
    registry.register_named("RSQRT", _unary_float(lambda value: 1.0 / np.sqrt(value)))
    registry.register_named("TANH", _unary_float(np.tanh))
    registry.register_named("LOGISTIC", _unary_float(_logistic))
    # TFLite ROUND rounds half to even, which is NumPy's default.
    registry.register_named("ROUND", _unary_float(np.round))
    registry.register_named("RELU", _unary_float(_relu, allow_integer=True))
    registry.register_named("RELU6", _unary_float(_relu6, allow_integer=True))
    registry.register_named(
        "RELU_N1_TO_1", _unary_float(_relu_n1_to_1, allow_integer=True)
    )
    registry.register_named("ELU", _unary_float(_elu))
    registry.register_named("LEAKY_RELU", _leaky_relu)
    registry.register_named("GELU", _gelu)
    registry.register_named("PRELU", _prelu)
    registry.register_named("SELECT_V2", _select_v2)
    registry.register_named("SELECT", _select_v2)


__all__ = ["register_elementwise_kernels"]
