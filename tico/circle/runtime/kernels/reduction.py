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

"""Reduction, arg-reduction, cumulative-sum, and softmax kernels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tico.circle.runtime.kernels.base import (
    Kernel,
    KernelContext,
    KernelRegistry,
    normalize_axis,
)


def _reduction_axes(ctx: KernelContext, rank: int) -> tuple[int, ...]:
    """Read and normalize the ``axes`` input, removing duplicates."""

    raw_axes = ctx.index_vector(1, name="axes")
    axes: list[int] = []
    for axis in raw_axes:
        normalized = normalize_axis(axis, rank, ctx, name="reduction axis")
        if normalized not in axes:
            axes.append(normalized)
    return tuple(axes)


def _reducer(
    operation: Callable[..., np.ndarray],
    *,
    kinds: set[str],
) -> Kernel:
    def kernel(ctx: KernelContext) -> tuple[np.ndarray, ...]:
        ctx.require_inputs(2)
        value = ctx.input(0)
        if value.dtype.kind not in kinds:
            raise ctx.fail(f"{ctx.name} is not defined for dtype {value.dtype}.")
        axes = _reduction_axes(ctx, value.ndim)
        keep_dims = bool(ctx.option("keepDims", False))
        if not axes:
            result = np.array(value, copy=True)
        else:
            result = operation(value, axis=axes, keepdims=keep_dims)
        return (np.asarray(result).astype(value.dtype, copy=False),)

    return kernel


def _mean(value: np.ndarray, *, axis: tuple[int, ...], keepdims: bool) -> np.ndarray:
    # TFLite computes the float mean as sum / count in the input precision.
    return np.mean(value, axis=axis, keepdims=keepdims, dtype=value.dtype)


def _sum(value: np.ndarray, *, axis: tuple[int, ...], keepdims: bool) -> np.ndarray:
    return np.sum(value, axis=axis, keepdims=keepdims, dtype=value.dtype)


def _reduce_max(
    value: np.ndarray, *, axis: tuple[int, ...], keepdims: bool
) -> np.ndarray:
    return np.max(value, axis=axis, keepdims=keepdims)


def _reduce_min(
    value: np.ndarray, *, axis: tuple[int, ...], keepdims: bool
) -> np.ndarray:
    return np.min(value, axis=axis, keepdims=keepdims)


def _reduce_prod(
    value: np.ndarray, *, axis: tuple[int, ...], keepdims: bool
) -> np.ndarray:
    return np.prod(value, axis=axis, keepdims=keepdims, dtype=value.dtype)


def _reduce_any(
    value: np.ndarray, *, axis: tuple[int, ...], keepdims: bool
) -> np.ndarray:
    return np.any(value, axis=axis, keepdims=keepdims)


def _arg_reduction(operation: Callable[..., np.ndarray], option_name: str) -> Kernel:
    def kernel(ctx: KernelContext) -> tuple[np.ndarray, ...]:
        ctx.require_inputs(2)
        value = ctx.input(0)
        if value.dtype.kind not in {"f", "i", "u"}:
            raise ctx.fail(f"{ctx.name} is not defined for dtype {value.dtype}.")
        if value.ndim == 0:
            raise ctx.invalid(f"{ctx.name} requires an input of rank >= 1.")
        axis = normalize_axis(
            ctx.index_scalar(1, name="axis"), value.ndim, ctx, name="axis"
        )
        output_type = ctx.option(option_name, None)
        output_tensor = ctx.output_tensor(0)
        if output_type is not None and int(output_type) != output_tensor.tensor_type:
            raise ctx.invalid(
                f"{ctx.name} {option_name} {int(output_type)} does not match the "
                f"output tensor type {output_tensor.tensor_type}."
            )
        dtype = ctx.output_dtype(0)
        if dtype.kind not in {"i", "u"}:
            raise ctx.fail(
                f"{ctx.name} output must be an integer tensor, found {dtype}."
            )
        if value.shape[axis] == 0:
            raise ctx.invalid(f"{ctx.name} cannot reduce an empty axis {axis}.")
        # NumPy returns the first extreme index, matching TFLite's strict comparison.
        return (operation(value, axis=axis).astype(dtype),)

    return kernel


def _cumsum(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    value = ctx.input(0)
    if value.dtype.kind not in {"f", "i", "u"}:
        raise ctx.fail(f"CUMSUM is not defined for dtype {value.dtype}.")
    if value.ndim == 0:
        raise ctx.invalid("CUMSUM requires an input of rank >= 1.")
    axis = normalize_axis(
        ctx.index_scalar(1, name="axis"), value.ndim, ctx, name="axis"
    )
    exclusive = bool(ctx.option("exclusive", False))
    reverse = bool(ctx.option("reverse", False))
    source = np.flip(value, axis=axis) if reverse else value
    result = np.cumsum(source, axis=axis, dtype=value.dtype)
    if exclusive:
        result = result - source
    if reverse:
        result = np.flip(result, axis=axis)
    return (np.ascontiguousarray(result).astype(value.dtype, copy=False),)


def _softmax(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    if value.dtype.kind != "f":
        raise ctx.fail(f"SOFTMAX is not defined for dtype {value.dtype}.")
    if value.ndim == 0:
        raise ctx.invalid("SOFTMAX requires an input of rank >= 1.")
    beta = value.dtype.type(float(ctx.option("beta", 1.0)))
    shifted = (value - np.max(value, axis=-1, keepdims=True)) * beta
    exponent = np.exp(shifted)
    return ((exponent / np.sum(exponent, axis=-1, keepdims=True)).astype(value.dtype),)


def _log_softmax(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    if value.dtype.kind != "f":
        raise ctx.fail(f"LOG_SOFTMAX is not defined for dtype {value.dtype}.")
    shifted = value - np.max(value, axis=-1, keepdims=True)
    log_sum = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return ((shifted - log_sum).astype(value.dtype),)


def register_reduction_kernels(registry: KernelRegistry) -> None:
    """Register reduction, arg-reduction, cumsum, and softmax kernels."""

    registry.register_named("MEAN", _reducer(_mean, kinds={"f"}))
    registry.register_named("SUM", _reducer(_sum, kinds={"f", "i", "u"}))
    registry.register_named(
        "REDUCE_MAX", _reducer(_reduce_max, kinds={"f", "i", "u", "b"})
    )
    registry.register_named(
        "REDUCE_MIN", _reducer(_reduce_min, kinds={"f", "i", "u", "b"})
    )
    registry.register_named(
        "REDUCE_PROD", _reducer(_reduce_prod, kinds={"f", "i", "u"})
    )
    registry.register_named("REDUCE_ANY", _reducer(_reduce_any, kinds={"b"}))
    registry.register_named("ARG_MAX", _arg_reduction(np.argmax, "outputType"))
    registry.register_named("ARG_MIN", _arg_reduction(np.argmin, "outputType"))
    registry.register_named("CUMSUM", _cumsum)
    registry.register_named("SOFTMAX", _softmax)
    registry.register_named("LOG_SOFTMAX", _log_softmax)


__all__ = ["register_reduction_kernels"]
