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

"""Data-movement kernels: reshape, transpose, slicing, padding, gather, concat."""

from __future__ import annotations

from math import prod

import numpy as np

from tico.circle.runtime.kernels.base import (
    apply_fused_activation,
    KernelContext,
    KernelRegistry,
    normalize_axis,
)
from tico.circle.runtime.program import tensor_type_spec


def _resolve_reshape(
    ctx: KernelContext, element_count: int, requested: tuple[int, ...]
) -> tuple[int, ...]:
    """Resolve at most one ``-1`` dimension and validate the element count."""

    if any(dim < -1 for dim in requested):
        raise ctx.invalid(
            f"RESHAPE target {list(requested)} contains an invalid dimension."
        )
    inferred = [position for position, dim in enumerate(requested) if dim == -1]
    if len(inferred) > 1:
        raise ctx.invalid(f"RESHAPE target {list(requested)} has more than one -1.")
    resolved = list(requested)
    known = prod(dim for dim in resolved if dim != -1)
    if inferred:
        if known == 0 or element_count % known != 0:
            raise ctx.invalid(
                f"RESHAPE cannot infer dimension {inferred[0]} of {list(requested)} "
                f"from {element_count} elements."
            )
        resolved[inferred[0]] = element_count // known
    elif known != element_count:
        raise ctx.invalid(
            f"RESHAPE target {list(requested)} has {known} elements but the input "
            f"has {element_count}."
        )
    return tuple(resolved)


def _reshape(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1, 2)
    value = ctx.input(0)
    if ctx.has_input(1):
        requested = ctx.index_vector(1, name="shape")
    else:
        new_shape = ctx.option("newShape", None)
        if new_shape is None:
            raise ctx.invalid("RESHAPE has neither a shape input nor newShape options.")
        requested = tuple(int(dim) for dim in new_shape)
    resolved = _resolve_reshape(ctx, value.size, requested)
    return (value.reshape(resolved),)


def _transpose(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    value = ctx.input(0)
    permutation = ctx.index_vector(1, name="perm")
    if len(permutation) != value.ndim or sorted(permutation) != list(range(value.ndim)):
        raise ctx.invalid(
            f"TRANSPOSE permutation {list(permutation)} is invalid for rank {value.ndim}."
        )
    return (np.transpose(value, permutation),)


def _squeeze(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    raw_axes = ctx.option("squeezeDims", None)
    if raw_axes is None or len(raw_axes) == 0:
        axes = tuple(axis for axis, dim in enumerate(value.shape) if dim == 1)
    else:
        axes = tuple(
            normalize_axis(int(axis), value.ndim, ctx, name="squeezeDims entry")
            for axis in raw_axes
        )
        for axis in axes:
            if value.shape[axis] != 1:
                raise ctx.invalid(
                    f"SQUEEZE axis {axis} has size {value.shape[axis]}, not 1."
                )
    return (np.squeeze(value, axis=axes) if axes else value,)


def _expand_dims(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    value = ctx.input(0)
    axis = ctx.index_scalar(1, name="axis")
    rank = value.ndim + 1
    if axis < -rank or axis >= rank:
        raise ctx.invalid(
            f"EXPAND_DIMS axis {axis} is outside the valid range for rank {rank}."
        )
    if axis < 0:
        axis += rank
    return (np.expand_dims(value, axis),)


def _broadcast_to(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    value = ctx.input(0)
    shape = ctx.index_vector(1, name="shape")
    try:
        result = np.broadcast_to(value, shape)
    except ValueError as error:
        raise ctx.invalid(
            f"BROADCAST_TO cannot broadcast shape {value.shape} to {list(shape)}."
        ) from error
    return (np.array(result, copy=True),)


def _concatenation(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    if len(ctx.operator.inputs) < 1:
        raise ctx.invalid("CONCATENATION requires at least one input.")
    values = [ctx.input(position) for position in range(len(ctx.operator.inputs))]
    dtypes = {value.dtype for value in values}
    if len(dtypes) != 1:
        raise ctx.fail(
            f"CONCATENATION inputs must share one dtype, found {sorted(map(str, dtypes))}."
        )
    rank = values[0].ndim
    axis = normalize_axis(int(ctx.option("axis", 0)), rank, ctx, name="axis")
    for position, value in enumerate(values):
        if value.ndim != rank:
            raise ctx.invalid(
                f"CONCATENATION input {position} has rank {value.ndim}, expected {rank}."
            )
        for dim in range(rank):
            if dim != axis and value.shape[dim] != values[0].shape[dim]:
                raise ctx.invalid(
                    f"CONCATENATION input {position} shape {value.shape} does not "
                    f"match input 0 shape {values[0].shape} outside axis {axis}."
                )
    result = np.concatenate(values, axis=axis)
    return (apply_fused_activation(result, ctx),)


def _split_v(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(3)
    value = ctx.input(0)
    sizes = list(ctx.index_vector(1, name="size_splits"))
    axis = normalize_axis(
        ctx.index_scalar(2, name="axis"), value.ndim, ctx, name="axis"
    )
    num_splits = int(ctx.option("numSplits", len(sizes)))
    if num_splits != len(sizes) or len(ctx.operator.outputs) != len(sizes):
        raise ctx.invalid(
            f"SPLIT_V declares numSplits={num_splits} and {len(ctx.operator.outputs)} "
            f"outputs, but size_splits has {len(sizes)} entries."
        )
    inferred = [position for position, size in enumerate(sizes) if size == -1]
    if len(inferred) > 1 or any(size < -1 for size in sizes):
        raise ctx.invalid(f"SPLIT_V size_splits {sizes} is invalid.")
    total = value.shape[axis]
    if inferred:
        known = sum(size for size in sizes if size != -1)
        if known > total:
            raise ctx.invalid(f"SPLIT_V size_splits {sizes} exceed axis size {total}.")
        sizes[inferred[0]] = total - known
    if sum(sizes) != total:
        raise ctx.invalid(
            f"SPLIT_V size_splits {sizes} do not sum to axis size {total}."
        )
    boundaries = np.cumsum(sizes)[:-1]
    return tuple(
        np.array(part, copy=True) for part in np.split(value, boundaries, axis=axis)
    )


def _split(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    axis_value = ctx.index_scalar(0, name="axis")
    value = ctx.input(1)
    axis = normalize_axis(axis_value, value.ndim, ctx, name="axis")
    num_splits = int(ctx.option("numSplits", len(ctx.operator.outputs)))
    if num_splits != len(ctx.operator.outputs) or num_splits <= 0:
        raise ctx.invalid(
            f"SPLIT declares numSplits={num_splits} but has "
            f"{len(ctx.operator.outputs)} outputs."
        )
    if value.shape[axis] % num_splits != 0:
        raise ctx.invalid(
            f"SPLIT cannot divide axis size {value.shape[axis]} into {num_splits} parts."
        )
    return tuple(
        np.array(part, copy=True) for part in np.split(value, num_splits, axis=axis)
    )


def _slice(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(3)
    value = ctx.input(0)
    begin = ctx.index_vector(1, name="begin")
    size = ctx.index_vector(2, name="size")
    if len(begin) != value.ndim or len(size) != value.ndim:
        raise ctx.invalid(
            f"SLICE begin {list(begin)} and size {list(size)} must have rank "
            f"{value.ndim}."
        )
    slices = []
    for axis, (start, extent) in enumerate(zip(begin, size)):
        limit = value.shape[axis]
        if start < 0 or start > limit:
            raise ctx.invalid(
                f"SLICE begin {start} is outside axis {axis} of size {limit}."
            )
        if extent == -1:
            extent = limit - start
        if extent < 0 or start + extent > limit:
            raise ctx.invalid(
                f"SLICE begin {start} and size {extent} exceed axis {axis} of size {limit}."
            )
        slices.append(slice(start, start + extent))
    return (np.array(value[tuple(slices)], copy=True),)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _strided_slice(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(4)
    value = ctx.input(0)
    begin = ctx.index_vector(1, name="begin")
    end = ctx.index_vector(2, name="end")
    strides = ctx.index_vector(3, name="strides")
    rank = value.ndim
    if not (len(begin) == len(end) == len(strides) == rank):
        raise ctx.invalid(
            f"STRIDED_SLICE begin/end/strides ranks ({len(begin)}, {len(end)}, "
            f"{len(strides)}) must equal the input rank {rank}."
        )
    if int(ctx.option("ellipsisMask", 0)) or int(ctx.option("newAxisMask", 0)):
        raise ctx.fail("STRIDED_SLICE ellipsisMask and newAxisMask are not supported.")
    begin_mask = int(ctx.option("beginMask", 0))
    end_mask = int(ctx.option("endMask", 0))
    shrink_mask = int(ctx.option("shrinkAxisMask", 0))
    offset = bool(ctx.option("offset", False))
    index: list[slice | int] = []
    for axis in range(rank):
        axis_size = value.shape[axis]
        stride = strides[axis]
        if stride == 0:
            raise ctx.invalid(f"STRIDED_SLICE stride for axis {axis} must not be zero.")
        start = begin[axis]
        if start < 0:
            start += axis_size
        start = (
            _clamp(start, 0, axis_size)
            if stride > 0
            else _clamp(start, -1, axis_size - 1)
        )
        if begin_mask & (1 << axis):
            start = 0 if stride > 0 else axis_size - 1
        if shrink_mask & (1 << axis):
            if start >= axis_size:
                raise ctx.invalid(
                    f"STRIDED_SLICE shrink index {start} is outside axis {axis} of "
                    f"size {axis_size}."
                )
            index.append(start)
            continue
        stop = end[axis]
        if offset:
            stop += start
        if stop < 0:
            stop += axis_size
        stop = (
            _clamp(stop, 0, axis_size)
            if stride > 0
            else _clamp(stop, -1, axis_size - 1)
        )
        if end_mask & (1 << axis):
            stop = axis_size if stride > 0 else -1
        if stride > 0:
            index.append(slice(start, stop, stride))
        else:
            index.append(slice(start, None if stop < 0 else stop, stride))
    return (np.array(value[tuple(index)], copy=True),)


def _pad_common(ctx: KernelContext, constant: np.ndarray) -> np.ndarray:
    value = ctx.input(0)
    paddings = ctx.input(1)
    if paddings.dtype.kind not in {"i", "u"} or paddings.shape != (value.ndim, 2):
        raise ctx.invalid(
            f"{ctx.name} paddings must be an integer tensor of shape "
            f"[{value.ndim}, 2], found {paddings.dtype}{list(paddings.shape)}."
        )
    pad_width = [(int(before), int(after)) for before, after in paddings.tolist()]
    if any(before < 0 or after < 0 for before, after in pad_width):
        raise ctx.invalid(f"{ctx.name} paddings {pad_width} must not be negative.")
    return np.pad(value, pad_width, mode="constant", constant_values=constant)


def _pad(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    value = ctx.input(0)
    return (_pad_common(ctx, np.zeros((), dtype=value.dtype)),)


def _pad_v2(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(3)
    value = ctx.input(0)
    constant = ctx.input(2)
    if constant.size != 1:
        raise ctx.invalid(
            f"PADV2 constant value must hold one element, found shape {constant.shape}."
        )
    if constant.dtype != value.dtype:
        raise ctx.fail(
            f"PADV2 constant dtype {constant.dtype} must match the input dtype {value.dtype}."
        )
    return (_pad_common(ctx, constant.reshape(())),)


def _check_indices(
    ctx: KernelContext, indices: np.ndarray, limit: int, *, what: str
) -> np.ndarray:
    if indices.dtype.kind not in {"i", "u"}:
        raise ctx.fail(
            f"{ctx.name} {what} must be an integer tensor, found {indices.dtype}."
        )
    if indices.size and (np.any(indices < 0) or np.any(indices >= limit)):
        raise ctx.invalid(
            f"{ctx.name} {what} contain values outside [0, {limit}): "
            f"min={int(indices.min())}, max={int(indices.max())}."
        )
    return indices.astype(np.intp, copy=False)


def _gather(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    params = ctx.input(0)
    indices = ctx.input(1)
    axis = normalize_axis(int(ctx.option("axis", 0)), params.ndim, ctx, name="axis")
    batch_dims = int(ctx.option("batchDims", 0))
    if batch_dims < 0:
        batch_dims += indices.ndim
    if batch_dims < 0 or batch_dims > indices.ndim or batch_dims > axis:
        raise ctx.invalid(
            f"GATHER batchDims {batch_dims} is invalid for axis {axis} and indices "
            f"rank {indices.ndim}."
        )
    if params.shape[:batch_dims] != indices.shape[:batch_dims]:
        raise ctx.invalid(
            f"GATHER batch dimensions of params {params.shape} and indices "
            f"{indices.shape} differ."
        )
    normalized = _check_indices(ctx, indices, params.shape[axis], what="indices")
    if batch_dims == 0:
        return (np.take(params, normalized, axis=axis),)
    batch_shape = params.shape[:batch_dims]
    result_shape = (
        batch_shape
        + params.shape[batch_dims:axis]
        + indices.shape[batch_dims:]
        + params.shape[axis + 1 :]
    )
    result = np.empty(result_shape, dtype=params.dtype)
    for batch_index in np.ndindex(*batch_shape):
        result[batch_index] = np.take(
            params[batch_index], normalized[batch_index], axis=axis - batch_dims
        )
    return (result,)


def _gather_nd(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    params = ctx.input(0)
    indices = ctx.input(1)
    if indices.dtype.kind not in {"i", "u"}:
        raise ctx.fail(
            f"GATHER_ND indices must be an integer tensor, found {indices.dtype}."
        )
    if indices.ndim == 0:
        raise ctx.invalid("GATHER_ND indices must have rank >= 1.")
    depth = indices.shape[-1]
    if depth > params.ndim:
        raise ctx.invalid(
            f"GATHER_ND index depth {depth} exceeds params rank {params.ndim}."
        )
    coordinates = []
    for position in range(depth):
        coordinates.append(
            _check_indices(
                ctx,
                indices[..., position],
                params.shape[position],
                what=f"indices[..., {position}]",
            )
        )
    return (np.array(params[tuple(coordinates)], copy=True),)


def _shape(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    out_type = ctx.option("outType", None)
    dtype = ctx.output_dtype(0)
    if out_type is not None and int(out_type) != ctx.output_tensor(0).tensor_type:
        raise ctx.invalid(
            f"SHAPE outType {int(out_type)} does not match the output tensor type "
            f"{ctx.output_tensor(0).tensor_type}."
        )
    if dtype.kind not in {"i", "u"}:
        raise ctx.fail(f"SHAPE output must be an integer tensor, found {dtype}.")
    return (np.asarray(value.shape, dtype=dtype),)


def _cast(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    input_tensor = ctx.input_tensor(0)
    output_tensor = ctx.output_tensor(0)
    in_type = ctx.option("inDataType", None)
    out_type = ctx.option("outDataType", None)
    if in_type is not None and int(in_type) != input_tensor.tensor_type:
        raise ctx.invalid(
            f"CAST inDataType {int(in_type)} does not match the input tensor type "
            f"{input_tensor.tensor_type}."
        )
    if out_type is not None and int(out_type) != output_tensor.tensor_type:
        raise ctx.invalid(
            f"CAST outDataType {int(out_type)} does not match the output tensor type "
            f"{output_tensor.tensor_type}."
        )
    target = ctx.output_dtype(0)
    if tensor_type_spec(output_tensor.tensor_type).packed:
        raise ctx.fail("CAST to a packed four-bit tensor type is not supported.")
    if target.kind == "b":
        return (np.asarray(value != 0, dtype=target),)
    if target.kind in {"i", "u"} and value.dtype.kind == "f":
        with np.errstate(all="ignore"):
            return (np.trunc(value).astype(target),)
    return (value.astype(target),)


def register_shape_kernels(registry: KernelRegistry) -> None:
    """Register data-movement kernels."""

    registry.register_named("RESHAPE", _reshape)
    registry.register_named("TRANSPOSE", _transpose)
    registry.register_named("SQUEEZE", _squeeze)
    registry.register_named("EXPAND_DIMS", _expand_dims)
    registry.register_named("BROADCAST_TO", _broadcast_to)
    registry.register_named("CONCATENATION", _concatenation)
    registry.register_named("SPLIT_V", _split_v)
    registry.register_named("SPLIT", _split)
    registry.register_named("SLICE", _slice)
    registry.register_named("STRIDED_SLICE", _strided_slice)
    registry.register_named("PAD", _pad)
    registry.register_named("PADV2", _pad_v2)
    registry.register_named("GATHER", _gather)
    registry.register_named("GATHER_ND", _gather_nd)
    registry.register_named("SHAPE", _shape)
    registry.register_named("CAST", _cast)


__all__ = ["register_shape_kernels"]
