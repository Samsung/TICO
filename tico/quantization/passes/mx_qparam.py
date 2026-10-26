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

"""Preserve MX block membership when tensor layouts change."""

from __future__ import annotations

import copy
import math
import operator
from typing import cast

import torch

from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.dtypes import is_mx_dtype, normalize_mx_axis
from tico.utils.validate_args_kwargs import (
    CatArgs,
    PermuteArgs,
    SliceArgs,
    SplitWithSizesArgs,
)

_MX_BLOCK_SIZE = 32


def _axis(qparam: QuantParam, rank: int) -> int:
    if qparam.quantized_dimension is None:
        raise ValueError("MX qparams require an explicit quantized dimension.")
    return normalize_mx_axis(qparam.quantized_dimension, rank)


def _shape(node: torch.fx.Node, *, static: bool = False) -> tuple:
    shape = getattr(node.meta.get("val"), "shape", None)
    if shape is None:
        raise NotYetSupportedError(
            f"MX axis propagation requires tensor shape metadata for {node.name}."
        )
    shape = tuple(shape)
    if static and any(type(size) is not int or size <= 0 for size in shape):
        raise NotYetSupportedError(
            "MX block-preserving layout changes require positive static dimensions; "
            f"{node.name} has shape {shape}."
        )
    return shape


def _unsupported(node: torch.fx.Node, reason: str) -> NotYetSupportedError:
    return NotYetSupportedError(
        f"Cannot preserve MX blocks through {node.name} ({node.target}): {reason}. "
        "An explicit dequantize/requantize boundary is required."
    )


def _reshape_axis(
    input_shape: tuple,
    output_shape: tuple,
    axis: int,
    node: torch.fx.Node,
) -> int:
    """Find an output axis with exactly the same flat-index block partition.

    The trailing-element stride fixes the spacing between block members. Axis
    lengths may differ only when both reset at complete 32-element boundaries.
    This permits splitting/merging heads with aligned head dimensions without
    allowing an arbitrary reshape to regroup already quantized values.
    """
    if math.prod(input_shape) != math.prod(output_shape):
        raise _unsupported(node, "reshape changes the number of elements")
    length = input_shape[axis]
    stride = math.prod(input_shape[axis + 1 :])
    for output_axis, output_length in enumerate(output_shape):
        if math.prod(output_shape[output_axis + 1 :]) != stride:
            continue
        if length == output_length or (
            length % _MX_BLOCK_SIZE == 0 and output_length % _MX_BLOCK_SIZE == 0
        ):
            return output_axis
    raise _unsupported(
        node,
        f"reshape {input_shape} -> {output_shape} changes the groups on axis {axis}",
    )


def _check_slice(
    node: torch.fx.Node, length: int, start: int, end: int, step: int
) -> None:
    if (
        step != 1
        or start % _MX_BLOCK_SIZE != 0
        or (end % _MX_BLOCK_SIZE != 0 and end != length)
    ):
        raise _unsupported(
            node,
            f"slice [{start}:{end}:{step}] cuts or regroups an MX block",
        )


def _cat_axis(qparam: QuantParam, node: torch.fx.Node, *, backward: bool) -> int:
    args = CatArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
    output_shape = _shape(node, static=True)
    axis = _axis(qparam, len(output_shape))
    dim = normalize_mx_axis(args.dim, len(output_shape))
    if not args.tensors:
        raise _unsupported(node, "concatenation has no tensor inputs")
    for index, source in enumerate(args.tensors):
        shape = _shape(source, static=True)
        if len(shape) != len(output_shape):
            raise _unsupported(node, "concatenation input ranks differ")
        if not backward:
            source_qparam = source.meta.get(QPARAM_KEY)
            if (
                source_qparam is None
                or source_qparam.dtype != qparam.dtype
                or _axis(source_qparam, len(shape)) != axis
            ):
                raise _unsupported(
                    node, "concatenation inputs use different MX domains"
                )
        if (
            dim == axis
            and index < len(args.tensors) - 1
            and shape[axis] % _MX_BLOCK_SIZE != 0
        ):
            raise _unsupported(node, "concatenation joins incomplete MX blocks")
    return axis


def remap_mx_split_output(
    qparam: QuantParam,
    split: torch.fx.Node,
    index: int,
) -> QuantParam | None:
    """Return one split output's MX domain, including an unused output slot."""
    if not is_mx_dtype(qparam.dtype):
        return None
    if split.target != torch.ops.aten.split_with_sizes.default:
        raise _unsupported(split, "only split_with_sizes outputs are supported")
    args = SplitWithSizesArgs(*split.args, **split.kwargs)  # type: ignore[arg-type]
    shape = _shape(args.input, static=True)
    axis = _axis(qparam, len(shape))
    dim = normalize_mx_axis(args.dim, len(shape))
    if any(type(size) is not int or size <= 0 for size in args.split_sizes):
        raise _unsupported(split, "split requires positive static sizes")
    sizes = cast(list[int], args.split_sizes)
    if sum(sizes) != shape[dim]:
        raise _unsupported(split, "split sizes do not cover its input dimension")
    if type(index) is not int or not -len(sizes) <= index < len(sizes):
        raise _unsupported(split, "split output index is invalid")
    index %= len(sizes)
    if dim == axis:
        start = sum(sizes[:index])
        _check_slice(split, shape[axis], start, start + sizes[index], 1)
    result = copy.deepcopy(qparam)
    result.quantized_dimension = axis
    return result


def remap_mx_qparam(
    qparam: QuantParam,
    node: torch.fx.Node,
    *,
    backward: bool = False,
) -> QuantParam | None:
    """Copy MX metadata across a proven block-preserving layout operation.

    ``qparam`` belongs to the input in the forward direction, or to ``node``
    in the backward direction. Affine qparams return ``None`` so callers retain
    their existing affine policy. Unsupported MX transformations raise rather
    than silently losing quantization or assigning a different block partition.

    Split outputs are represented by their ``getitem`` nodes. Concatenation
    requires matching input domains in the forward direction. The returned
    metadata always uses a nonnegative tensor axis and never aliases the input.
    """
    if not is_mx_dtype(qparam.dtype):
        return None

    if node.target == torch.ops.aten.cat.default:
        axis = _cat_axis(qparam, node, backward=backward)
    elif node.target == operator.getitem:
        split = node.args[0]
        if (
            backward
            or not isinstance(split, torch.fx.Node)
            or split.target != torch.ops.aten.split_with_sizes.default
        ):
            raise _unsupported(node, "only forward split outputs are supported")
        index = node.args[1]
        if not isinstance(index, int):
            raise _unsupported(node, "split output index is invalid")
        return remap_mx_split_output(qparam, split, index)
    else:
        source = node.args[0] if node.args else None
        if not isinstance(source, torch.fx.Node):
            raise _unsupported(node, "tensor input is unavailable")
        input_shape = _shape(source)
        output_shape = _shape(node)
        shape = output_shape if backward else input_shape
        axis = _axis(qparam, len(shape))

        if node.target == torch.ops.aten.permute.default:
            permute_args = PermuteArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            rank = len(input_shape)
            dims = [normalize_mx_axis(dim, rank) for dim in permute_args.dims]
            if len(dims) != rank or len(set(dims)) != rank:
                raise _unsupported(node, "permutation is not a bijection")
            axis = dims[axis] if backward else dims.index(axis)
        elif node.target == torch.ops.aten.reshape.default:
            input_shape = _shape(source, static=True)
            output_shape = _shape(node, static=True)
            if backward:
                input_shape, output_shape = output_shape, input_shape
            axis = _reshape_axis(input_shape, output_shape, axis, node)
        elif node.target == torch.ops.aten.slice.Tensor and not backward:
            slice_args = SliceArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            input_shape = _shape(source, static=True)
            dim = normalize_mx_axis(slice_args.dim, len(input_shape))
            if dim == axis:
                start, end, step = slice(
                    slice_args.start, slice_args.end, slice_args.step
                ).indices(input_shape[axis])
                _check_slice(node, input_shape[axis], start, end, step)
        elif node.target == torch.ops.aten.expand.default and not backward:
            input_shape = _shape(source, static=True)
            output_shape = _shape(node, static=True)
            offset = len(output_shape) - len(input_shape)
            if offset < 0 or any(
                size != 1 and size != output_shape[index + offset]
                for index, size in enumerate(input_shape)
            ):
                raise _unsupported(node, "expand is not a broadcast of its input")
            # Expanding a singleton MX axis repeats the same value in each new
            # block; its exponent and rounded value therefore remain unchanged.
            axis += offset
        elif node.target == torch.ops.aten.constant_pad_nd.default and not backward:
            input_shape = _shape(source, static=True)
            pads = node.args[1]
            value = node.kwargs.get("value", node.args[2] if len(node.args) > 2 else 0)
            if (
                not isinstance(pads, (tuple, list))
                or len(pads) % 2
                or len(pads) > 2 * len(input_shape)
                or any(type(pad) is not int or pad < 0 for pad in pads)
                or value != 0
            ):
                raise _unsupported(node, "only nonnegative zero padding is supported")
            pads = cast(list[int], pads)
            pair = len(input_shape) - 1 - axis
            if pair < len(pads) // 2 and pads[2 * pair] % _MX_BLOCK_SIZE != 0:
                raise _unsupported(node, "padding shifts the start of an MX block")
        else:
            raise _unsupported(node, "operation has no MX block-preservation rule")

    result = copy.deepcopy(qparam)
    result.quantized_dimension = axis
    return result
