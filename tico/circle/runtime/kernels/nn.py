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

"""Convolution, pooling, matrix, resize, and normalization kernels.

All spatial kernels follow the Circle/TFLite NHWC layout conventions:

- ``CONV_2D`` filter is ``[out_channels, kh, kw, in_channels]``.
- ``DEPTHWISE_CONV_2D`` filter is ``[1, kh, kw, in_channels * depth_multiplier]``
  with output channel ``ic * depth_multiplier + m``.
- ``TRANSPOSE_CONV`` filter is ``[out_channels, kh, kw, in_channels]`` and the
  first input holds the explicit output shape.
- ``SAME`` padding distributes the total padding with the extra element at the
  end (bottom/right), exactly like TensorFlow.

CPU PyTorch is used only as a numerical primitive for the convolution kernels;
padding, layout, and shape semantics are computed here from the Circle options.
"""

from __future__ import annotations

import numpy as np

from tico.circle.runtime.kernels.base import (
    apply_fused_activation,
    KernelContext,
    KernelRegistry,
)

PADDING_SAME = 0
PADDING_VALID = 1


def _same_padding(
    input_size: int, filter_size: int, stride: int, dilation: int
) -> tuple[int, int, int]:
    """Return (output_size, pad_before, pad_after) for TF SAME padding."""

    effective = (filter_size - 1) * dilation + 1
    output_size = -(-input_size // stride)  # ceil division
    total = max((output_size - 1) * stride + effective - input_size, 0)
    before = total // 2
    return output_size, before, total - before


def _valid_output_size(
    input_size: int, filter_size: int, stride: int, dilation: int
) -> int:
    effective = (filter_size - 1) * dilation + 1
    return max((input_size - effective) // stride + 1, 0)


def _spatial_geometry(
    ctx: KernelContext,
    input_hw: tuple[int, int],
    filter_hw: tuple[int, int],
    stride_hw: tuple[int, int],
    dilation_hw: tuple[int, int],
) -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    """Return output (H, W) and ((top, bottom), (left, right)) padding."""

    padding = int(ctx.option("padding", -1))
    if min(stride_hw) <= 0 or min(dilation_hw) <= 0:
        raise ctx.invalid(
            f"{ctx.name} strides {stride_hw} and dilations {dilation_hw} must be positive."
        )
    if padding == PADDING_SAME:
        out_h, top, bottom = _same_padding(
            input_hw[0], filter_hw[0], stride_hw[0], dilation_hw[0]
        )
        out_w, left, right = _same_padding(
            input_hw[1], filter_hw[1], stride_hw[1], dilation_hw[1]
        )
        return (out_h, out_w), ((top, bottom), (left, right))
    if padding == PADDING_VALID:
        out_h = _valid_output_size(
            input_hw[0], filter_hw[0], stride_hw[0], dilation_hw[0]
        )
        out_w = _valid_output_size(
            input_hw[1], filter_hw[1], stride_hw[1], dilation_hw[1]
        )
        return (out_h, out_w), ((0, 0), (0, 0))
    raise ctx.fail(f"{ctx.name} padding enum value {padding} is not supported.")


def _require_rank(
    ctx: KernelContext, value: np.ndarray, rank: int, *, what: str
) -> None:
    if value.ndim != rank:
        raise ctx.invalid(
            f"{ctx.name} {what} must have rank {rank}, found shape {value.shape}."
        )


def _to_torch(value: np.ndarray):  # type: ignore[no-untyped-def]
    import torch

    # Copy so that PyTorch never aliases a read-only runtime value.
    return torch.from_numpy(np.array(value, dtype=value.dtype, copy=True, order="C"))


def _conv2d_core(
    ctx: KernelContext,
    value: np.ndarray,
    filters_oihw: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride_hw: tuple[int, int],
    dilation_hw: tuple[int, int],
    padding: tuple[tuple[int, int], tuple[int, int]],
    groups: int,
) -> np.ndarray:
    """Run an NHWC convolution through PyTorch with explicit asymmetric padding."""

    import torch
    import torch.nn.functional as F

    (top, bottom), (left, right) = padding
    with torch.no_grad():
        nchw = _to_torch(value).permute(0, 3, 1, 2)
        if any((top, bottom, left, right)):
            nchw = F.pad(nchw, (left, right, top, bottom))
        result = F.conv2d(
            nchw,
            _to_torch(filters_oihw),
            None if bias is None else _to_torch(bias),
            stride=stride_hw,
            dilation=dilation_hw,
            groups=groups,
        )
        return result.permute(0, 2, 3, 1).contiguous().numpy()


def _conv_2d(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2, 3)
    value = ctx.input(0)
    filters = ctx.input(1)
    bias = ctx.optional_input(2)
    if (
        value.dtype.kind != "f"
        or filters.dtype != value.dtype
        or (bias is not None and bias.dtype != value.dtype)
    ):
        raise ctx.fail(
            f"CONV_2D requires FLOAT32 input, filter, and bias in native mode; found "
            f"{value.dtype}, {filters.dtype}, {None if bias is None else bias.dtype}."
        )
    _require_rank(ctx, value, 4, what="input")
    _require_rank(ctx, filters, 4, what="filter")
    out_channels, kh, kw, in_channels = filters.shape
    if in_channels != value.shape[3]:
        raise ctx.invalid(
            f"CONV_2D filter input channels {in_channels} do not match the input "
            f"channels {value.shape[3]}."
        )
    if bias is not None and bias.shape != (out_channels,):
        raise ctx.invalid(f"CONV_2D bias shape {bias.shape} must be [{out_channels}].")
    stride_hw = (int(ctx.option("strideH", 0)), int(ctx.option("strideW", 0)))
    dilation_hw = (
        int(ctx.option("dilationHFactor", 1)),
        int(ctx.option("dilationWFactor", 1)),
    )
    _, padding = _spatial_geometry(
        ctx, (value.shape[1], value.shape[2]), (kh, kw), stride_hw, dilation_hw
    )
    result = _conv2d_core(
        ctx,
        value,
        np.transpose(filters, (0, 3, 1, 2)),
        bias,
        stride_hw=stride_hw,
        dilation_hw=dilation_hw,
        padding=padding,
        groups=1,
    )
    return (apply_fused_activation(result.astype(value.dtype, copy=False), ctx),)


def _depthwise_conv_2d(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2, 3)
    value = ctx.input(0)
    filters = ctx.input(1)
    bias = ctx.optional_input(2)
    if (
        value.dtype.kind != "f"
        or filters.dtype != value.dtype
        or (bias is not None and bias.dtype != value.dtype)
    ):
        raise ctx.fail(
            "DEPTHWISE_CONV_2D requires FLOAT32 input, filter, and bias in native "
            f"mode; found {value.dtype}, {filters.dtype}, "
            f"{None if bias is None else bias.dtype}."
        )
    _require_rank(ctx, value, 4, what="input")
    _require_rank(ctx, filters, 4, what="filter")
    one, kh, kw, out_channels = filters.shape
    in_channels = value.shape[3]
    multiplier = int(ctx.option("depthMultiplier", 0))
    if one != 1:
        raise ctx.invalid(
            f"DEPTHWISE_CONV_2D filter shape {filters.shape} must start with 1."
        )
    if multiplier <= 0 or out_channels != in_channels * multiplier:
        raise ctx.invalid(
            f"DEPTHWISE_CONV_2D depthMultiplier {multiplier} does not satisfy "
            f"out_channels {out_channels} == in_channels {in_channels} * multiplier."
        )
    if bias is not None and bias.shape != (out_channels,):
        raise ctx.invalid(
            f"DEPTHWISE_CONV_2D bias shape {bias.shape} must be [{out_channels}]."
        )
    stride_hw = (int(ctx.option("strideH", 0)), int(ctx.option("strideW", 0)))
    dilation_hw = (
        int(ctx.option("dilationHFactor", 1)),
        int(ctx.option("dilationWFactor", 1)),
    )
    _, padding = _spatial_geometry(
        ctx, (value.shape[1], value.shape[2]), (kh, kw), stride_hw, dilation_hw
    )
    # [1, kh, kw, ic * m] -> [ic * m, 1, kh, kw]; PyTorch groups=ic yields the same
    # output channel order (group-major, multiplier-minor) as Circle.
    weight = np.transpose(filters.reshape(kh, kw, out_channels), (2, 0, 1))[
        :, None, :, :
    ]
    result = _conv2d_core(
        ctx,
        value,
        weight,
        bias,
        stride_hw=stride_hw,
        dilation_hw=dilation_hw,
        padding=padding,
        groups=in_channels,
    )
    return (apply_fused_activation(result.astype(value.dtype, copy=False), ctx),)


def _transpose_conv(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(3, 4)
    output_shape = ctx.index_vector(0, name="output_shape")
    filters = ctx.input(1)
    value = ctx.input(2)
    bias = ctx.optional_input(3)
    if (
        value.dtype.kind != "f"
        or filters.dtype != value.dtype
        or (bias is not None and bias.dtype != value.dtype)
    ):
        raise ctx.fail(
            "TRANSPOSE_CONV requires FLOAT32 input, filter, and bias in native mode; "
            f"found {value.dtype}, {filters.dtype}, {None if bias is None else bias.dtype}."
        )
    _require_rank(ctx, value, 4, what="input")
    _require_rank(ctx, filters, 4, what="filter")
    if len(output_shape) != 4 or any(dim <= 0 for dim in output_shape):
        raise ctx.invalid(
            f"TRANSPOSE_CONV output_shape {list(output_shape)} must hold four positive dimensions."
        )
    batch, in_h, in_w, in_channels = value.shape
    out_channels, kh, kw, filter_in = filters.shape
    if filter_in != in_channels:
        raise ctx.invalid(
            f"TRANSPOSE_CONV filter input channels {filter_in} do not match the input "
            f"channels {in_channels}."
        )
    out_batch, out_h, out_w, out_c = output_shape
    if out_batch != batch or out_c != out_channels:
        raise ctx.invalid(
            f"TRANSPOSE_CONV output_shape {list(output_shape)} does not match batch "
            f"{batch} and filter output channels {out_channels}."
        )
    if bias is not None and bias.shape != (out_channels,):
        raise ctx.invalid(
            f"TRANSPOSE_CONV bias shape {bias.shape} must be [{out_channels}]."
        )
    stride_h = int(ctx.option("strideH", 0))
    stride_w = int(ctx.option("strideW", 0))
    if stride_h <= 0 or stride_w <= 0:
        raise ctx.invalid(
            f"TRANSPOSE_CONV strides ({stride_h}, {stride_w}) must be positive."
        )
    padding = int(ctx.option("padding", -1))
    if padding not in (PADDING_SAME, PADDING_VALID):
        raise ctx.fail(f"TRANSPOSE_CONV padding enum value {padding} is not supported.")

    # luci-interpreter computes the implicit padding from the declared output size.
    def implicit_padding(out_size: int, filter_size: int, stride: int) -> int:
        if padding == PADDING_SAME:
            unused = (out_size + stride - 1) // stride
        else:
            unused = (out_size + stride - filter_size) // stride
        pad = ((unused - 1) * stride + filter_size - out_size) // 2
        return max(pad, 0)

    pad_h = implicit_padding(out_h, kh, stride_h)
    pad_w = implicit_padding(out_w, kw, stride_w)

    accum = np.zeros((batch, out_h, out_w, out_channels), dtype=np.float64)
    # out[n, iy*sh - pad_h + fy, ix*sw - pad_w + fx, oc] += sum_ic in[n, iy, ix, ic] * w[oc, fy, fx, ic]
    for fy in range(kh):
        for fx in range(kw):
            tap = filters[:, fy, fx, :].astype(np.float64)  # [oc, ic]
            contribution = np.tensordot(
                value.astype(np.float64), tap, axes=([3], [1])
            )  # [n, ih, iw, oc]
            for iy in range(in_h):
                oy = iy * stride_h - pad_h + fy
                if oy < 0 or oy >= out_h:
                    continue
                ox_start = -pad_w + fx
                ix_values = np.arange(in_w)
                ox_values = ix_values * stride_w + ox_start
                keep = (ox_values >= 0) & (ox_values < out_w)
                if not np.any(keep):
                    continue
                accum[:, oy, ox_values[keep], :] += contribution[
                    :, iy, ix_values[keep], :
                ]
    if bias is not None:
        accum += bias.astype(np.float64).reshape(1, 1, 1, out_channels)
    result = accum.astype(value.dtype)
    return (apply_fused_activation(result, ctx),)


def _pool_windows(
    ctx: KernelContext,
    value: np.ndarray,
    pad_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sliding pooling windows and a validity mask, both shaped [N,OH,OW,C,KH,KW]."""

    _require_rank(ctx, value, 4, what="input")
    filter_hw = (int(ctx.option("filterHeight", 0)), int(ctx.option("filterWidth", 0)))
    stride_hw = (int(ctx.option("strideH", 0)), int(ctx.option("strideW", 0)))
    if min(filter_hw) <= 0:
        raise ctx.invalid(f"{ctx.name} filter size {filter_hw} must be positive.")
    (out_h, out_w), ((top, bottom), (left, right)) = _spatial_geometry(
        ctx, (value.shape[1], value.shape[2]), filter_hw, stride_hw, (1, 1)
    )
    padded = np.pad(
        value.astype(np.float64),
        ((0, 0), (top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )
    valid = np.pad(
        np.ones(value.shape[1:3], dtype=bool),
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=False,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, filter_hw, axis=(1, 2)  # type: ignore[call-overload]
    )
    windows = windows[:, :: stride_hw[0], :: stride_hw[1], :, :, :][:, :out_h, :out_w]
    mask = np.lib.stride_tricks.sliding_window_view(valid, filter_hw)
    mask = mask[:: stride_hw[0], :: stride_hw[1]][:out_h, :out_w]
    return windows, mask


def _average_pool_2d(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    if value.dtype.kind != "f":
        raise ctx.fail(
            f"AVERAGE_POOL_2D requires a FLOAT32 input in native mode, found {value.dtype}."
        )
    windows, mask = _pool_windows(ctx, value, 0.0)
    counts = mask.sum(axis=(-1, -2))[None, :, :, None]
    if np.any(counts == 0):
        raise ctx.invalid("AVERAGE_POOL_2D produced a window without valid elements.")
    result = windows.sum(axis=(-1, -2)) / counts
    return (apply_fused_activation(result.astype(value.dtype), ctx),)


def _max_pool_2d(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    if value.dtype.kind != "f":
        raise ctx.fail(
            f"MAX_POOL_2D requires a FLOAT32 input in native mode, found {value.dtype}."
        )
    windows, mask = _pool_windows(ctx, value, -np.inf)
    if not np.all(mask.any(axis=(-1, -2))):
        raise ctx.invalid("MAX_POOL_2D produced a window without valid elements.")
    result = windows.max(axis=(-1, -2))
    return (apply_fused_activation(result.astype(value.dtype), ctx),)


def _fully_connected(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2, 3)
    value = ctx.input(0)
    weights = ctx.input(1)
    bias = ctx.optional_input(2)
    if (
        value.dtype.kind != "f"
        or weights.dtype != value.dtype
        or (bias is not None and bias.dtype != value.dtype)
    ):
        raise ctx.fail(
            "FULLY_CONNECTED requires FLOAT32 input, weights, and bias in native mode; "
            f"found {value.dtype}, {weights.dtype}, {None if bias is None else bias.dtype}."
        )
    if int(ctx.option("weightsFormat", 0)) != 0:
        raise ctx.fail(
            f"FULLY_CONNECTED weightsFormat {int(ctx.option('weightsFormat', 0))} is not supported."
        )
    _require_rank(ctx, weights, 2, what="weights")
    units, input_size = weights.shape
    if value.ndim == 0 or input_size == 0 or value.size % input_size != 0:
        raise ctx.invalid(
            f"FULLY_CONNECTED input shape {value.shape} is not divisible by the "
            f"weight input size {input_size}."
        )
    if bias is not None and bias.shape != (units,):
        raise ctx.invalid(f"FULLY_CONNECTED bias shape {bias.shape} must be [{units}].")
    flat = value.reshape(-1, input_size)
    result = np.matmul(flat, weights.T)
    if bias is not None:
        result = result + bias.reshape(1, units)
    if bool(ctx.option("keepNumDims", False)):
        if value.shape[-1] != input_size:
            raise ctx.invalid(
                f"FULLY_CONNECTED keepNumDims requires the last input dimension to be "
                f"{input_size}, found shape {value.shape}."
            )
        result = result.reshape(value.shape[:-1] + (units,))
    return (apply_fused_activation(result.astype(value.dtype, copy=False), ctx),)


def _batch_matmul(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    dtype = ctx.require_same_dtype(0, 1)
    if dtype.kind not in {"f", "i"}:
        raise ctx.fail(f"BATCH_MATMUL is not defined for dtype {dtype}.")
    lhs = ctx.input(0)
    rhs = ctx.input(1)
    if lhs.ndim < 2 or rhs.ndim < 2:
        raise ctx.invalid(
            f"BATCH_MATMUL inputs must have rank >= 2, found {lhs.shape} and {rhs.shape}."
        )
    if bool(ctx.option("adjointLhs", False)):
        lhs = np.swapaxes(lhs, -1, -2)
    if bool(ctx.option("adjointRhs", False)):
        rhs = np.swapaxes(rhs, -1, -2)
    if lhs.shape[-1] != rhs.shape[-2]:
        raise ctx.invalid(
            f"BATCH_MATMUL contraction sizes differ: {lhs.shape} x {rhs.shape}."
        )
    try:
        np.broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2])
    except ValueError as error:
        raise ctx.invalid(
            f"BATCH_MATMUL batch dimensions {lhs.shape[:-2]} and {rhs.shape[:-2]} "
            "are not broadcast-compatible."
        ) from error
    return (np.matmul(lhs, rhs).astype(dtype, copy=False),)


def _resize_size(ctx: KernelContext, value: np.ndarray) -> tuple[int, int]:
    _require_rank(ctx, value, 4, what="input")
    size = ctx.index_vector(1, name="size")
    if len(size) != 2 or min(size) <= 0:
        raise ctx.invalid(
            f"{ctx.name} size {list(size)} must hold two positive values."
        )
    return size[0], size[1]


def _resize_scale(in_size: int, out_size: int, align_corners: bool) -> float:
    if align_corners and out_size > 1:
        return (in_size - 1) / float(out_size - 1)
    return in_size / float(out_size)


def _bilinear_axis(
    in_size: int, out_size: int, align_corners: bool, half_pixel: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lower, upper, fraction) per output index using TFLite reference rules."""

    scale = np.float32(_resize_scale(in_size, out_size, align_corners))
    positions = np.arange(out_size, dtype=np.float32)
    if half_pixel:
        scaled = (positions + np.float32(0.5)) * scale - np.float32(0.5)
    else:
        scaled = positions * scale
    lower = np.maximum(np.floor(scaled).astype(np.int64), 0)
    upper = np.minimum(np.ceil(scaled).astype(np.int64), in_size - 1)
    fraction = scaled - np.floor(scaled)
    return lower, upper, fraction


def _resize_bilinear(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    value = ctx.input(0)
    if value.dtype.kind != "f":
        raise ctx.fail(
            f"RESIZE_BILINEAR requires a FLOAT32 input in native mode, found {value.dtype}."
        )
    out_h, out_w = _resize_size(ctx, value)
    align_corners = bool(ctx.option("alignCorners", False))
    half_pixel = bool(ctx.option("halfPixelCenters", False))
    if align_corners and half_pixel:
        raise ctx.invalid(
            "RESIZE_BILINEAR cannot enable both alignCorners and halfPixelCenters."
        )
    y0, y1, fy = _bilinear_axis(value.shape[1], out_h, align_corners, half_pixel)
    x0, x1, fx = _bilinear_axis(value.shape[2], out_w, align_corners, half_pixel)
    source = value.astype(np.float32)
    top = (
        source[:, y0][:, :, x0] * (1 - fx)[None, None, :, None]
        + source[:, y0][:, :, x1] * fx[None, None, :, None]
    )
    bottom = (
        source[:, y1][:, :, x0] * (1 - fx)[None, None, :, None]
        + source[:, y1][:, :, x1] * fx[None, None, :, None]
    )
    result = top * (1 - fy)[None, :, None, None] + bottom * fy[None, :, None, None]
    return (result.astype(value.dtype),)


def _nearest_axis(
    in_size: int, out_size: int, align_corners: bool, half_pixel: bool
) -> np.ndarray:
    scale = _resize_scale(in_size, out_size, align_corners)
    offset = 0.5 if half_pixel else 0.0
    positions = (
        np.arange(out_size, dtype=np.float32) + np.float32(offset)
    ) * np.float32(scale)
    if align_corners:
        # TfLiteRound rounds half away from zero.
        indices = np.where(
            positions >= 0, np.floor(positions + 0.5), np.ceil(positions - 0.5)
        )
    else:
        indices = np.floor(positions)
    indices = np.minimum(indices.astype(np.int64), in_size - 1)
    if half_pixel:
        indices = np.maximum(indices, 0)
    return indices


def _resize_nearest_neighbor(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    value = ctx.input(0)
    if value.dtype.kind not in {"f", "i", "u"}:
        raise ctx.fail(
            f"RESIZE_NEAREST_NEIGHBOR is not defined for dtype {value.dtype}."
        )
    out_h, out_w = _resize_size(ctx, value)
    align_corners = bool(ctx.option("alignCorners", False))
    half_pixel = bool(ctx.option("halfPixelCenters", False))
    ys = _nearest_axis(value.shape[1], out_h, align_corners, half_pixel)
    xs = _nearest_axis(value.shape[2], out_w, align_corners, half_pixel)
    return (np.ascontiguousarray(value[:, ys][:, :, xs]),)


def _instance_norm(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(3)
    value = ctx.input(0)
    gamma = ctx.input(1)
    beta = ctx.input(2)
    if (
        value.dtype.kind != "f"
        or gamma.dtype != value.dtype
        or beta.dtype != value.dtype
    ):
        raise ctx.fail(
            f"INSTANCE_NORM requires FLOAT32 input, gamma, and beta; found "
            f"{value.dtype}, {gamma.dtype}, {beta.dtype}."
        )
    if value.ndim == 4:
        reduce_axes: tuple[int, ...] = (1, 2)
        channel_axis = 3
    elif value.ndim == 3:
        reduce_axes = (2,)
        channel_axis = 1
    else:
        raise ctx.fail(
            f"INSTANCE_NORM supports rank 3 or 4 inputs, found shape {value.shape}."
        )
    channels = value.shape[channel_axis]
    for name, param in (("gamma", gamma), ("beta", beta)):
        if param.ndim != 1 or param.shape[0] not in (1, channels):
            raise ctx.invalid(
                f"INSTANCE_NORM {name} shape {param.shape} must be [1] or [{channels}]."
            )
    epsilon = float(ctx.option("epsilon", 0.0))
    source = value.astype(np.float64)
    mean = source.mean(axis=reduce_axes, keepdims=True)
    variance = (source * source).mean(axis=reduce_axes, keepdims=True) - mean * mean
    param_shape = [1] * value.ndim
    param_shape[channel_axis] = -1
    scale = gamma.astype(np.float64).reshape(param_shape) / np.sqrt(variance + epsilon)
    shift = -mean * scale + beta.astype(np.float64).reshape(param_shape)
    result = (source * scale + shift).astype(value.dtype)
    return (apply_fused_activation(result, ctx),)


def _rms_norm(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(2)
    value = ctx.input(0)
    gamma = ctx.input(1)
    if value.dtype.kind != "f" or gamma.dtype != value.dtype:
        raise ctx.fail(
            f"RMS_NORM requires FLOAT32 input and gamma; found {value.dtype}, {gamma.dtype}."
        )
    if value.ndim not in (3, 4):
        raise ctx.fail(
            f"RMS_NORM supports rank 3 or 4 inputs, found shape {value.shape}."
        )
    size = value.shape[-1]
    if gamma.ndim != 1 or gamma.shape[0] not in (1, size):
        raise ctx.invalid(
            f"RMS_NORM gamma shape {gamma.shape} must be [1] or [{size}]."
        )
    epsilon = float(ctx.option("epsilon", 0.0))
    source = value.astype(np.float64)
    rms = np.sqrt((source * source).mean(axis=-1, keepdims=True) + epsilon)
    result = gamma.astype(np.float64) * (source / rms)
    return (result.astype(value.dtype),)


def register_nn_kernels(registry: KernelRegistry) -> None:
    """Register convolution, pooling, matrix, resize, and normalization kernels."""

    registry.register_named("CONV_2D", _conv_2d)
    registry.register_named("DEPTHWISE_CONV_2D", _depthwise_conv_2d)
    registry.register_named("TRANSPOSE_CONV", _transpose_conv)
    registry.register_named("AVERAGE_POOL_2D", _average_pool_2d)
    registry.register_named("MAX_POOL_2D", _max_pool_2d)
    registry.register_named("FULLY_CONNECTED", _fully_connected)
    registry.register_named("BATCH_MATMUL", _batch_matmul)
    registry.register_named("RESIZE_BILINEAR", _resize_bilinear)
    registry.register_named("RESIZE_NEAREST_NEIGHBOR", _resize_nearest_neighbor)
    registry.register_named("INSTANCE_NORM", _instance_norm)
    registry.register_named("RMS_NORM", _rms_norm)


__all__ = ["register_nn_kernels"]
