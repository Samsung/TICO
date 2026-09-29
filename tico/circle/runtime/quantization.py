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

"""Affine quantization arithmetic shared by kernels and the fake-quantize mode.

Rounding follows the TFLite reference ``AffineQuantize`` used by
luci-interpreter: ``round(value / scale)`` with ties rounded away from zero,
then ``+ zero_point`` and clamping to the representable range of the target
tensor type. Per-channel parameters broadcast along ``quantizedDimension``.
"""

from __future__ import annotations

import numpy as np

from tico.circle.runtime.errors import (
    CircleRuntimeValidationError,
    UnsupportedCircleOperatorError,
)
from tico.circle.runtime.program import (
    RuntimeTensor,
    tensor_type_name,
    tensor_type_spec,
)
from tico.circle.value import TensorQuantization


def quantized_range(tensor_type: int) -> tuple[int, int]:
    """Return the representable integer range of a quantized tensor type."""

    spec = tensor_type_spec(tensor_type)
    if spec.packed:
        return (-8, 7) if spec.signed else (0, 15)
    if spec.logical_dtype.kind not in {"i", "u"}:
        raise UnsupportedCircleOperatorError(
            f"TensorType.{spec.name} is not an integer quantized storage type."
        )
    info = np.iinfo(spec.logical_dtype)
    return int(info.min), int(info.max)


def round_half_away_from_zero(value: np.ndarray) -> np.ndarray:
    """Round like C ``std::round``: ties move away from zero."""

    return np.where(value >= 0, np.floor(value + 0.5), np.ceil(value - 0.5))


def _broadcast_parameters(
    quantization: TensorQuantization,
    shape: tuple[int, ...],
    tensor: RuntimeTensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scale and zero-point arrays broadcastable to ``shape``."""

    scales = np.asarray(quantization.scale, dtype=np.float32)
    zero_points = np.asarray(quantization.zero_point, dtype=np.float32)
    if scales.size == 0:
        raise CircleRuntimeValidationError(
            f"{tensor.describe()} has quantization metadata without scales."
        )
    if zero_points.size != scales.size:
        raise CircleRuntimeValidationError(
            f"{tensor.describe()} has {scales.size} scales but {zero_points.size} "
            "zero points."
        )
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise CircleRuntimeValidationError(
            f"{tensor.describe()} has a non-positive or non-finite quantization "
            f"scale: {scales.tolist()}."
        )
    if scales.size == 1:
        return scales.reshape(()), zero_points.reshape(())
    rank = len(shape)
    axis = int(quantization.quantized_dimension)
    if axis < 0 or axis >= rank or shape[axis] != scales.size:
        raise CircleRuntimeValidationError(
            f"{tensor.describe()} has {scales.size} per-channel scales on "
            f"quantized dimension {axis}, which does not match shape {shape}."
        )
    broadcast_shape = [1] * rank
    broadcast_shape[axis] = scales.size
    return scales.reshape(broadcast_shape), zero_points.reshape(broadcast_shape)


def dequantize_array(value: np.ndarray, tensor: RuntimeTensor) -> np.ndarray:
    """Convert an integer tensor value to FLOAT32 with its serialized qparams."""

    quantization = tensor.quantization
    if quantization is None or not quantization.scale:
        raise CircleRuntimeValidationError(
            f"{tensor.describe()} cannot be dequantized without quantization "
            "parameters."
        )
    if value.dtype.kind not in {"i", "u"}:
        raise UnsupportedCircleOperatorError(
            f"{tensor.describe()} holds {value.dtype} data, but dequantization "
            "requires integer storage."
        )
    scale, zero_point = _broadcast_parameters(quantization, value.shape, tensor)
    return ((value.astype(np.float32) - zero_point) * scale).astype(
        np.float32, copy=False
    )


def quantize_array(value: np.ndarray, tensor: RuntimeTensor) -> np.ndarray:
    """Quantize a FLOAT32 value into the integer dtype of ``tensor``."""

    quantization = tensor.quantization
    if quantization is None or not quantization.scale:
        raise CircleRuntimeValidationError(
            f"{tensor.describe()} cannot be quantized without quantization "
            "parameters."
        )
    if value.dtype.kind != "f":
        raise UnsupportedCircleOperatorError(
            f"Quantization to {tensor_type_name(tensor.tensor_type)} requires a "
            f"float source, found {value.dtype}."
        )
    scale, zero_point = _broadcast_parameters(quantization, value.shape, tensor)
    minimum, maximum = quantized_range(tensor.tensor_type)
    with np.errstate(all="ignore"):
        scaled = round_half_away_from_zero(value.astype(np.float32) / scale)
        scaled = scaled + zero_point
    clipped = np.clip(scaled, minimum, maximum)
    return clipped.astype(tensor.dtype)


def fake_quantize_array(value: np.ndarray, tensor: RuntimeTensor) -> np.ndarray:
    """Round-trip a FLOAT32 value through ``tensor``'s quantization grid."""

    quantized = quantize_array(value, tensor)
    return dequantize_array(quantized, tensor)


__all__ = [
    "dequantize_array",
    "fake_quantize_array",
    "quantize_array",
    "quantized_range",
    "round_half_away_from_zero",
]
