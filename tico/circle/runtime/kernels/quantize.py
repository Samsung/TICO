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

"""QUANTIZE and DEQUANTIZE kernels."""

from __future__ import annotations

import numpy as np

from tico.circle.runtime.kernels.base import (
    ExecutionMode,
    KernelContext,
    KernelRegistry,
)
from tico.circle.runtime.quantization import (
    dequantize_array,
    fake_quantize_array,
    quantize_array,
)


def _quantize(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    input_tensor = ctx.input_tensor(0)
    output_tensor = ctx.output_tensor(0)
    if not output_tensor.is_quantized_integer:
        raise ctx.invalid(
            "QUANTIZE output must be an integer tensor with quantization parameters."
        )
    if ctx.mode is ExecutionMode.FAKE_QUANTIZE:
        # The activation is already FLOAT32 here; the executor keeps quantized
        # activations dequantized, so QUANTIZE reduces to a grid round trip.
        if value.dtype.kind != "f":
            raise ctx.fail(
                f"QUANTIZE in fake-quantize mode expects a float value, found {value.dtype}."
            )
        return (fake_quantize_array(value, output_tensor),)
    if value.dtype.kind == "f":
        return (quantize_array(value, output_tensor),)
    if input_tensor.is_quantized_integer:
        # Requantization: dequantize with the input parameters, then quantize with
        # the output parameters. TFLite realises this through fixed-point
        # multipliers; the float composition is the reference definition.
        return (quantize_array(dequantize_array(value, input_tensor), output_tensor),)
    raise ctx.fail(
        f"QUANTIZE from {value.dtype} without quantization parameters is not defined."
    )


def _dequantize(ctx: KernelContext) -> tuple[np.ndarray, ...]:
    ctx.require_inputs(1)
    value = ctx.input(0)
    input_tensor = ctx.input_tensor(0)
    target = ctx.output_dtype(0)
    if target.kind != "f":
        raise ctx.fail(f"DEQUANTIZE output must be a float tensor, found {target}.")
    if value.dtype.kind == "f":
        # FLOAT16 -> FLOAT32 conversion, or an already dequantized activation in
        # fake-quantize mode.
        return (value.astype(target),)
    if not input_tensor.is_quantized:
        raise ctx.fail(
            f"DEQUANTIZE input {value.dtype} carries no quantization parameters."
        )
    return (dequantize_array(value, input_tensor).astype(target, copy=False),)


def register_quantize_kernels(registry: KernelRegistry) -> None:
    """Register QUANTIZE and DEQUANTIZE."""

    registry.register_named("QUANTIZE", _quantize)
    registry.register_named("DEQUANTIZE", _dequantize)


__all__ = ["register_quantize_kernels"]
