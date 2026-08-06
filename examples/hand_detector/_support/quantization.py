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

"""WrapQ preparation and Circle export helpers for the hand detector."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Sequence

import tico
import torch

from examples.hand_detector._support.circle import save_layout_optimized_circle
from tico.ops import Concat, ResizeBilinear2d
from tico.quantization import convert as freeze_quantization, prepare, QuantStub
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_conv2d import QuantConv2d
from tico.quantization.wrapq.wrappers.nn.quant_maxpool2d import QuantMaxPool2d
from tico.quantization.wrapq.wrappers.nn.quant_prelu import QuantPReLU
from tico.quantization.wrapq.wrappers.ops.quant_concat import QuantConcat
from tico.quantization.wrapq.wrappers.ops.quant_resize_bilinear import (
    QuantResizeBilinear2d,
)
from tico.quantization.wrapq.wrappers.quant_stub import QuantStubWrapper
from torch import nn


_FLOAT_MODULE_TYPES = (
    QuantStub,
    nn.Conv2d,
    nn.MaxPool2d,
    nn.PReLU,
    Concat,
    ResizeBilinear2d,
)
_QUANT_MODULE_TYPES = (
    QuantStubWrapper,
    QuantConv2d,
    QuantMaxPool2d,
    QuantPReLU,
    QuantConcat,
    QuantResizeBilinear2d,
)
_SUPPORTED_BIT_WIDTHS = (8, 16)


def validate_bit_width(bit_width: int) -> None:
    """Reject bit widths unsupported by this example backend profile."""
    if bit_width not in _SUPPORTED_BIT_WIDTHS:
        raise ValueError(
            f"Expected one of {_SUPPORTED_BIT_WIDTHS}, but received {bit_width}."
        )


def quantization_name(bit_width: int) -> str:
    """Return the lowercase dtype name for one example bit width."""
    validate_bit_width(bit_width)
    return "uint8" if bit_width == 8 else "int16"


def quantization_label(bit_width: int) -> str:
    """Return the uppercase dtype label for one example bit width."""
    return quantization_name(bit_width).upper()


def make_ptq_config(
    bit_width: int,
    *,
    activation_observer: type[ObserverBase] = MinMaxObserver,
    activation_observer_kwargs: Mapping[str, object] | None = None,
) -> PTQConfig:
    """Create the example PTQ policy with a selectable activation observer."""
    validate_bit_width(bit_width)
    if bit_width == 8:
        dtype = DType.uint(8)
        activation_qscheme = QScheme.PER_TENSOR_ASYMM
        weight_qscheme = QScheme.PER_CHANNEL_ASYMM
    else:
        dtype = DType.int(16)
        activation_qscheme = QScheme.PER_TENSOR_SYMM
        weight_qscheme = QScheme.PER_CHANNEL_SYMM

    return PTQConfig(
        activation=affine(
            dtype,
            qscheme=activation_qscheme,
            observer=activation_observer,
            **dict(activation_observer_kwargs or {}),
        ),
        weight=affine(
            dtype,
            qscheme=weight_qscheme,
            observer=MinMaxObserver,
        ),
        strict_wrap=False,
    )


def calibrate(model: nn.Module, samples: Sequence[torch.Tensor]) -> None:
    """Collect observer statistics from representative NHWC inputs."""
    if not samples:
        raise ValueError("Calibration requires at least one input sample.")
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            model(sample)


def prepare_quantized_candidate(
    float_model: nn.Module,
    bit_width: int,
    *,
    activation_observer: type[ObserverBase] = MinMaxObserver,
    activation_observer_kwargs: Mapping[str, object] | None = None,
) -> nn.Module:
    """Clone, WrapQ-prepare, and validate one hand-detector candidate."""
    candidate = copy.deepcopy(float_model).eval()
    expected_wrappers = sum(
        isinstance(module, _FLOAT_MODULE_TYPES) for module in candidate.modules()
    )
    candidate = prepare(
        candidate,
        make_ptq_config(
            bit_width,
            activation_observer=activation_observer,
            activation_observer_kwargs=activation_observer_kwargs,
        ),
        inplace=True,
    )
    actual_wrappers = sum(
        isinstance(module, _QUANT_MODULE_TYPES) for module in candidate.modules()
    )
    if actual_wrappers != expected_wrappers:
        raise RuntimeError(
            f"Expected {expected_wrappers} quantization wrappers for "
            f"{quantization_label(bit_width)}, but found {actual_wrappers}."
        )
    return candidate


def quantize_candidate(
    float_model: nn.Module,
    bit_width: int,
    calibration_samples: Sequence[torch.Tensor],
    *,
    activation_observer: type[ObserverBase] = MinMaxObserver,
    activation_observer_kwargs: Mapping[str, object] | None = None,
) -> nn.Module:
    """Prepare, calibrate, and freeze one hand-detector candidate."""
    candidate = prepare_quantized_candidate(
        float_model,
        bit_width,
        activation_observer=activation_observer,
        activation_observer_kwargs=activation_observer_kwargs,
    )
    calibrate(candidate, calibration_samples)
    candidate = freeze_quantization(candidate, inplace=True)
    return candidate.eval()


def get_example_inputs(model: nn.Module) -> tuple[torch.Tensor, ...]:
    """Return static example inputs exposed by a converted detector module."""
    if not hasattr(model, "get_example_inputs"):
        raise TypeError("The hand detector must expose get_example_inputs().")
    inputs = model.get_example_inputs()  # type: ignore[attr-defined]
    if not isinstance(inputs, tuple) or not inputs:
        raise TypeError("get_example_inputs() must return a non-empty tuple.")
    if not all(isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("Every example input must be a Tensor.")
    return inputs


def export_quantized_circle(
    quantized_model: nn.Module,
    output_path: str | Path,
) -> Path:
    """Export, layout-optimize, and save a frozen fake-quantized Circle model."""
    with torch.inference_mode():
        circle_model = tico.convert(
            quantized_model.eval(),
            get_example_inputs(quantized_model),
        )
    output, result = save_layout_optimized_circle(circle_model, output_path)
    print(f"Circle layout optimization reported {result.changes} changes.")
    return output
