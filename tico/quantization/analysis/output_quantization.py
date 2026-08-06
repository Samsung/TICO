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

"""Per-tensor affine quantization used by output-range analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme


@dataclass(frozen=True)
class AffineQuantizationPolicy:
    """Describe one per-tensor affine integer quantization grid."""

    dtype: DType
    qscheme: QScheme

    def __post_init__(self) -> None:
        if self.qscheme.is_per_channel():
            raise ValueError("Output analysis supports per-tensor policies only.")
        if not self.dtype.signed and self.qscheme.is_symmetric():
            raise ValueError("Unsigned dtypes cannot use symmetric quantization.")

    @property
    def quant_min(self) -> int:
        return self.dtype.qmin

    @property
    def quant_max(self) -> int:
        return self.dtype.qmax

    @property
    def name(self) -> str:
        return str(self.dtype)

    @classmethod
    def uint8(cls) -> "AffineQuantizationPolicy":
        """Return a per-tensor asymmetric UINT8 policy."""
        return cls(DType.uint(8), QScheme.PER_TENSOR_ASYMM)

    @classmethod
    def int16(cls) -> "AffineQuantizationPolicy":
        """Return a per-tensor symmetric INT16 policy."""
        return cls(DType.int(16), QScheme.PER_TENSOR_SYMM)


@dataclass(frozen=True)
class AffineQParams:
    """Store scalar affine qparams and the clipping range that produced them."""

    scale: float
    zero_point: int
    minimum: float
    maximum: float


def calculate_qparams(
    policy: AffineQuantizationPolicy,
    minimum: float,
    maximum: float,
) -> AffineQParams:
    """Calculate scalar qparams with the same range rules as WrapQ affine PTQ."""
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("Quantization bounds must be finite.")
    if minimum > maximum:
        raise ValueError(f"Expected minimum <= maximum, got {minimum} > {maximum}.")

    qmin = policy.quant_min
    qmax = policy.quant_max
    eps = 1e-12
    observed_minimum = float(minimum)
    observed_maximum = float(maximum)

    if policy.qscheme.is_symmetric():
        maximum_absolute = max(abs(observed_minimum), abs(observed_maximum))
        scale = max(maximum_absolute, eps) / qmax
        zero_point = 0
    else:
        value_range = observed_maximum - observed_minimum
        if abs(value_range) < 1e-8:
            if observed_minimum == 0.0:
                scale = 1.0
                zero_point = 0
            elif observed_minimum > 0.0:
                scale = max(observed_minimum, eps)
                zero_point = 0
            else:
                scale = max(abs(observed_minimum), eps)
                zero_point = qmax
        else:
            minimum_with_zero = min(observed_minimum, 0.0)
            maximum_with_zero = max(observed_maximum, 0.0)
            value_range = maximum_with_zero - minimum_with_zero
            scale = max(value_range, eps) / (qmax - qmin)
            zero_point = int(round(qmin - observed_minimum / scale))
            zero_point = min(max(zero_point, qmin), qmax)

    return AffineQParams(
        scale=float(scale),
        zero_point=int(zero_point),
        minimum=observed_minimum,
        maximum=observed_maximum,
    )


class OutputTensorQuantizer:
    """Apply one frozen per-tensor quantization grid to arbitrary tensors."""

    def __init__(
        self,
        name: str,
        policy: AffineQuantizationPolicy,
        qparams: AffineQParams,
    ) -> None:
        self.name = name
        self.policy = policy
        self.qparams = qparams

    @classmethod
    def from_range(
        cls,
        name: str,
        policy: AffineQuantizationPolicy,
        minimum: float,
        maximum: float,
    ) -> "OutputTensorQuantizer":
        """Construct a quantizer from explicit real-valued bounds."""
        return cls(name, policy, calculate_qparams(policy, minimum, maximum))

    def fake_quant(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize ``tensor`` with the frozen scalar grid."""
        return torch.fake_quantize_per_tensor_affine(
            tensor,
            scale=self.qparams.scale,
            zero_point=self.qparams.zero_point,
            quant_min=self.policy.quant_min,
            quant_max=self.policy.quant_max,
        )

    def integer_codes(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unclamped and clamped integer codes."""
        unclamped = (
            torch.round(tensor.detach() / self.qparams.scale) + self.qparams.zero_point
        )
        clamped = unclamped.clamp(
            self.policy.quant_min,
            self.policy.quant_max,
        ).to(torch.int64)
        return unclamped, clamped

    def representable_range(self) -> tuple[float, float]:
        """Return the dequantized integer endpoints."""
        minimum = (self.policy.quant_min - self.qparams.zero_point) * self.qparams.scale
        maximum = (self.policy.quant_max - self.qparams.zero_point) * self.qparams.scale
        return float(minimum), float(maximum)


@dataclass
class OutputCodeStatistics:
    """Aggregate saturation and integer-code utilization for one output."""

    count: int = 0
    evaluation_minimum: float = math.inf
    evaluation_maximum: float = -math.inf
    low_saturation_count: int = 0
    high_saturation_count: int = 0
    histogram: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int64)
    )

    def update(
        self,
        tensor: torch.Tensor,
        quantizer: OutputTensorQuantizer,
    ) -> None:
        """Accumulate statistics from one floating-point output tensor."""
        value = tensor.detach()
        if value.numel() == 0:
            return
        unclamped, clamped = quantizer.integer_codes(value)
        qmin = quantizer.policy.quant_min
        qmax = quantizer.policy.quant_max
        number_of_codes = qmax - qmin + 1
        if self.histogram.numel() == 0:
            self.histogram = torch.zeros(number_of_codes, dtype=torch.int64)

        shifted = (clamped.reshape(-1).cpu() - qmin).to(torch.int64)
        self.histogram += torch.bincount(shifted, minlength=number_of_codes)
        self.count += value.numel()
        self.evaluation_minimum = min(self.evaluation_minimum, float(value.min()))
        self.evaluation_maximum = max(self.evaluation_maximum, float(value.max()))
        self.low_saturation_count += int((unclamped < qmin).sum().cpu())
        self.high_saturation_count += int((unclamped > qmax).sum().cpu())

    def summary(self, quantizer: OutputTensorQuantizer) -> dict[str, float | int]:
        """Return JSON-compatible output-grid statistics."""
        if self.count == 0:
            raise RuntimeError("No output values were accumulated.")
        representable_minimum, representable_maximum = quantizer.representable_range()
        used = torch.nonzero(self.histogram, as_tuple=False).reshape(-1)
        total_saturation = self.low_saturation_count + self.high_saturation_count
        return {
            "dtype": quantizer.policy.name,
            "quant_min": quantizer.policy.quant_min,
            "quant_max": quantizer.policy.quant_max,
            "clip_minimum": quantizer.qparams.minimum,
            "clip_maximum": quantizer.qparams.maximum,
            "scale": quantizer.qparams.scale,
            "zero_point": quantizer.qparams.zero_point,
            "representable_minimum": representable_minimum,
            "representable_maximum": representable_maximum,
            "estimated_uniform_rounding_mae": quantizer.qparams.scale / 4.0,
            "evaluation_minimum": self.evaluation_minimum,
            "evaluation_maximum": self.evaluation_maximum,
            "low_saturation_count": self.low_saturation_count,
            "high_saturation_count": self.high_saturation_count,
            "saturation_count": total_saturation,
            "low_saturation_ratio": self.low_saturation_count / self.count,
            "high_saturation_ratio": self.high_saturation_count / self.count,
            "saturation_ratio": total_saturation / self.count,
            "used_integer_codes": int(used.numel()),
            "value_count": self.count,
        }
