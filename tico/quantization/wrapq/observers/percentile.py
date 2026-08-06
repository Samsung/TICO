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

"""Bounded-memory percentile observer for per-tensor affine quantization."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from tico.quantization.wrapq.dtypes import DType, UINT8
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme


class PercentileObserver(AffineObserverBase):
    """Estimate clipping bounds from a bounded calibration-value reservoir.

    The observer always tracks the exact global minimum and maximum, but keeps
    only a bounded, approximately uniform sample of activation values for
    percentile estimation. This avoids retaining every element from every
    calibration tensor.

    ``percentile`` describes the retained central mass when explicit lower and
    upper percentiles are not supplied. For example, ``99.9`` means:

    * non-negative data: ``[0, q(99.9%)]``;
    * non-positive data: ``[q(0.1%), 0]``;
    * signed asymmetric data: equal clipping in both tails;
    * symmetric quantization: the 99.9th percentile of ``abs(x)``.

    Explicit ``lower_percentile`` and ``upper_percentile`` values may be used
    for asymmetric clipping. Per-channel percentile collection is intentionally
    unsupported because a separate bounded reservoir per channel would have a
    substantially different memory and API contract.
    """

    def __init__(
        self,
        *,
        name: str,
        dtype: DType = UINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_ASYMM,
        channel_axis: Optional[int] = None,
        percentile: float = 99.99,
        lower_percentile: Optional[float] = None,
        upper_percentile: Optional[float] = None,
        max_samples: int = 131_072,
        samples_per_batch: int = 4_096,
        seed: int = 0,
    ) -> None:
        """Create a bounded-memory percentile observer."""
        if qscheme.is_per_channel() or channel_axis is not None:
            raise ValueError(
                "PercentileObserver supports per-tensor quantization only."
            )
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in the interval (0, 100].")
        if (lower_percentile is None) != (upper_percentile is None):
            raise ValueError(
                "lower_percentile and upper_percentile must be set together."
            )
        if lower_percentile is not None and upper_percentile is not None:
            if qscheme.is_symmetric():
                raise ValueError(
                    "Explicit lower/upper percentiles require an asymmetric qscheme."
                )
            if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
                raise ValueError(
                    "Expected 0 <= lower_percentile < upper_percentile <= 100."
                )
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        if samples_per_batch <= 0:
            raise ValueError("samples_per_batch must be positive.")

        self.percentile = float(percentile)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.max_samples = int(max_samples)
        self.samples_per_batch = int(samples_per_batch)
        self.seed = int(seed)

        # Calibration samples intentionally remain on CPU and outside the state
        # dict. They are temporary statistics rather than model parameters.
        self._samples_cpu = torch.empty(0, dtype=torch.float32)
        self._priorities_cpu = torch.empty(0, dtype=torch.float64)
        self._generator = torch.Generator(device="cpu")
        self._collection_count = 0

        super().__init__(
            name=name,
            dtype=dtype,
            qscheme=qscheme,
            channel_axis=None,
        )
        self.register_buffer(
            "clip_min_val",
            torch.tensor(math.inf),
            persistent=False,
        )
        self.register_buffer(
            "clip_max_val",
            torch.tensor(-math.inf),
            persistent=False,
        )
        self.reset()

    @property
    def sampled_value_count(self) -> int:
        """Return the number of calibration values retained in memory."""
        return int(self._samples_cpu.numel())

    def reset(self) -> None:
        """Clear exact ranges, percentile samples, and cached qparams."""
        super().reset()
        self._samples_cpu = torch.empty(0, dtype=torch.float32)
        self._priorities_cpu = torch.empty(0, dtype=torch.float64)
        self._generator.manual_seed(self.seed)
        self._collection_count = 0
        if hasattr(self, "clip_min_val"):
            self.clip_min_val.fill_(math.inf)
            self.clip_max_val.fill_(-math.inf)

    def _update_stats(self, x: torch.Tensor) -> None:
        """Update the exact range and bounded percentile reservoir."""
        values = x.detach().reshape(-1)
        if values.numel() == 0:
            return

        finite = torch.isfinite(values)
        if not bool(finite.all()):
            values = values[finite]
        if values.numel() == 0:
            return

        current_minimum = values.min().to(self.min_val.device, self.min_val.dtype)
        current_maximum = values.max().to(self.max_val.device, self.max_val.dtype)
        self.min_val.copy_(torch.minimum(self.min_val, current_minimum))
        self.max_val.copy_(torch.maximum(self.max_val, current_maximum))

        sampled = self._sample_values(values)
        sampled_cpu = sampled.to(device="cpu", dtype=torch.float32)
        priorities = torch.rand(
            sampled_cpu.numel(),
            generator=self._generator,
            dtype=torch.float64,
        )
        self._merge_reservoir(sampled_cpu, priorities)
        self._collection_count += 1

    def _sample_values(self, values: torch.Tensor) -> torch.Tensor:
        """Select a bounded, spatially distributed subset from one tensor."""
        count = min(values.numel(), self.samples_per_batch)
        if count == values.numel():
            return values

        # Shift the regular grid between calibration calls. This avoids always
        # selecting the same spatial locations while avoiding a full randperm
        # of a potentially very large activation tensor.
        phase = math.fmod(self._collection_count * 0.6180339887498949, 1.0)
        positions = (
            torch.arange(count, device=values.device, dtype=torch.float64) + phase
        ) / count
        indices = torch.floor(positions * values.numel()).to(torch.long)
        return values.index_select(0, indices)

    def _merge_reservoir(
        self,
        values: torch.Tensor,
        priorities: torch.Tensor,
    ) -> None:
        """Keep the highest-priority samples seen across all calibration calls."""
        if values.numel() == 0:
            return

        if self._samples_cpu.numel() < self.max_samples:
            merged_values = torch.cat((self._samples_cpu, values))
            merged_priorities = torch.cat((self._priorities_cpu, priorities))
        else:
            threshold = self._priorities_cpu.min()
            admitted = priorities > threshold
            if not bool(admitted.any()):
                return
            merged_values = torch.cat((self._samples_cpu, values[admitted]))
            merged_priorities = torch.cat((self._priorities_cpu, priorities[admitted]))

        if merged_values.numel() > self.max_samples:
            keep = torch.topk(
                merged_priorities,
                k=self.max_samples,
                largest=True,
                sorted=False,
            ).indices
            merged_values = merged_values.index_select(0, keep)
            merged_priorities = merged_priorities.index_select(0, keep)

        self._samples_cpu = merged_values
        self._priorities_cpu = merged_priorities

    def _clipping_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate scalar clipping bounds from retained samples."""
        if self._samples_cpu.numel() == 0:
            raise RuntimeError(
                f"No calibration values were collected for observer {self.name!r}."
            )

        values = self._samples_cpu.to(torch.float64)
        observed_minimum = float(self.min_val.detach().cpu())
        observed_maximum = float(self.max_val.detach().cpu())

        if self.qscheme.is_symmetric():
            if self.percentile == 100.0:
                threshold = values.new_tensor(
                    max(abs(observed_minimum), abs(observed_maximum))
                )
            else:
                threshold = torch.quantile(
                    values.abs(),
                    self.percentile / 100.0,
                )
            minimum = -threshold
            maximum = threshold
        elif self.lower_percentile is not None and self.upper_percentile is not None:
            minimum = (
                values.new_tensor(observed_minimum)
                if self.lower_percentile == 0.0
                else torch.quantile(values, self.lower_percentile / 100.0)
            )
            maximum = (
                values.new_tensor(observed_maximum)
                if self.upper_percentile == 100.0
                else torch.quantile(values, self.upper_percentile / 100.0)
            )
        elif self.percentile == 100.0:
            minimum = values.new_tensor(observed_minimum)
            maximum = values.new_tensor(observed_maximum)
        elif observed_minimum >= 0.0:
            minimum = values.new_tensor(0.0)
            maximum = torch.quantile(values, self.percentile / 100.0)
        elif observed_maximum <= 0.0:
            minimum = torch.quantile(values, 1.0 - self.percentile / 100.0)
            maximum = values.new_tensor(0.0)
        else:
            tail = (1.0 - self.percentile / 100.0) / 2.0
            minimum = torch.quantile(values, tail)
            maximum = torch.quantile(values, 1.0 - tail)

        # Affine quantization must represent real zero exactly. The base affine
        # observer also enforces this, but making it explicit keeps the reported
        # clipping bounds aligned with the actual quantization interval.
        minimum = torch.minimum(minimum, minimum.new_tensor(0.0))
        maximum = torch.maximum(maximum, maximum.new_tensor(0.0))
        return minimum, maximum

    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute affine qparams from percentile-clipped scalar bounds."""
        minimum, maximum = self._clipping_bounds()
        minimum = minimum.to(self.min_val.device, self.min_val.dtype)
        maximum = maximum.to(self.max_val.device, self.max_val.dtype)
        self.clip_min_val.copy_(minimum)
        self.clip_max_val.copy_(maximum)

        observed_minimum = self.min_val.detach().clone()
        observed_maximum = self.max_val.detach().clone()
        self.min_val.copy_(minimum)
        self.max_val.copy_(maximum)
        try:
            return super().compute_qparams()
        finally:
            self.min_val.copy_(observed_minimum)
            self.max_val.copy_(observed_maximum)

    def extra_repr(self) -> str:
        """Return percentile-specific settings for module summaries."""
        explicit = ""
        if self.lower_percentile is not None:
            explicit = (
                f", lower_percentile={self.lower_percentile}, "
                f"upper_percentile={self.upper_percentile}"
            )
        return (
            f"percentile={self.percentile}{explicit}, "
            f"max_samples={self.max_samples}, "
            f"samples_per_batch={self.samples_per_batch}"
        )
