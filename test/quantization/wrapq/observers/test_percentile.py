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

"""Tests for bounded-memory percentile activation observation."""

from __future__ import annotations

import unittest
from typing import Any

import torch

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.qscheme import QScheme


class PercentileObserverTest(unittest.TestCase):
    """Verify clipping, qparams, and bounded calibration storage."""

    @staticmethod
    def _uint8(**kwargs: Any) -> PercentileObserver:
        """Create one UINT8 per-tensor observer."""
        return PercentileObserver(
            name="activation",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
            **kwargs,
        )

    def test_positive_distribution_clips_only_the_upper_tail(self) -> None:
        """Keep zero as the lower endpoint for non-negative activations."""
        observer = self._uint8(percentile=80.0, max_samples=64)
        observer.collect(torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0]))
        scale, zero_point = observer.compute_qparams()

        expected_maximum = torch.quantile(
            torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0], dtype=torch.float64),
            0.8,
        )
        self.assertEqual(float(observer.clip_min_val), 0.0)
        self.assertAlmostEqual(
            float(observer.clip_max_val),
            float(expected_maximum),
            places=5,
        )
        self.assertEqual(int(zero_point), 0)
        self.assertAlmostEqual(float(scale), float(expected_maximum) / 255.0)

    def test_signed_distribution_uses_equal_default_tails(self) -> None:
        """Interpret percentile as central retained mass for signed data."""
        values = torch.tensor([-100.0, -3.0, -2.0, 0.0, 2.0, 4.0, 80.0])
        observer = self._uint8(percentile=80.0, max_samples=64)
        observer.collect(values)
        observer.compute_qparams()

        expected_minimum = torch.quantile(values.to(torch.float64), 0.1)
        expected_maximum = torch.quantile(values.to(torch.float64), 0.9)
        self.assertAlmostEqual(
            float(observer.clip_min_val),
            float(expected_minimum),
            places=5,
        )
        self.assertAlmostEqual(
            float(observer.clip_max_val),
            float(expected_maximum),
            places=5,
        )

    def test_explicit_asymmetric_percentiles_are_supported(self) -> None:
        """Allow independent lower and upper clipping percentages."""
        values = torch.arange(-10.0, 11.0)
        observer = self._uint8(
            lower_percentile=10.0,
            upper_percentile=80.0,
            max_samples=64,
        )
        observer.collect(values)
        observer.compute_qparams()

        self.assertAlmostEqual(
            float(observer.clip_min_val),
            float(torch.quantile(values.to(torch.float64), 0.1)),
            places=5,
        )
        self.assertAlmostEqual(
            float(observer.clip_max_val),
            float(torch.quantile(values.to(torch.float64), 0.8)),
            places=5,
        )

    def test_symmetric_scheme_uses_absolute_percentile(self) -> None:
        """Generate equal-magnitude bounds for signed symmetric quantization."""
        values = torch.tensor([-100.0, -2.0, -1.0, 0.0, 1.0, 3.0])
        observer = PercentileObserver(
            name="activation",
            dtype=DType.int(16),
            qscheme=QScheme.PER_TENSOR_SYMM,
            percentile=80.0,
            max_samples=64,
        )
        observer.collect(values)
        scale, zero_point = observer.compute_qparams()

        threshold = torch.quantile(values.abs().to(torch.float64), 0.8)
        self.assertAlmostEqual(float(observer.clip_min_val), -float(threshold))
        self.assertAlmostEqual(float(observer.clip_max_val), float(threshold))
        self.assertEqual(int(zero_point), 0)
        self.assertAlmostEqual(float(scale), float(threshold) / 32767.0)

    def test_percentile_100_uses_exact_observed_extrema(self) -> None:
        """Make the 100-percent setting numerically identical to MinMax."""
        observer = self._uint8(
            percentile=100.0,
            max_samples=2,
            samples_per_batch=2,
        )
        observer.collect(torch.tensor([-100.0, -1.0, 0.0, 2.0, 80.0]))
        observer.compute_qparams()

        self.assertEqual(float(observer.clip_min_val), -100.0)
        self.assertEqual(float(observer.clip_max_val), 80.0)

    def test_reservoir_is_bounded(self) -> None:
        """Never retain more than the configured number of values."""
        observer = self._uint8(
            percentile=99.0,
            max_samples=17,
            samples_per_batch=13,
            seed=7,
        )
        for index in range(20):
            observer.collect(torch.arange(100.0) + index)
        self.assertEqual(observer.sampled_value_count, 17)
        self.assertEqual(float(observer.min_val), 0.0)
        self.assertEqual(float(observer.max_val), 118.0)

    def test_reset_clears_samples_and_cached_range(self) -> None:
        """Start a new calibration run without retaining previous statistics."""
        observer = self._uint8(max_samples=16)
        observer.collect(torch.tensor([-1.0, 0.0, 1.0]))
        observer.compute_qparams()
        observer.reset()

        self.assertEqual(observer.sampled_value_count, 0)
        self.assertTrue(torch.isinf(observer.min_val))
        self.assertTrue(torch.isinf(observer.max_val))
        self.assertTrue(torch.isinf(observer.clip_min_val))
        self.assertTrue(torch.isinf(observer.clip_max_val))
        self.assertFalse(observer.has_qparams)

    def test_per_channel_configuration_is_rejected(self) -> None:
        """Avoid silently allocating one percentile reservoir per channel."""
        with self.assertRaisesRegex(ValueError, "per-tensor"):
            PercentileObserver(
                name="weight",
                dtype=DType.uint(8),
                qscheme=QScheme.PER_CHANNEL_ASYMM,
                channel_axis=0,
            )


if __name__ == "__main__":
    unittest.main()
