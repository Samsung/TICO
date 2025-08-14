# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import unittest

import torch

from tico.experimental.quantization.ptq.dtypes import UINT8
from tico.experimental.quantization.ptq.observers.histogram import HistogramObserver


class TestHistogramObserver(unittest.TestCase):
    """Sanity-checks for the patched HistogramObserver."""

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_obs(**kwargs):
        """Factory for a fresh observer with optional ctor kwargs."""
        return HistogramObserver(name="hist", dtype=UINT8, **kwargs)

    # ------------------------------------------------------------------ #
    # 1. Histogram accumulation
    # ------------------------------------------------------------------ #
    def test_histogram_accumulation(self):
        obs = self._make_obs(bins=256)
        x = torch.randn(4, 16)
        obs.collect(x)

        self.assertAlmostEqual(
            obs.hist.sum().item(),
            x.numel(),
            delta=1e-3,
            msg="Histogram counts must equal number of collected samples",
        )
        self.assertFalse(torch.isinf(obs.min_val))
        self.assertFalse(torch.isinf(obs.max_val))

    # ------------------------------------------------------------------ #
    # 2. Q-param computation
    # ------------------------------------------------------------------ #
    def test_compute_qparams_sets_scale_and_zp(self):
        obs = self._make_obs(bins=512)
        obs.collect(torch.randn(8, 32))
        scale, zp = obs.compute_qparams()

        self.assertTrue(obs.has_qparams)
        self.assertGreater(scale.item(), 0.0)
        self.assertIsInstance(zp.item(), int)

    # ------------------------------------------------------------------ #
    # 3. fake_quant clamps properly
    # ------------------------------------------------------------------ #
    def test_fake_quant_within_clip(self):
        obs = self._make_obs(bins=512)
        x = torch.randn(8, 32)
        obs.collect(x)
        obs.compute_qparams()
        y = obs.fake_quant(x)

        # One LSB slack:
        # fake_quantize rounds to the nearest integer level, so the
        #  de-quantized value can deviate from the exact clip_min/clip_max
        # by <= 1 quantum step.
        lsb = obs._cached_scale.abs().max().item()
        self.assertTrue(torch.all(y >= obs.min_val - lsb - 1e-6))
        self.assertTrue(torch.all(y <= obs.max_val + lsb + 1e-6))

    # ------------------------------------------------------------------ #
    # 4. reset clears state
    # ------------------------------------------------------------------ #
    def test_reset_clears_histogram(self):
        obs = self._make_obs(bins=128)
        obs.collect(torch.randn(4, 16))
        obs.reset()

        self.assertEqual(obs.hist.sum().item(), 0.0)
        self.assertTrue(torch.isinf(obs.min_val))
        self.assertTrue(torch.isinf(obs.max_val))

    # ------------------------------------------------------------------ #
    # 5. bins < num_qbins must not raise
    # ------------------------------------------------------------------ #
    def test_small_bins_no_stride_zero(self):
        """bins (64) < num_qbins (256) handled gracefully."""
        obs = self._make_obs(bins=64)  # intentionally tiny
        obs.collect(torch.randn(4, 32))
        # Should *not* raise (stride=0 or length mismatch)
        obs.compute_qparams()

    # ------------------------------------------------------------------ #
    # 6. bins not divisible by num_qbins must not raise
    # ------------------------------------------------------------------ #
    def test_non_multiple_bins(self):
        """bins (1000) not divisible by 256 handled gracefully."""
        obs = self._make_obs(bins=1000)
        obs.collect(torch.randn(4, 32))
        obs.compute_qparams()  # should succeed without exception
