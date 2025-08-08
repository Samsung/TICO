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

import math
import unittest

import torch
from tico.experimental.quantization.ptq.dtypes import UINT8
from tico.experimental.quantization.ptq.observers.percentile import PercentileObserver
from tico.experimental.quantization.ptq.qscheme import QScheme


class TestPercentileObserver(unittest.TestCase):
    def test_percentile_clip_ignores_outlier(self):
        torch.manual_seed(0)

        obs = PercentileObserver(
            name="dummy",
            percentile=90.0,  # low=5 %, high=95 %
            dtype=UINT8,
            qscheme=QScheme.PER_TENSOR_ASYMM,
        )

        # Main distribution: uniform -10 … +10
        main = torch.linspace(-10.0, 10.0, 1000)
        # Add two extreme outliers
        data = torch.cat([main, torch.tensor([+500.0, -500.0])])

        obs.collect(data)
        self.assertGreaterEqual(obs.max_val.item(), 0.0)
        self.assertLessEqual(obs.max_val.item(), 11.0)  # outlier ignored
        self.assertLessEqual(obs.min_val.item(), 0.0)
        self.assertGreaterEqual(obs.min_val.item(), -11.0)

    def test_reset(self):
        obs = PercentileObserver(name="dummy", percentile=95.0, dtype=UINT8)
        obs.collect(torch.tensor([-1.0, 2.0]))
        obs.reset()
        self.assertEqual(obs.min_val, float("inf"))
        self.assertEqual(obs.max_val, float("-inf"))

    def test_fake_quant_output_range(self):
        obs = PercentileObserver(
            name="dummy",
            percentile=90.0,
            dtype=UINT8,
            qscheme=QScheme.PER_TENSOR_ASYMM,
        )
        x = torch.randn(256) * 3
        obs.collect(x)
        scale, zp = obs.compute_qparams()

        fq = obs.fake_quant(x)

        qmin, qmax = obs.dtype.qmin, obs.dtype.qmax
        lower = scale * (qmin - zp)
        upper = scale * (qmax - zp)
        self.assertTrue(torch.all(fq >= lower - 1e-6))
        self.assertTrue(torch.all(fq <= upper + 1e-6))

    def test_per_channel_collect(self):
        # shape (C=3, N)
        t = torch.tensor(
            [
                [1.0, 2.0, 50.0],  # ch-0 outlier high
                [-3.0, -2.0, -100.0],  # ch-1 outlier low
                [0.5, 0.6, 0.7],  # ch-2 tame
            ]
        )  # shape (3,3)

        pct = 50.0  # keep 25–75th pct   -> median-ish
        obs = PercentileObserver(
            name="dummy",
            percentile=pct,
            dtype=UINT8,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )

        obs.collect(t)

        # expected quantiles per channel
        low = torch.quantile(t, (100 - pct) / 200.0, dim=1)
        high = torch.quantile(t, 1 - (100 - pct) / 200.0, dim=1)

        self.assertTrue(torch.allclose(obs.min_val, low))
        self.assertTrue(torch.allclose(obs.max_val, high))
        self.assertEqual(obs.min_val.shape, torch.Size([3]))  # vector len C

    def test_multiple_collects_accumulate(self):
        obs = PercentileObserver(name="dummy", percentile=90.0)
        x1 = torch.randn(100)
        x2 = torch.randn(100) + 5.0  # shifted distribution

        obs.collect(x1)
        first_min, first_max = obs.min_val.clone(), obs.max_val.clone()

        obs.collect(x2)
        # global mins should be ≤ first_min; max ≥ first_max
        self.assertLessEqual(obs.min_val.item(), first_min.item())
        self.assertGreaterEqual(obs.max_val.item(), first_max.item())
