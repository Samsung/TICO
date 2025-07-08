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
from tico.experimental.quantization.custom.dtypes import UINT8
from tico.experimental.quantization.custom.observers.percentile import (
    PercentileObserver,
)
from tico.experimental.quantization.custom.qscheme import QScheme


class TestPercentileObserver(unittest.TestCase):
    def test_percentile_clip_ignores_outlier(self):
        torch.manual_seed(0)

        obs = PercentileObserver(
            percentile=90.0,  # low=5 %, high=95 %
            dtype=UINT8,
            qscheme=QScheme.PER_TENSOR_AFFINE,
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
        obs = PercentileObserver(percentile=95.0, dtype=UINT8)
        obs.collect(torch.tensor([-1.0, 2.0]))
        obs.reset()
        self.assertEqual(obs.min_val, float("inf"))
        self.assertEqual(obs.max_val, float("-inf"))

    def test_fake_quant_output_range(self):
        obs = PercentileObserver(
            percentile=90.0,
            dtype=UINT8,
            qscheme=QScheme.PER_TENSOR_AFFINE,
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
