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
from tico.experimental.quantization.custom.dtypes import DType
from tico.experimental.quantization.custom.observers.base import ObserverBase
from tico.experimental.quantization.custom.qscheme import QScheme


class TestObserverBase(unittest.TestCase):
    def test_per_tensor_stats(self):
        obs = ObserverBase(dtype=DType.uint(4))  # 4-bit unsigned

        x1 = torch.tensor([-1.0, 2.0, 3.0])
        obs.collect(x1)

        x2 = torch.tensor([4.0])
        obs.collect(x2)

        self.assertEqual(obs.min_val, -1.0)
        self.assertEqual(obs.max_val, 4.0)

        scale, zp = obs.compute_qparams()
        qmin, qmax = obs.dtype.qmin, obs.dtype.qmax
        expected_scale = (4.0 - (-1.0)) / (qmax - qmin)
        expected_zp = round(qmin - (-1.0) / expected_scale)
        self.assertAlmostEqual(scale.item(), expected_scale, places=6)
        self.assertEqual(zp.item(), expected_zp)

    def test_per_channel_stats(self):
        obs = ObserverBase(
            dtype=DType.int(5),  # 5-bit signed
            qscheme=QScheme.PER_CHANNEL_AFFINE,
            channel_axis=0,
        )
        # Tensor shape (C, N)
        x = torch.tensor([[1.0, 3.0, -2.0], [4.0, -5.0, 0.5]])
        obs.collect(x)
        self.assertTrue(
            torch.equal(torch.as_tensor(obs.min_val), torch.tensor([-2.0, -5.0]))
        )
        self.assertTrue(
            torch.equal(torch.as_tensor(obs.max_val), torch.tensor([3.0, 4.0]))
        )

    def test_per_channel_affine(self):
        torch.manual_seed(0)
        # Toy conv-like activations: shape (C=2, H=2, W=2)
        x = torch.tensor([[[1.0, -3.0], [2.0, 0.5]], [[4.0, -6.0], [3.0, -1.0]]])

        obs = ObserverBase(
            dtype=DType.uint(4),
            qscheme=QScheme.PER_CHANNEL_AFFINE,
            channel_axis=0,
        )
        obs.collect(x)
        obs.enabled = False  # freeze
        obs.compute_qparams()
        y = obs.fake_quant(x)

        # Each channel must have been clipped/quantized independently.
        # A simple invariant: values > channel-max should map to channel-max after dequant.
        ch_max = torch.tensor([2.0, 4.0])  # from data
        y_flat = y.view(2, -1)
        self.assertTrue(torch.all(y_flat[0] <= ch_max[0] + 1e-5))
        self.assertTrue(torch.all(y_flat[1] <= ch_max[1] + 1e-5))
