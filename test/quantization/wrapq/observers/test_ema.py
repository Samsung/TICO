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
from tico.experimental.quantization.wrapq.dtypes import DType
from tico.experimental.quantization.wrapq.observers.ema import EMAObserver
from tico.experimental.quantization.wrapq.qscheme import QScheme


class TestEMAObserver(unittest.TestCase):
    def test_ema_updates(self):
        torch.manual_seed(0)
        obs = EMAObserver(
            name="dummy",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
            momentum=0.5,
        )

        # First batch initialises stats hard
        obs.collect(torch.tensor([-1.0, 2.0, 3.0]))
        self.assertAlmostEqual(obs.min_val.item(), -1.0)
        self.assertAlmostEqual(obs.max_val.item(), 3.0)

        # Second batch updates via EMA
        obs.collect(torch.tensor([-9.0, 9.0]))
        # new_min = 0.5 * (-1) + 0.5 * (-9) = -5
        # new_max = 0.5 * 3   + 0.5 * 9     =  6
        self.assertAlmostEqual(obs.min_val.item(), -5.0)
        self.assertAlmostEqual(obs.max_val.item(), 6.0)

    def test_per_channel_ema(self):
        x1 = torch.tensor([[1.0, -2.0], [3.0, -4.0]])  # shape (C=2, N=2)
        x2 = torch.tensor([[10.0, -10.0], [5.0, -5.0]])
        obs = EMAObserver(
            name="dummy",
            dtype=DType.int(4),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
            momentum=0.5,
        )

        obs.collect(x1)
        self.assertTrue(
            torch.equal(obs.min_val, torch.tensor([-2.0, -4.0])),
            f"obs.min_val: {obs.min_val}",
        )
        self.assertTrue(
            torch.equal(obs.max_val, torch.tensor([1.0, 3.0])),
            f"obs.max_val: {obs.max_val}",
        )

        obs.collect(x2)
        # Manual EMA check
        expected_min = 0.5 * torch.tensor([-2.0, -4.0]) + 0.5 * torch.tensor(
            [-10.0, -5.0]
        )
        expected_max = 0.5 * torch.tensor([1.0, 3.0]) + 0.5 * torch.tensor([10.0, 5.0])
        self.assertTrue(
            torch.allclose(obs.min_val, expected_min),
            f"{obs.min_val} != {expected_min}",
        )
        self.assertTrue(
            torch.allclose(obs.max_val, expected_max),
            f"{obs.max_val} != {expected_max}",
        )

    def test_first_batch(self):
        x = torch.randn(10)
        obs = EMAObserver(name="dummy", momentum=0.9, channel_axis=None)
        obs.collect(x)
        self.assertFalse(torch.isinf(obs.min_val).any())
        self.assertFalse(torch.isinf(obs.max_val).any())
