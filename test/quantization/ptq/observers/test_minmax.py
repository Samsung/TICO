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
from tico.experimental.quantization.ptq.dtypes import DType
from tico.experimental.quantization.ptq.observers.minmax import MinMaxObserver
from tico.experimental.quantization.ptq.qscheme import QScheme


class TestMinMaxObserver(unittest.TestCase):
    def test_per_tensor_minmax(self):
        obs = MinMaxObserver(name="dummy", dtype=DType.uint(4))  # unsigned 4-bit

        x = torch.tensor([-2.0, 1.0, 7.0, -8.0])
        obs.collect(x)

        self.assertEqual(obs.min_val, -8.0)
        self.assertEqual(obs.max_val, 7.0)

        # verify scale/zero-point math
        qmin, qmax = obs.dtype.qmin, obs.dtype.qmax
        scale, zp = obs.compute_qparams()
        expected_scale = (7.0 - (-8.0)) / (qmax - qmin)
        expected_zp = round(qmin - (-8.0) / expected_scale)
        self.assertAlmostEqual(scale.item(), expected_scale, places=6)
        self.assertEqual(zp.item(), expected_zp)

    def test_reset(self):
        obs = MinMaxObserver(name="dummy", dtype=DType.int(8))
        obs.collect(torch.tensor([-1.0, 3.0]))
        obs.reset()
        self.assertEqual(obs.min_val, float("inf"))
        self.assertEqual(obs.max_val, float("-inf"))

    def test_per_channel_minmax(self):
        # Shape (C=3, L=2)
        x = torch.tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 0.5]])

        obs = MinMaxObserver(
            name="dummy",
            dtype=DType.int(5),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )
        obs.collect(x)

        self.assertTrue(torch.equal(obs.min_val, torch.tensor([-2.0, 3.0, -5.0])))
        self.assertTrue(torch.equal(obs.max_val, torch.tensor([1.0, 4.0, 0.5])))


class TestScalarObserver(unittest.TestCase):
    def _check_scalar(self, value: float, dtype: DType):
        obs = MinMaxObserver(name="dummy", dtype=dtype)
        x = torch.full((4, 5), value)

        # collect + freeze
        obs.collect(x)
        scale, zp = obs.compute_qparams()

        # fake-quant should round-trip the constant
        fq = obs.fake_quant(x)
        self.assertTrue(
            torch.allclose(fq, x, atol=1e-6), msg=f"failed round-trip for {value}"
        )

        # sanity on qparams
        if value == 0.0:
            self.assertAlmostEqual(scale.item(), 1.0, places=6)
            self.assertEqual(zp.item(), 0)
        elif value > 0:
            self.assertAlmostEqual(scale.item(), value, places=6)
            self.assertEqual(zp.item(), 0)
        else:  # negative scalar
            self.assertAlmostEqual(scale.item(), abs(value), places=6)
            self.assertEqual(zp.item(), obs.dtype.qmax)

    def test_positive_scalar_uint8(self):
        self._check_scalar(0.5, DType.uint(8))

    def test_negative_scalar_uint8(self):
        self._check_scalar(-0.3, DType.uint(8))

    def test_zero_scalar_uint8(self):
        self._check_scalar(0.0, DType.uint(8))
