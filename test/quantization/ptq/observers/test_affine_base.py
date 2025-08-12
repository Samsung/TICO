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
from tico.experimental.quantization.ptq.observers.affine_base import AffineObserverBase
from tico.experimental.quantization.ptq.qscheme import QScheme
from tico.experimental.quantization.ptq.utils import channelwise_minmax


class _MinMaxLikeObserver(AffineObserverBase):
    """
    Minimal affine observer that updates min/max in collect().
    This is enough to exercise AffineObserverBase behavior (qparams, fake-quant).
    """

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        if self.channel_axis is None:
            cur_min, cur_max = x.min(), x.max()
        else:
            cur_min, cur_max = channelwise_minmax(x, self.channel_axis)
        self.min_val = torch.minimum(self.min_val, cur_min)
        self.max_val = torch.maximum(self.max_val, cur_max)


class TestAffineObserverBase(unittest.TestCase):
    def test_per_tensor_asymm_qparams(self):
        # 4-bit unsigned: qmin=0, qmax=15
        obs = _MinMaxLikeObserver(name="pt_asymm", dtype=DType.uint(4))
        obs.collect(torch.tensor([-1.0, 2.0, 3.0]))
        obs.collect(torch.tensor([4.0]))

        self.assertEqual(obs.min_val.item(), -1.0)
        self.assertEqual(obs.max_val.item(), 4.0)

        scale, zp = obs.compute_qparams()
        qmin, qmax = obs.dtype.qmin, obs.dtype.qmax
        expected_scale = (4.0 - (-1.0)) / (qmax - qmin)
        expected_zp = round(qmin - (-1.0) / expected_scale)

        self.assertAlmostEqual(scale.item(), expected_scale, places=6)
        self.assertEqual(zp.item(), int(expected_zp))

    def test_per_tensor_symmetric_qparams(self):
        # 8-bit signed symmetric: zp=0, scale = max(|min|,|max|)/qmax
        obs = _MinMaxLikeObserver(
            name="pt_symm",
            dtype=DType.int(8),
            qscheme=QScheme.PER_TENSOR_SYMM,
        )
        obs.collect(torch.tensor([-3.0, 4.0]))

        scale, zp = obs.compute_qparams()
        qmax = obs.dtype.qmax  # 127
        expected_scale = max(3.0, 4.0) / qmax

        self.assertAlmostEqual(scale.item(), expected_scale, places=6)
        self.assertEqual(zp.item(), 0)

    def test_per_channel_asymm_stats_and_qparams(self):
        # shape (C=2, N=3)
        x = torch.tensor([[1.0, 3.0, -2.0], [4.0, -5.0, 0.5]])

        obs = _MinMaxLikeObserver(
            name="pc_asymm",
            dtype=DType.int(5),  # 5-bit signed
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )
        obs.collect(x)

        self.assertTrue(torch.equal(obs.min_val, torch.tensor([-2.0, -5.0])))
        self.assertTrue(torch.equal(obs.max_val, torch.tensor([3.0, 4.0])))

        scale, zp = obs.compute_qparams()
        qmin, qmax = obs.dtype.qmin, obs.dtype.qmax
        expected_scale = (obs.max_val - obs.min_val) / (qmax - qmin)
        expected_zp = (
            torch.round(qmin - obs.min_val / expected_scale)
            .clamp(qmin, qmax)
            .to(torch.int)
        )

        self.assertTrue(torch.allclose(scale, expected_scale, atol=1e-6))
        self.assertTrue(torch.equal(zp, expected_zp))

    def test_fake_quant_requires_qparams(self):
        obs = _MinMaxLikeObserver(name="requires", dtype=DType.uint(8))
        with self.assertRaises(RuntimeError):
            _ = obs.fake_quant(torch.randn(2, 3))

    def test_load_qparams_and_fake_quant(self):
        obs = _MinMaxLikeObserver(name="load", dtype=DType.uint(8))
        scale = torch.tensor(0.1)
        zp = torch.tensor(5, dtype=torch.int)
        obs.load_qparams(scale, zp, lock=True)

        self.assertFalse(obs.enabled)
        self.assertTrue(obs.has_qparams)

        x = torch.tensor([0.0, 0.05, 0.15])
        y = obs.fake_quant(x)
        q = torch.round(x / scale) + zp
        y_expected = (q - zp) * scale
        self.assertTrue(torch.allclose(y, y_expected, atol=1e-6))

    def test_reset_clears_minmax_and_qparams(self):
        obs = _MinMaxLikeObserver(name="reset", dtype=DType.uint(8))
        obs.collect(torch.tensor([-1.0, 2.0]))
        obs.load_qparams(
            torch.tensor(0.2), torch.tensor(3, dtype=torch.int), lock=False
        )
        self.assertTrue(obs.has_qparams)

        obs.reset()
        self.assertEqual(obs.min_val.item(), float("inf"))
        self.assertEqual(obs.max_val.item(), float("-inf"))
        self.assertFalse(obs.has_qparams)

    def test_per_channel_fake_quant_path(self):
        # Ensure per-channel branch is exercised end-to-end
        x = torch.tensor([[0.05, 0.10, 0.15], [0.02, 0.07, 0.12]])  # (C=2, N=3)

        obs = _MinMaxLikeObserver(
            name="pc_fq",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )
        # Directly load qparams to avoid relying on data stats for this test
        scale = torch.tensor([0.05, 0.02])
        zp = torch.tensor([10, 3], dtype=torch.int)
        obs.load_qparams(scale, zp, lock=True)

        y = obs.fake_quant(x)

        q0 = torch.round(x[0] / scale[0]) + zp[0]
        y0 = (q0 - zp[0]) * scale[0]
        q1 = torch.round(x[1] / scale[1]) + zp[1]
        y1 = (q1 - zp[1]) * scale[1]

        self.assertTrue(torch.allclose(y[0], y0, atol=1e-6))
        self.assertTrue(torch.allclose(y[1], y1, atol=1e-6))

    def test_degenerate_constant_cases(self):
        # Verify constant-tensor handling (0, positive, negative) for asymmetric affine
        def _check_scalar(value: float, dtype: DType):
            obs = _MinMaxLikeObserver(name="scalar", dtype=dtype)
            x = torch.full((4, 5), value)
            obs.collect(x)
            scale, zp = obs.compute_qparams()
            fq = obs.fake_quant(x)
            self.assertTrue(torch.allclose(fq, x, atol=1e-6))

            if value == 0.0:
                self.assertAlmostEqual(scale.item(), 1.0, places=6)
                self.assertEqual(zp.item(), 0)
            elif value > 0:
                self.assertAlmostEqual(scale.item(), value, places=6)
                self.assertEqual(zp.item(), 0)
            else:
                self.assertAlmostEqual(scale.item(), abs(value), places=6)
                self.assertEqual(zp.item(), obs.dtype.qmax)

        _check_scalar(0.5, DType.uint(8))
        _check_scalar(-0.3, DType.uint(8))
        _check_scalar(0.0, DType.uint(8))
