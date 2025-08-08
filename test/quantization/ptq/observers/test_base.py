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
from tico.experimental.quantization.ptq.observers.base import ObserverBase
from tico.experimental.quantization.ptq.qscheme import QScheme


class _NoopObserver(ObserverBase):
    """
    Minimal concrete subclass for testing ObserverBase behavior.
    Does not collect statistics; tests rely on load_qparams().
    """

    def collect(self, x: torch.Tensor) -> None:
        return  # no-op


class TestObserverBase(unittest.TestCase):
    def test_fake_quant_requires_qparams(self):
        obs = _NoopObserver(name="dummy", dtype=DType.uint(8))
        x = torch.randn(4)
        with self.assertRaises(RuntimeError):
            _ = obs.fake_quant(x)

    def test_load_qparams_locks_and_is_used(self):
        obs = _NoopObserver(name="dummy", dtype=DType.uint(8))  # qmin=0, qmax=255
        self.assertTrue(obs.enabled)

        scale = torch.tensor(0.1)
        zp = torch.tensor(5, dtype=torch.int)
        obs.load_qparams(scale, zp, lock=True)

        # Locked after load
        self.assertFalse(obs.enabled)
        self.assertTrue(obs.has_qparams)

        # fake-quant uses injected qparams (avoid clamping region)
        x = torch.tensor([0.0, 0.05, 0.15])
        y = obs.fake_quant(x)

        q = torch.round(x / scale) + zp
        y_expected = (q - zp) * scale
        self.assertTrue(torch.allclose(y, y_expected, atol=1e-6))

    def test_reset_clears_minmax_and_cached_qparams(self):
        obs = _NoopObserver(name="dummy", dtype=DType.uint(8))
        obs.load_qparams(
            torch.tensor(0.2), torch.tensor(3, dtype=torch.int), lock=False
        )
        self.assertTrue(obs.has_qparams)

        obs.reset()
        # min/max sentinels restored
        self.assertEqual(obs.min_val.item(), float("inf"))
        self.assertEqual(obs.max_val.item(), float("-inf"))
        # cached qparams removed
        self.assertFalse(obs.has_qparams)

    def test_per_channel_fake_quant_path(self):
        # Ensure per-channel branch in base.fake_quant() behaves with provided axis/params.
        obs = _NoopObserver(
            name="dummy",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )
        x = torch.tensor([[0.05, 0.10, 0.15], [0.02, 0.07, 0.12]])  # shape (C=2, N=3)

        scale = torch.tensor([0.05, 0.02])  # per-channel scales (C,)
        zp = torch.tensor([10, 3], dtype=torch.int)  # per-channel zero-points (C,)
        obs.load_qparams(scale, zp, lock=True)

        y = obs.fake_quant(x)

        # Reproduce expected dequant per channel (no clamping region)
        q0 = torch.round(x[0] / scale[0]) + zp[0]
        y0 = (q0 - zp[0]) * scale[0]
        q1 = torch.round(x[1] / scale[1]) + zp[1]
        y1 = (q1 - zp[1]) * scale[1]

        self.assertTrue(torch.allclose(y[0], y0, atol=1e-6))
        self.assertTrue(torch.allclose(y[1], y1, atol=1e-6))
