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

from tico.experimental.quantization.ptq.dtypes import DType, UINT8
from tico.experimental.quantization.ptq.observers.base import ObserverBase
from tico.experimental.quantization.ptq.qscheme import QScheme


class _NoopObserver(ObserverBase):
    """
    Minimal concrete subclass for testing ObserverBase behavior.

    - Does not collect any statistics
    - compute_qparams() returns None (valid for non-affine observers like MX)
    - fake_quant() returns input as-is
    """

    def reset(self) -> None:
        # No internal state
        return

    def _update_stats(self, x: torch.Tensor) -> None:
        # No stats to update
        return

    def compute_qparams(self):
        # Non-affine observer may return None; still satisfies the interface
        self._called_compute = True  # test hook
        return None

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _CountingObserver(_NoopObserver):
    """
    Same as _NoopObserver but counts collect() calls to verify 'enabled' gating.
    """

    def reset(self) -> None:
        self.n = 0

    def _update_stats(self, x: torch.Tensor) -> None:
        self.n += 1


class TestObserverBase(unittest.TestCase):
    def test_compute_qparams_contract_allows_none(self):
        # A non-affine observer may return None from compute_qparams()
        obs = _NoopObserver(name="dummy", dtype=DType.uint(8))
        self.assertIsNone(obs.compute_qparams())
        self.assertTrue(getattr(obs, "_called_compute", False))

    def test_collect_respects_enabled_flag(self):
        obs = _CountingObserver(name="ctr", dtype=UINT8)
        obs.reset()
        self.assertEqual(obs.n, 0)

        obs.collect(torch.randn(3))
        self.assertEqual(obs.n, 1)

        # Disable and verify collect() does nothing
        obs.enabled = False
        obs.collect(torch.randn(3))
        self.assertEqual(obs.n, 1)

        # Re-enable and verify it resumes updating
        obs.enabled = True
        obs.collect(torch.randn(3))
        self.assertEqual(obs.n, 2)

    def test_repr_smoke(self):
        obs = _NoopObserver(
            name="repr_test", dtype=DType.int(4), qscheme=QScheme.PER_TENSOR_SYMM
        )
        s = repr(obs)
        self.assertIn("repr_test", s)
        self.assertIn("_NoopObserver", s)
