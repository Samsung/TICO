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

import unittest

import torch
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mse_grad_octav import MSEGradOCTAVObserver
from tico.quantization.wrapq.qscheme import QScheme


class TestMSEGradOCTAVDegeneracyGuard(unittest.TestCase):
    """Degeneracy guard: gradient-weighted OCTAV must not collapse ranges.

    When the task-loss gradient is ~0 at the values being clipped (e.g. a
    saturated causal mask ``{0, -120}``, where ``exp(-120) ~ 0`` makes
    ``|dL/dx| ~ 0`` at masked positions), the gradient-weighted solver
    treats clipping those values as free and collapses the quantization
    range.  The observer must detect such collapses and fall back to the
    observed min/max range.
    """

    @staticmethod
    def _saturated_mask(seq_len: int = 32, fill: float = -120.0) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), fill), diagonal=1)

    @staticmethod
    def _mask_gradient(
        x: torch.Tensor, unmasked: float = 1e-2, masked: float = 0.0
    ) -> torch.Tensor:
        return torch.where(x == 0, torch.tensor(unmasked), torch.tensor(masked))

    def _run(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        qscheme: QScheme,
        max_merge: bool,
        **kwargs,
    ) -> MSEGradOCTAVObserver:
        obs = MSEGradOCTAVObserver(
            name="attn_mask",
            dtype=DType.int(16),
            qscheme=qscheme,
            max_merge=max_merge,
            **kwargs,
        )
        if max_merge:
            obs.collect(x, grad=grad)
        else:
            obs.collect(x)  # stage 1: plain global min/max grid
            obs.prepare_stage2()
            obs.collect(x, grad=grad)  # stage 2: gradient-weighted histogram
        obs.compute_qparams()
        return obs

    def test_saturated_mask_range_preserved_all_modes(self):
        """The {0, -120} causal mask must keep -120 in range in every mode."""
        x = self._saturated_mask()
        g = self._mask_gradient(x)
        for qs in (QScheme.PER_TENSOR_SYMM, QScheme.PER_TENSOR_ASYMM):
            for mm in (True, False):
                with self.subTest(qscheme=qs, max_merge=mm):
                    obs = self._run(x, g, qs, mm)
                    self.assertLessEqual(obs.min_val.item(), -119.0)
                    fq = obs.fake_quant(x)
                    # Masked positions must stay saturated (exp(-100) ~ 0).
                    self.assertTrue(torch.all(fq[x == -120.0] <= -100.0))

    @staticmethod
    def _collapsed_range_input() -> tuple[torch.Tensor, torch.Tensor]:
        """Bulk with normal gradient + far outliers with ~0 gradient mass.

        The gradient-weighted solver clips the outliers for free, collapsing
        the solution range to ~0 regardless of the observed range.
        """
        gen = torch.Generator().manual_seed(0)
        bulk = torch.randn(998, generator=gen) * 1e-3
        x = torch.cat([bulk, torch.tensor([100.0, -100.0])])
        grad = torch.cat([torch.full((998,), 1.0), torch.full((2,), 1e-16)])
        return x, grad

    def test_two_stage_guard_restores_collapsed_range(self):
        """Two-stage: a valid-but-collapsed solution falls back to observed range."""
        x, grad = self._collapsed_range_input()

        # Guard disabled: documents the failure mode (range collapses to ~0).
        obs = self._run(
            x, grad, QScheme.PER_TENSOR_SYMM, False, degenerate_range_ratio=0.0
        )
        self.assertLess((obs.max_val - obs.min_val).item(), 1.0)

        # Guard enabled (default): the observed range is restored.
        obs = self._run(x, grad, QScheme.PER_TENSOR_SYMM, False)
        self.assertAlmostEqual(obs.min_val.item(), -100.0, places=3)
        self.assertAlmostEqual(obs.max_val.item(), 100.0, places=3)
        fq = obs.fake_quant(x)
        self.assertTrue(
            torch.allclose(fq[998:], torch.tensor([100.0, -100.0]), atol=0.05)
        )

    def test_max_merge_guard_restores_collapsed_range(self):
        """max_merge: a collapsed per-batch solution falls back to batch range."""
        x, grad = self._collapsed_range_input()

        # Guard disabled: documents the failure mode (range collapses to ~0).
        obs = self._run(
            x, grad, QScheme.PER_TENSOR_SYMM, True, degenerate_range_ratio=0.0
        )
        self.assertLess((obs.max_val - obs.min_val).item(), 1.0)

        # Guard enabled (default): the batch's observed range is used.
        obs = self._run(x, grad, QScheme.PER_TENSOR_SYMM, True)
        self.assertAlmostEqual(obs.min_val.item(), -100.0, places=3)
        self.assertAlmostEqual(obs.max_val.item(), 100.0, places=3)

    def test_guard_does_not_trigger_on_normal_data(self):
        """Uniform gradients: OCTAV may clip outliers but must not fall back."""
        gen = torch.Generator().manual_seed(1)
        bulk = torch.randn(995, generator=gen) * 0.5
        x = torch.cat([bulk, torch.tensor([6.0, -7.0, 8.0, -9.0, 10.0])])
        grad = torch.ones_like(x)
        raw_range = x.max().item() - x.min().item()
        for mm in (True, False):
            with self.subTest(max_merge=mm):
                obs = self._run(x, grad, QScheme.PER_TENSOR_SYMM, mm)
                self.assertGreater(
                    (obs.max_val - obs.min_val).item(), 0.01 * raw_range
                )


if __name__ == "__main__":
    unittest.main()
