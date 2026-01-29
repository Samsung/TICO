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
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.wrappers.ops.quant_rmsnorm import QuantRMSNorm


class _DummyRMSNorm(nn.Module):
    """
    Minimal RMSNorm-like module that matches the attributes used by QuantRMSNorm:
      - .weight (learnable affine scale)
      - .variance_epsilon (epsilon for numerical stability)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


def _rmsnorm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    PyTorch reference RMSNorm:
      y = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight
    """
    # Use float math for stability, then cast back to input dtype.
    x_f = x.float()
    denom = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (x_f * denom) * weight.float()
    return y.to(dtype=x.dtype)


class TestQuantRMSNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        self.hidden = 16
        self.fp = _DummyRMSNorm(self.hidden, eps=1e-6)

        # Make weights non-trivial so weight quantization has an effect.
        with torch.no_grad():
            self.fp.weight.copy_(torch.randn_like(self.fp.weight) * 0.5 + 0.2)

        # Use a non-trivial rank so reduction dims are exercised.
        self.x = torch.randn(64, 8, self.hidden)

        self.q_rms = QuantRMSNorm(self.fp)

    def test_mode_transitions(self):
        self.assertIs(self.q_rms._mode, Mode.NO_QUANT)

        self.q_rms.enable_calibration()
        _ = self.q_rms(self.x)
        self.assertIs(self.q_rms._mode, Mode.CALIB)

        self.q_rms.freeze_qparams()
        self.assertIs(self.q_rms._mode, Mode.QUANT)

    def test_no_quant_matches_reference_various_shapes(self):
        """
        In NO_QUANT mode, the wrapper must match the reference RMSNorm exactly
        (up to tight tolerances), across several input shapes.
        """
        cases = [
            (16, (4, 7, 16)),
            (32, (2, 5, 3, 32)),
            (7, (11, 7)),
        ]

        for hidden, inp_shape in cases:
            with self.subTest(hidden=hidden, inp_shape=inp_shape):
                fp = _DummyRMSNorm(hidden, eps=1e-6)
                with torch.no_grad():
                    fp.weight.copy_(torch.randn_like(fp.weight) * 0.5 - 0.1)

                q = QuantRMSNorm(fp)  # stays in NO_QUANT unless calibration is enabled
                x = torch.randn(*inp_shape)

                y = q(x)
                y_ref = _rmsnorm_reference(x, fp.weight, float(fp.variance_epsilon))

                self.assertIs(q._mode, Mode.NO_QUANT)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.allclose(y, y_ref, atol=1e-6, rtol=1e-6))

    def test_quantized_output_close(self):
        """
        After a standard calibrate->freeze cycle, quantized output should:
          - differ from the FP reference (quantization actually applied)
          - stay within a reasonable error bound
        """
        self.q_rms.enable_calibration()
        _ = self.q_rms(self.x)
        self.q_rms.freeze_qparams()

        with torch.no_grad():
            q_out = self.q_rms(self.x)
            fp_out = _rmsnorm_reference(self.x, self.fp.weight, self.q_rms.eps)

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # should not be identical
        self.assertLess(diff, 0.4)  # loose bound; adjust if your quant is tighter

    def test_weight_stats_survive(self):
        """
        Re-running a calibration cycle should not change the fixed weight quant stats.
        (Weights are constant; their computed scales should be stable.)
        """
        self.q_rms.enable_calibration()

        # Force a compute now and snapshot scale
        self.q_rms.obs_weight.compute_qparams()
        self.assertTrue(hasattr(self.q_rms.obs_weight, "_cached_scale"))
        self.assertIsInstance(self.q_rms.obs_weight, AffineObserverBase)
        pre_scale = self.q_rms.obs_weight._cached_scale.clone()

        # Full calibration cycle again
        self.q_rms.enable_calibration()
        _ = self.q_rms(self.x)
        self.q_rms.freeze_qparams()

        post_scale = self.q_rms.obs_weight._cached_scale
        self.assertTrue(torch.allclose(pre_scale, post_scale))

    def test_dtype_override(self):
        """
        PTQConfig overrides should propagate to observers created by QuantRMSNorm.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act_in": {"dtype": DType.uint(4)},
                "act_out": {"dtype": DType.uint(4)},
                "weight": {"dtype": DType.uint(4)},
            },
        )
        fp = _DummyRMSNorm(self.hidden, eps=1e-6)
        q = QuantRMSNorm(fp, qcfg=cfg)

        self.assertIsInstance(q.obs_weight, AffineObserverBase)
        self.assertIsInstance(q.obs_act_in, AffineObserverBase)
        self.assertIsInstance(q.obs_act_out, AffineObserverBase)

        self.assertEqual(q.obs_weight.dtype, DType.uint(4))
        self.assertEqual(q.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q.obs_act_out.dtype, DType.uint(4))
