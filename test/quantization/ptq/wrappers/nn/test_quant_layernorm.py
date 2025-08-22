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
import torch.nn.functional as F
from tico.experimental.quantization.ptq.dtypes import DType
from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.observers.affine_base import AffineObserverBase
from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.nn.quant_layernorm import (
    QuantLayerNorm,
)


class TestQuantLayerNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.shape = (16,)  # normalized_shape
        self.fp32 = nn.LayerNorm(list(self.shape), eps=1e-5, elementwise_affine=True)
        # Use a non-trivial input rank so reduction dims are exercised
        self.x = torch.randn(64, 8, *self.shape)

        self.q_ln = QuantLayerNorm(self.fp32)  # default config (e.g., uint8)

    def test_mode_transitions(self):
        self.assertIs(self.q_ln._mode, Mode.NO_QUANT)

        self.q_ln.enable_calibration()
        _ = self.q_ln(self.x)
        self.assertIs(self.q_ln._mode, Mode.CALIB)

        self.q_ln.freeze_qparams()
        self.assertIs(self.q_ln._mode, Mode.QUANT)

    def test_quantised_output_close(self):
        """
        After a standard calibrate->freeze cycle, quantized output should:
          - differ from exact FP (quantization actually applied)
          - stay within a reasonable error bound
        """
        self.q_ln.enable_calibration()
        _ = self.q_ln(self.x)
        self.q_ln.freeze_qparams()

        with torch.no_grad():
            q_out = self.q_ln(self.x)
            fp_out = F.layer_norm(
                self.x,
                self.fp32.normalized_shape,
                self.fp32.weight,
                self.fp32.bias,
                self.fp32.eps,
            )

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # should not be identical
        self.assertLess(diff, 0.4)  # loose bound

    def test_weight_stats_survive(self):
        """
        Re-running a calibration cycle should not change fixed affine param stats.
        (γ/β are constant; their computed scales should be stable.)
        """
        # Pre-compute once
        self.q_ln.enable_calibration()
        # Collect the fixed params immediately (QuantLayerNorm does this in enable_calibration)
        if self.q_ln.weight_obs is None:
            self.skipTest("No weight observer (elementwise_affine=False)")

        # Force a compute now and snapshot scale
        self.q_ln.weight_obs.compute_qparams()
        self.assertTrue(hasattr(self.q_ln.weight_obs, "_cached_scale"))
        assert isinstance(self.q_ln.weight_obs, AffineObserverBase)
        pre_scale = self.q_ln.weight_obs._cached_scale.clone()

        # Full calibration cycle again
        self.q_ln.enable_calibration()
        _ = self.q_ln(self.x)
        self.q_ln.freeze_qparams()

        post_scale = self.q_ln.weight_obs._cached_scale
        self.assertTrue(torch.allclose(pre_scale, post_scale))

    def test_dtype_override(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act_in": {"dtype": DType.uint(4)},
                "act_out": {"dtype": DType.uint(4)},
                "square": {"dtype": DType.uint(4)},
                "affine_mul": {"dtype": DType.uint(4)},
                "affine_add": {"dtype": DType.uint(4)},
            },
        )
        qcustom = QuantLayerNorm(
            nn.LayerNorm(list(self.shape), elementwise_affine=True), qcfg=cfg
        )
        assert isinstance(qcustom.square_obs, AffineObserverBase)
        assert isinstance(qcustom.affine_mul_obs, AffineObserverBase)
        assert isinstance(qcustom.affine_add_obs, AffineObserverBase)
        self.assertEqual(qcustom.square_obs.dtype, DType.uint(4))
        self.assertEqual(qcustom.affine_mul_obs.dtype, DType.uint(4))
        self.assertEqual(qcustom.affine_add_obs.dtype, DType.uint(4))

    def test_no_quant_matches_reference_various_shapes(self):
        """
        In NO_QUANT mode, the wrapper must match PyTorch LayerNorm exactly,
        for various (normalized_shape, input_shape) combinations.
        """
        cases = [
            ((12,), (4, 7, 12)),
            ((6, 8), (2, 5, 6, 8)),
            ((5,), (7, 5)),
        ]
        for norm_shape, inp_shape in cases:
            with self.subTest(norm_shape=norm_shape, inp_shape=inp_shape):
                fp = nn.LayerNorm(list(norm_shape), eps=1e-5, elementwise_affine=True)
                qln = QuantLayerNorm(fp)  # default NO_QUANT

                x = torch.randn(*inp_shape)
                y_ref = F.layer_norm(
                    x, list(fp.normalized_shape), fp.weight, fp.bias, fp.eps
                )
                y = qln(x)

                self.assertEqual(qln._mode, Mode.NO_QUANT)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.allclose(y, y_ref, atol=1e-6, rtol=1e-6))

    def test_no_affine_path(self):
        ln_noaff = nn.LayerNorm(list(self.shape), eps=1e-5, elementwise_affine=False)
        qln = QuantLayerNorm(ln_noaff)

        # Calibrate → Quant
        qln.enable_calibration()
        _ = qln(self.x)
        qln.freeze_qparams()

        with torch.no_grad():
            q_out = qln(self.x)
            fp_out = F.layer_norm(
                self.x, ln_noaff.normalized_shape, None, None, ln_noaff.eps
            )

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
