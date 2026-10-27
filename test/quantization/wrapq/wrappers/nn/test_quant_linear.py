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
import torch.nn.functional as F
from tico.quantization import convert, prepare
from tico.quantization.config import affine, no_quant
from tico.quantization.config.ptq import PTQConfig

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

from test.quantization.quant_spec_helpers import make_affine_ptq_config


class TestQuantLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.fp32 = torch.nn.Linear(4, 2)
        self.x = torch.randn(128, 4)

        self.q_lin = QuantLinear(self.fp32)  # default uint8

    def test_mode_transitions(self):
        self.assertIs(self.q_lin._mode, Mode.NO_QUANT)

        # Calibration (re-collect static weight range right here)
        self.q_lin.enable_calibration()
        _ = self.q_lin(self.x)
        self.assertIs(self.q_lin._mode, Mode.CALIB)

        self.q_lin.freeze_qparams()
        self.assertIs(self.q_lin._mode, Mode.QUANT)

    def test_quantised_output_close(self):
        self.q_lin.enable_calibration()
        _ = self.q_lin(self.x)
        self.q_lin.freeze_qparams()

        with torch.no_grad():
            q_out = self.q_lin(self.x)
            fp_out = F.linear(self.x, self.fp32.weight, self.fp32.bias)

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)

    def test_weight_stats_survive(self):
        self.q_lin.enable_calibration()
        self.q_lin.obs_weight.compute_qparams()
        assert hasattr(self.q_lin.obs_weight, "_cached_scale")
        assert isinstance(self.q_lin.obs_weight, AffineObserverBase)
        pre_scale = self.q_lin.obs_weight._cached_scale.clone()

        # calibration cycle
        self.q_lin.enable_calibration()
        self.q_lin(self.x)
        self.q_lin.freeze_qparams()

        post_scale = self.q_lin.obs_weight._cached_scale
        self.assertTrue(torch.allclose(pre_scale, post_scale))

    def test_dtype_override(self):
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "act_in": {"dtype": DType.uint(4)},
                "act_out": {"dtype": DType.uint(4)},
            },
        )
        qcustom = QuantLinear(self.fp32, qcfg=cfg)
        self.assertEqual(qcustom.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(qcustom.obs_act_out.dtype, DType.uint(4))

    def test_locked_percentile_input_grid_survives_conversion(self):
        """Conversion needs no local samples for a locked external grid."""
        for calibrated_before_lock in (False, True):
            with self.subTest(calibrated_before_lock=calibrated_before_lock):
                linear = torch.nn.Linear(2, 2, bias=False).eval()
                with torch.no_grad():
                    linear.weight.copy_(torch.eye(2))
                model = prepare(
                    linear,
                    PTQConfig(
                        overrides={
                            "act_in": affine(
                                DType.uint(8), observer=PercentileObserver
                            ),
                            "act_out": no_quant(),
                        }
                    ),
                )
                observer = model.get_observer("act_in")
                self.assertIsInstance(observer, PercentileObserver)
                assert isinstance(observer, PercentileObserver)
                if calibrated_before_lock:
                    model(torch.tensor([[-2.0, 4.0]]))
                    observer.compute_qparams()
                sample_count = observer.sampled_value_count
                clip_bounds = (
                    observer.clip_min_val.clone(),
                    observer.clip_max_val.clone(),
                )
                scale = torch.tensor(0.25)
                zero_point = torch.tensor(3, dtype=torch.int)
                observer.load_qparams(scale, zero_point, lock=True)
                model(torch.tensor([[0.0, 100.0]]))

                quantized = convert(model)

                torch.testing.assert_close(
                    observer.compute_qparams(), (scale, zero_point), rtol=0, atol=0
                )
                self.assertFalse(observer.enabled)
                self.assertEqual(observer.sampled_value_count, sample_count)
                torch.testing.assert_close(
                    (observer.clip_min_val, observer.clip_max_val),
                    clip_bounds,
                    rtol=0,
                    atol=0,
                )
                actual = quantized(torch.tensor([[-0.9, 0.3], [1.1, 64.0]]))
                torch.testing.assert_close(
                    actual,
                    torch.tensor([[-0.75, 0.25], [1.0, 63.0]]),
                    rtol=0,
                    atol=0,
                )
