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

from tico.experimental.quantization.ptq.dtypes import DType
from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.nn.quant_linear import QuantLinear


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
        self.q_lin.weight_obs.compute_qparams()
        assert hasattr(self.q_lin.weight_obs, "_cached_scale")
        pre_scale = self.q_lin.weight_obs._cached_scale.clone()

        # calibration cycle
        self.q_lin.enable_calibration()
        self.q_lin.freeze_qparams()

        post_scale = self.q_lin.weight_obs._cached_scale
        self.assertTrue(torch.allclose(pre_scale, post_scale))

    def test_dtype_override(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act_in": {"dtype": DType.uint(4)},
                "act_out": {"dtype": DType.uint(4)},
            },
        )
        qcustom = QuantLinear(self.fp32, qcfg=cfg)
        self.assertEqual(qcustom.act_in_obs.dtype, DType.uint(4))
        self.assertEqual(qcustom.act_out_obs.dtype, DType.uint(4))
