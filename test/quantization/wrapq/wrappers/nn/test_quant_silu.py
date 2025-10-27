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
from tico.quantization.wrapq.wrappers.nn.quant_silu import QuantSiLU


class TestQuantSiLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(128, 4) * 3  # wider than N(0,1)
        self.fp_silu = nn.SiLU()
        self.qsilu = QuantSiLU(self.fp_silu)  # default uint8

    def test_mode_transitions(self):
        self.assertIs(self.qsilu._mode, Mode.NO_QUANT)
        self.qsilu.enable_calibration()
        self.assertIs(self.qsilu._mode, Mode.CALIB)
        _ = self.qsilu(self.x)  # collect stats
        self.qsilu.freeze_qparams()
        self.assertIs(self.qsilu._mode, Mode.QUANT)

    def test_quantised_output(self):
        self.qsilu.enable_calibration()
        _ = self.qsilu(self.x)
        self.qsilu.freeze_qparams()

        with torch.no_grad():
            q_out = self.qsilu(self.x)
            fp_out = torch.nn.SiLU()(self.x)

        diff = (q_out - fp_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.3)  # acceptably close

    def test_dtype_override(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "sigmoid": {"dtype": DType.uint(4)},
                "mul": {"dtype": DType.uint(4)},
            },
        )
        qsilu = QuantSiLU(self.fp_silu, qcfg=cfg)
        self.assertEqual(qsilu.sig_obs.dtype, DType.uint(4))
        self.assertEqual(qsilu.mul_obs.dtype, DType.uint(4))
