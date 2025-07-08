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
from tico.experimental.quantization.custom.dtypes import DType
from tico.experimental.quantization.custom.wrappers.mode import Mode
from tico.experimental.quantization.custom.wrappers.quant_silu import QuantSiLU


class TestQuantSiLU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mod = QuantSiLU(dtype=DType.uint(8))
        self.x = torch.randn(128, 4) * 3

    def test_mode_flow(self):
        self.assertIs(self.mod._mode, Mode.NO_QUANT)
        self.mod.enable_calibration()
        self.assertIs(self.mod._mode, Mode.CALIB)
        _ = self.mod(self.x)  # collect stats
        self.mod.freeze_qparams()
        self.assertIs(self.mod._mode, Mode.QUANT)

    def test_quantised_output(self):
        self.mod.enable_calibration()
        _ = self.mod(self.x)
        self.mod.freeze_qparams()
        with torch.no_grad():
            q_out: torch.Tensor = self.mod(self.x)
            fp_out: torch.Tensor = torch.nn.SiLU()(self.x)
        diff = (q_out - fp_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.3)
