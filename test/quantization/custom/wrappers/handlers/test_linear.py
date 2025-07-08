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
from tico.experimental.quantization.custom.observers import MinMaxObserver
from tico.experimental.quantization.custom.wrappers.mode import Mode
from tico.experimental.quantization.custom.wrappers.ptq_wrapper import PTQWrapper


class TestLinearHandler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.fp32 = torch.nn.Linear(4, 2)
        act_obs = MinMaxObserver(dtype=DType.uint(8))
        w_obs = MinMaxObserver(dtype=DType.uint(8), channel_axis=0)
        self.wrapper = PTQWrapper(self.fp32, act_obs=act_obs, weight_obs=w_obs)
        self.inp = torch.randn(32, 4)

    def test_calibration_and_quant(self):
        # calibration
        self.wrapper.enable_calibration()
        _ = self.wrapper(self.inp)

        # freeze & quant
        self.wrapper.freeze_qparams()
        self.assertIs(self.wrapper._mode, Mode.QUANT)

        with torch.no_grad():
            q_out: torch.Tensor = self.wrapper(self.inp)
            fp32_out: torch.Tensor = self.fp32(self.inp)

        diff = (q_out - fp32_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.5)
