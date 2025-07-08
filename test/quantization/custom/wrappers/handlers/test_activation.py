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


class TestActivationHandler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        # Weight-less activation module
        self.act_fp32 = torch.nn.SiLU()

        # Single activation observer
        self.act_obs = MinMaxObserver(dtype=DType.uint(8))

        # Wrap with PTQWrapper (weight_obs=None)
        self.wrapper = PTQWrapper(
            module=self.act_fp32,
            act_obs=self.act_obs,
            weight_obs=None,  # ActivationHandler selected
        )

        self.input = torch.randn(64, 4)

    # ─────────────────────────────────────────────────────────────
    #  Mode sanity
    # ─────────────────────────────────────────────────────────────
    def test_mode_transitions(self):
        self.assertIs(self.wrapper._mode, Mode.NO_QUANT)

        self.wrapper.enable_calibration()
        self.assertIs(self.wrapper._mode, Mode.CALIB)

        # run a single pass to gather stats
        _ = self.wrapper(self.input)

        self.wrapper.freeze_qparams()
        self.assertIs(self.wrapper._mode, Mode.QUANT)

    # ─────────────────────────────────────────────────────────────
    #  Fake-quant vs FP32
    # ─────────────────────────────────────────────────────────────
    def test_quantized_output_reasonable(self):
        # Calibration
        self.wrapper.enable_calibration()
        _ = self.wrapper(self.input)
        self.wrapper.freeze_qparams()

        with torch.no_grad():
            q_out: torch.Tensor = self.wrapper(self.input)  # fake-quant
            fp_out: torch.Tensor = self.act_fp32(self.input)  # reference FP32

        diff = (q_out - fp_out).abs().mean().item()

        self.assertGreater(diff, 0.0)  # Should differ – quantization happened
        self.assertLess(diff, 0.3)  # But not wildly off
        self.assertEqual(q_out.shape, fp_out.shape)
