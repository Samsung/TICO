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
from typing import Optional

import torch
from tico.experimental.quantization.custom.dtypes import DType
from tico.experimental.quantization.custom.observers import (
    MinMaxObserver,
    ObserverBase,
    PercentileObserver,
)
from tico.experimental.quantization.custom.qscheme import QScheme
from tico.experimental.quantization.custom.wrappers.mode import Mode
from tico.experimental.quantization.custom.wrappers.ptq_wrapper import PTQWrapper
from torch.utils.data import DataLoader, TensorDataset

from test.modules.op.linear import SimpleLinear


class TestPTQWrapper(unittest.TestCase):
    def setUp(self, act_obs: Optional[ObserverBase] = None):
        torch.manual_seed(42)
        self.fp32 = torch.nn.Linear(4, 2)
        self.input = torch.randn(32, 4)

        # Activation observers (two algorithms)
        self.act_obs = (
            MinMaxObserver(dtype=DType.uint(8)) if act_obs is None else act_obs
        )

        # Per-channel weight observer (axis 0 → out_features)
        self.weight_obs = MinMaxObserver(
            dtype=DType.uint(8),
            qscheme=QScheme.PER_CHANNEL_AFFINE,
            channel_axis=0,
        )

        self.wrapper = PTQWrapper(
            module=self.fp32,
            act_obs=self.act_obs,
            weight_obs=self.weight_obs,
        )

    # ────────────────────────────────────────────────────────────────
    #  Mode sanity
    # ────────────────────────────────────────────────────────────────
    def test_default_mode_is_no_quant(self):
        self.assertIs(self.wrapper._mode, Mode.NO_QUANT)

    def test_mode_transitions(self):
        self.wrapper.enable_calibration()
        self.assertIs(self.wrapper._mode, Mode.CALIB)

        self.wrapper.freeze_qparams()
        self.assertIs(self.wrapper._mode, Mode.QUANT)

    # ────────────────────────────────────────────────────────────────
    #  Activation algorithm swap
    # ────────────────────────────────────────────────────────────────
    def test_switch_activation_observer(self):
        # Re-build and calibrate with minmax observer
        self.wrapper.enable_calibration()
        _ = self.wrapper(self.input)
        self.wrapper.freeze_qparams()
        out_mm: torch.Tensor = self.wrapper(self.input)

        # Calibrate with percentile observer
        percentile_obs = PercentileObserver(percentile=99.0, dtype=DType.uint(8))
        self.setUp(percentile_obs)  # reset
        self.wrapper.enable_calibration()
        _ = self.wrapper(self.input)
        self.wrapper.freeze_qparams()
        out_pct: torch.Tensor = self.wrapper(self.input)

        diff = (out_mm - out_pct).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.5)

    # ────────────────────────────────────────────────────────────────
    #  Weight fake-quant correctness (per-channel)
    # ────────────────────────────────────────────────────────────────
    def test_weight_fake_quant_channelwise(self):
        w_fp32 = self.fp32.weight.data.clone()  # [out, in]

        assert self.wrapper.weight_obs is not None
        self.wrapper.weight_obs.collect(w_fp32)  # stats
        self.wrapper.weight_obs.compute_qparams()  # cache
        fq_weight = self.wrapper.weight_obs.fake_quant(w_fp32)

        # Manual per-channel quant → dequant for reference
        scale, zp = self.wrapper.weight_obs.compute_qparams()
        ref = torch.empty_like(w_fp32)
        for c in range(w_fp32.size(0)):
            q = torch.round(w_fp32[c] / scale[c]) + zp[c]
            q = q.clamp(
                self.wrapper.weight_obs.dtype.qmin, self.wrapper.weight_obs.dtype.qmax
            )
            ref[c] = scale[c] * (q - zp[c])

        # Assertions
        self.assertTrue(torch.allclose(fq_weight, ref, atol=1e-6))
        self.assertFalse(torch.allclose(fq_weight, w_fp32, atol=1e-6))


class TestPTQSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        # Simple model
        self.model = SimpleLinear()
        self.model.eval()

        # Calibration dataset
        data = torch.randn(128, 3) * 2
        self.calib_loader = DataLoader(TensorDataset(data), batch_size=32)

        # Activation observers
        self.act_obs = MinMaxObserver(dtype=DType.uint(8))

        # Per-channel weight observer
        self.weight_obs = MinMaxObserver(
            dtype=DType.uint(8),
            qscheme=QScheme.PER_CHANNEL_AFFINE,
            channel_axis=0,
        )

        # Wrap
        self.model.linear = PTQWrapper(
            module=self.model.linear,
            act_obs=self.act_obs,
            weight_obs=self.weight_obs,
        )  # type: ignore[assignment]

    def test_smoke_forward_quantized(self):
        assert isinstance(self.model.linear, PTQWrapper)
        # Calibration with minmax
        self.model.linear.enable_calibration()
        for (x,) in self.calib_loader:
            _ = self.model(x)
        self.model.linear.freeze_qparams()
        self.assertIs(self.model.linear._mode, Mode.QUANT)

        # Forward
        inp = self.model.get_example_inputs()
        with torch.no_grad():
            fp32_out: torch.Tensor = self.model.linear.module(*inp)  # original module
            q_out: torch.Tensor = self.model(*inp)  # quant-sim

        diff = (fp32_out - q_out).abs().mean().item()

        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.5)
        self.assertEqual(fp32_out.shape, q_out.shape)
