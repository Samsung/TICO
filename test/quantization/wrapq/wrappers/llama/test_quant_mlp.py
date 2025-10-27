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

import pathlib
import tempfile
import unittest
import warnings

import tico

import torch
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.llama.quant_mlp import QuantLlamaMLP


class DummyMLP(torch.nn.Module):
    """Tiny stand-in for HF LlamaMLP (hidden=4, inter=8)."""

    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8)
        self.up_proj = torch.nn.Linear(4, 8)
        self.down_proj = torch.nn.Linear(8, 4)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TestQuantLlamaMLP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.fp32 = DummyMLP()
        self.quant = QuantLlamaMLP(self.fp32)
        self.x = torch.randn(32, 4)

    def test_mode_and_forward(self):
        # calibration
        self.quant.enable_calibration()
        _ = self.quant(self.x)
        self.quant.freeze_qparams()
        self.assertIs(self.quant._mode, Mode.QUANT)

        # forward diff
        with torch.no_grad():
            q = self.quant(self.x)
            f = self.fp32(self.x)
        diff = (q - f).abs().mean().item()
        self.assertLess(diff, 0.7)  # loose bound
        self.assertGreater(diff, 0.0)


class TestSubgraphExport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mlp_int8 = QuantLlamaMLP(DummyMLP()).eval()
        self.x = torch.randn(16, 4)

    def test_calib_quant_export(self):
        # calib
        self.mlp_int8.enable_calibration()
        _ = self.mlp_int8(self.x)
        self.mlp_int8.freeze_qparams()

        self.assertIs(self.mlp_int8._mode, Mode.QUANT)

        # export
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "mlp.circle"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                exported = tico.convert(self.mlp_int8, (self.x[:1],))
            exported.save(path)
            self.assertTrue(path.exists())
