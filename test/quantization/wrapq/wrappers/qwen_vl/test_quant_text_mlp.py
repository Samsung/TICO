# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

import math
import pathlib
import tempfile
import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

import warnings

import tico

import torch
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_mlp import QuantQwen3VLTextMLP

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.quantization.wrapq.wrappers.qwen_vl.text_mlp_fixture import (
    make_text_mlp_fixture,
)

skip_msg = "required transformers not installed — skipping Qwen3VLTextMLP tests"


class DummyTextMLP(torch.nn.Module):
    """Tiny stand-in for Qwen3VLTextMLP (hidden=4, inter=8)."""

    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.intermediate_size = 8
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLTextMLP(unittest.TestCase):
    mlp_fp: "Qwen3VLTextMLP"
    hidden_size: int
    intermediate_size: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

        cfg = Qwen3VLTextConfig(
            hidden_size=4,
            intermediate_size=8,
            hidden_act="silu",
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            cls.mlp_fp = Qwen3VLTextMLP(cfg).eval()
        cls.hidden_size = cfg.hidden_size
        cls.intermediate_size = cfg.intermediate_size

    def test_mode_transitions(self):
        qmlp = QuantQwen3VLTextMLP(self.mlp_fp)
        self.assertIs(qmlp._mode, Mode.NO_QUANT)

        qmlp.enable_calibration()
        self.assertIs(qmlp._mode, Mode.CALIB)

        x = torch.randn(2, 5, self.hidden_size)
        _ = qmlp(x)

        qmlp.freeze_qparams()
        self.assertIs(qmlp._mode, Mode.QUANT)

    def test_forward_diff(self):
        fp_mlp, calibration, x = make_text_mlp_fixture()
        qmlp = QuantQwen3VLTextMLP(
            fp_mlp, qcfg=make_affine_ptq_config(dtype=DType.uint(8))
        ).eval()
        qmlp.enable_calibration()
        with torch.no_grad():
            qmlp(calibration)
        qmlp.freeze_qparams()
        self.assertIs(qmlp._mode, Mode.QUANT)

        # Independent scalar affine arithmetic: weights 0, 1 and 1/2 are exact
        # endpoints of their per-channel uint8 grids. Input/projection grids are
        # [0, 1]; sigmoid, SiLU, gated and output grids are [0, sigmoid(1)].
        # Reapplying a grid is idempotent; down_proj averages identical copies.
        maximum = 1 / (1 + math.exp(-1))

        def quantize(value, upper):
            scale = upper / 255
            return min(255, max(0, round(value / scale))) * scale

        expected_values = []
        for value in x.flatten().tolist():
            x_q = quantize(value, 1.0)
            sigmoid_q = quantize(1 / (1 + math.exp(-x_q)), maximum)
            silu_q = quantize(x_q * sigmoid_q, maximum)
            expected_values.append(quantize(silu_q * x_q, maximum))
        golden = torch.tensor(expected_values).reshape(x.shape)
        with torch.no_grad():
            q_out = qmlp(x)

        # With identical grid codes the output is n * sigmoid(1) / 255. Budget
        # float32 rounding of the sigmoid endpoint, scale division, dequantization
        # multiply and golden cast: rtol covers 8 float32 eps, and atol covers
        # 3 ULPs at the maximum output (<1). Inputs avoid half-grid boundaries.
        torch.testing.assert_close(q_out, golden, rtol=1e-6, atol=2e-7)
        grid_values = q_out / (maximum / 255)
        torch.testing.assert_close(grid_values, grid_values.round(), rtol=0, atol=2e-5)

    def test_per_projection_override(self):
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "gate_proj": {
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                }
            },
        )
        qmlp = QuantQwen3VLTextMLP(self.mlp_fp, qcfg=cfg)
        q_lin = qmlp.gate_proj.wrapped

        self.assertIsInstance(q_lin, QuantLinear)
        self.assertEqual(q_lin.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_lin.obs_act_out.dtype, DType.uint(4))


class TestSubgraphExport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mlp_int8 = QuantQwen3VLTextMLP(DummyTextMLP()).eval()
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
