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

import importlib.util
import pathlib
import tempfile
import unittest
import warnings

import tico

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_mlp import QuantQwen3VLTextMLP


trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping Qwen3VLTextMLP tests"


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


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantQwen3VLTextMLP(unittest.TestCase):
    mlp_fp: torch.nn.Module
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

        cls.mlp_fp = Qwen3VLTextMLP(cfg)
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
        qmlp = QuantQwen3VLTextMLP(self.mlp_fp)
        qmlp.enable_calibration()
        for _ in range(4):
            inp = torch.randn(2, 6, self.hidden_size)
            _ = qmlp(inp)
        qmlp.freeze_qparams()

        x = torch.randn(2, 6, self.hidden_size)
        with torch.no_grad():
            q_out = qmlp(x)
            fp_out = self.mlp_fp(x)

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_per_projection_override(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
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
