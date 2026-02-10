# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import unittest

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_gelutanh import QuantGELUTanh

trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping TestQuantGELUTanh tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantGELUTanh(unittest.TestCase):
    gelutanh: type

    @classmethod
    def setUpClass(cls):
        import transformers

        cls.gelutanh = transformers.activations.GELUTanh

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(128, 4) * 3  # wider than N(0,1) for better tanh coverage
        self.fp_gelu_tanh = self.gelutanh()
        self.qgelu_tanh = QuantGELUTanh(self.fp_gelu_tanh)  # default uint8

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        self.assertIs(self.qgelu_tanh._mode, Mode.NO_QUANT)
        self.qgelu_tanh.enable_calibration()
        self.assertIs(self.qgelu_tanh._mode, Mode.CALIB)
        _ = self.qgelu_tanh(self.x)  # collect stats
        self.qgelu_tanh.freeze_qparams()
        self.assertIs(self.qgelu_tanh._mode, Mode.QUANT)

    def test_quantised_output(self):
        """
        Test that quantized output is acceptably close to FP32 reference.
        After calibration and freeze, quantized output should:
        - Differ from FP reference (quantization actually applied)
        - Stay within reasonable error bounds
        """
        self.qgelu_tanh.enable_calibration()
        _ = self.qgelu_tanh(self.x)
        self.qgelu_tanh.freeze_qparams()

        with torch.no_grad():
            q_out = self.qgelu_tanh(self.x)
            fp_out = self.gelutanh()(self.x)

        diff = (q_out - fp_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical (quantization applied)
        self.assertLess(diff, 0.3)  # acceptably close (same tolerance as SiLU)

    def test_dtype_override(self):
        """
        PTQConfig overrides should propagate to observers created by QuantGELUTanh.
        Test that different dtypes can be applied to intermediate activations.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "tanh": {"dtype": DType.uint(4)},
                "mul": {"dtype": DType.uint(4)},
            },
        )
        qgelu_custom = QuantGELUTanh(self.fp_gelu_tanh, qcfg=cfg)

        # Check that overrides were applied
        self.assertEqual(qgelu_custom.obs_tanh.dtype, DType.uint(4))
        self.assertEqual(qgelu_custom.obs_mul.dtype, DType.uint(4))

    def test_activation_stats_collected(self):
        """
        Test that activation statistics are properly collected during calibration.
        All three observers (act_in, tanh, mul) should collect statistics.
        """
        self.qgelu_tanh.enable_calibration()

        # Run forward pass to collect stats
        _ = self.qgelu_tanh(self.x)

        # Check that activation observers have collected stats
        self.assertTrue(
            self.qgelu_tanh.obs_act_in.has_qparams
            or self.qgelu_tanh.obs_act_in.min_val.numel() > 0
        )
        self.assertTrue(
            self.qgelu_tanh.obs_tanh.has_qparams
            or self.qgelu_tanh.obs_tanh.min_val.numel() > 0
        )
        self.assertTrue(
            self.qgelu_tanh.obs_mul.has_qparams
            or self.qgelu_tanh.obs_mul.min_val.numel() > 0
        )

        # Freeze and check qparams exist
        self.qgelu_tanh.freeze_qparams()
        self.assertTrue(self.qgelu_tanh.obs_act_in.has_qparams)
        self.assertTrue(self.qgelu_tanh.obs_tanh.has_qparams)
        self.assertTrue(self.qgelu_tanh.obs_mul.has_qparams)

    def test_no_quant_matches_reference(self):
        """
        In NO_QUANT mode, output should match FP32 reference exactly
        (up to numerical tolerances).
        """
        # Create fresh wrapper that stays in NO_QUANT mode
        qgelu = QuantGELUTanh(self.fp_gelu_tanh)

        with torch.no_grad():
            q_out = qgelu(self.x)
            fp_out = self.gelutanh()(self.x)

        self.assertIs(qgelu._mode, Mode.NO_QUANT)
        self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-6, rtol=1e-6))

    def test_registration_in_registry(self):
        """
        Test that GELUTanh is properly registered in the wrapper registry.
        """
        from tico.quantization.wrapq.wrappers.nn.quant_gelutanh import QuantGELUTanh
        from tico.quantization.wrapq.wrappers.registry import lookup

        # Verify GELUTanh maps to QuantGELUTanh
        wrapper_cls = lookup(self.gelutanh)
        self.assertIs(wrapper_cls, QuantGELUTanh)
