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
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_conv3d import QuantConv3d


class TestQuantConv3d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        # Create a simple Conv3d module (matches Qwen3-VL patch embed structure)
        self.fp32 = nn.Conv3d(
            in_channels=3,
            out_channels=16,  # Smaller for testing
            kernel_size=(2, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=True,
        )

        # Input tensor: (batch, in_channels, depth, height, width)
        self.x = torch.randn(2, 3, 8, 16, 16)

        # Create quantized wrapper
        self.q_conv = QuantConv3d(self.fp32)

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        # Initially in NO_QUANT mode
        self.assertIs(self.q_conv._mode, Mode.NO_QUANT)

        # Enable calibration
        self.q_conv.enable_calibration()
        _ = self.q_conv(self.x)
        self.assertIs(self.q_conv._mode, Mode.CALIB)

        # Freeze quantization parameters
        self.q_conv.freeze_qparams()
        self.assertIs(self.q_conv._mode, Mode.QUANT)

    def test_no_quant_matches_reference_various_shapes(self):
        """
        In NO_QUANT mode, output should match FP32 reference exactly
        (up to numerical tolerances) across various input shapes.
        """
        test_cases = [
            (2, 3, 4, 8, 8),  # Small input
            (1, 3, 16, 32, 32),  # Larger input
            (4, 3, 8, 16, 16),  # Larger batch
        ]

        for batch, in_ch, depth, height, width in test_cases:
            with self.subTest(
                batch=batch, in_ch=in_ch, depth=depth, height=height, width=width
            ):
                fp32 = nn.Conv3d(in_ch, 8, (2, 3, 3), padding=(0, 1, 1), bias=True)
                q_conv = QuantConv3d(
                    fp32
                )  # Stays in NO_QUANT unless calibration is enabled

                x = torch.randn(batch, in_ch, depth, height, width)

                q_out = q_conv(x)
                fp_out = F.conv3d(
                    x, fp32.weight, fp32.bias, stride=(1, 1, 1), padding=(0, 1, 1)
                )

                self.assertIs(q_conv._mode, Mode.NO_QUANT)
                self.assertEqual(q_out.shape, fp_out.shape)
                self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-6, rtol=1e-6))

    def test_quantized_output_close(self):
        """
        After calibration and freeze, quantized output should:
        - Differ from FP reference (quantization actually applied)
        - Stay within reasonable error bounds
        """
        # Calibration
        self.q_conv.enable_calibration()
        for _ in range(5):  # Multiple calibration iterations
            _ = self.q_conv(self.x)
        self.q_conv.freeze_qparams()

        # Compare outputs
        with torch.no_grad():
            q_out = self.q_conv(self.x)
            fp_out = F.conv3d(
                self.x,
                self.fp32.weight,
                self.fp32.bias,
                stride=self.fp32.stride,
                padding=self.fp32.padding,
            )

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0, "Quantized output should differ from FP32")
        self.assertLess(diff, 0.4, "Quantization error should be reasonable")

    def test_weight_stats_survive(self):
        """
        Re-running calibration cycles should not change the fixed weight quant stats.
        (Weights are constant; their computed scales should be stable.)
        """
        # First calibration
        self.q_conv.enable_calibration()
        self.q_conv.obs_weight.compute_qparams()
        self.assertTrue(hasattr(self.q_conv.obs_weight, "_cached_scale"))
        pre_scale = self.q_conv.obs_weight._cached_scale.clone()

        # Full calibration cycle again
        self.q_conv.enable_calibration()
        for _ in range(5):
            _ = self.q_conv(self.x)
        self.q_conv.freeze_qparams()

        # Check scale stability
        post_scale = self.q_conv.obs_weight._cached_scale
        self.assertTrue(torch.allclose(pre_scale, post_scale, atol=1e-6, rtol=1e-6))

    def test_per_channel_weight_quantization(self):
        """
        Test that per-channel weight quantization produces correct number of scales.
        For Conv3d, per-channel quantization along axis=0 should produce
        (out_channels,) scales and zero-points.
        """
        # Calibration
        self.q_conv.enable_calibration()
        self.q_conv.obs_weight.compute_qparams()
        self.q_conv.freeze_qparams()

        # Check that scale/zero_point have correct shape (per output channel)
        expected_num_channels = self.fp32.out_channels
        self.assertEqual(
            self.q_conv.obs_weight._cached_scale.shape[0], expected_num_channels
        )
        self.assertEqual(
            self.q_conv.obs_weight._cached_zp.shape[0], expected_num_channels
        )

    def test_activation_stats_collected(self):
        """
        Test that activation statistics are properly collected during calibration.
        """
        # Calibration
        self.q_conv.enable_calibration()

        # Run forward pass
        _ = self.q_conv(self.x)

        # Check that activation observers have collected stats
        self.assertTrue(
            self.q_conv.obs_act_in.has_qparams
            or self.q_conv.obs_act_in.min_val.numel() > 0
        )
        self.assertTrue(
            self.q_conv.obs_act_out.has_qparams
            or self.q_conv.obs_act_out.min_val.numel() > 0
        )

        # Freeze and check qparams exist
        self.q_conv.freeze_qparams()
        self.assertTrue(self.q_conv.obs_act_in.has_qparams)
        self.assertTrue(self.q_conv.obs_act_out.has_qparams)

    def test_dtype_override(self):
        """
        PTQConfig overrides should propagate to observers created by QuantConv3d.
        Test that different dtypes can be applied to input, output, and weights.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act_in": {"dtype": DType.uint(4)},
                "act_out": {"dtype": DType.uint(4)},
                "weight": {"dtype": DType.uint(4)},
            },
        )

        qcustom = QuantConv3d(self.fp32, qcfg=cfg)

        # Check that overrides were applied
        self.assertEqual(qcustom.obs_weight.dtype, DType.uint(4))
        self.assertEqual(qcustom.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(qcustom.obs_act_out.dtype, DType.uint(4))

    def test_conv3d_without_bias(self):
        """Test that Conv3d without bias is handled correctly."""
        fp32_no_bias = nn.Conv3d(3, 8, (2, 3, 3), bias=False)
        q_conv_no_bias = QuantConv3d(fp32_no_bias)

        # Calibration and forward
        q_conv_no_bias.enable_calibration()
        _ = q_conv_no_bias(self.x)
        q_conv_no_bias.freeze_qparams()

        # Should not raise
        with torch.no_grad():
            _ = q_conv_no_bias(self.x)

    def test_different_kernel_sizes(self):
        """Test with various kernel sizes (common in vision models)."""
        kernel_sizes = [
            (2, 3, 3),  # Temporal kernel = 2 (like Qwen3-VL)
            (3, 3, 3),  # Standard 3D kernel
            (1, 1, 1),  # 1x1x1 kernel
        ]

        for ksize in kernel_sizes:
            with self.subTest(kernel_size=ksize):
                fp32 = nn.Conv3d(3, 8, ksize, padding=(ksize[0] // 2, 1, 1))
                q_conv = QuantConv3d(fp32)

                q_conv.enable_calibration()
                _ = q_conv(self.x)
                q_conv.freeze_qparams()

                # Should not raise
                with torch.no_grad():
                    _ = q_conv(self.x)

                self.assertIs(q_conv._mode, Mode.QUANT)

    def test_bias_not_quantized(self):
        """
        Test that bias is not quantized (standard practice for convolution biases).
        The bias should remain as FP32 and be added to quantized output.
        """
        fp32_with_bias = nn.Conv3d(3, 8, (2, 3, 3), bias=True)
        q_conv = QuantConv3d(fp32_with_bias)

        q_conv.enable_calibration()
        _ = q_conv(self.x)
        q_conv.freeze_qparams()

        # Verify bias is still in the module and not modified
        self.assertIsNotNone(q_conv.module.bias)
        self.assertTrue(torch.equal(q_conv.module.bias, fp32_with_bias.bias))

    def test_registration_in_registry(self):
        """
        Test that nn.Conv3d is properly registered in the wrapper registry.
        """
        from tico.quantization.wrapq.wrappers.nn.quant_conv3d import QuantConv3d
        from tico.quantization.wrapq.wrappers.registry import lookup

        # Verify Conv3d maps to QuantConv3d
        wrapper_cls = lookup(nn.Conv3d)
        self.assertIs(wrapper_cls, QuantConv3d)
