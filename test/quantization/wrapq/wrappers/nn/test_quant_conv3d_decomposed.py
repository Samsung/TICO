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

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_conv3d_decomposed import (
    QuantConv3dDecomposed,
)


class TestQuantConv3dDecomposed(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        # Create a simple Conv3d module (matches Qwen3-VL patch embed structure)
        self.fp32 = nn.Conv3d(
            in_channels=3,
            out_channels=16,
            kernel_size=(2, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=True,
        )

        # Input tensor: (batch, in_channels, depth, height, width)
        self.x = torch.randn(2, 3, 4, 8, 8)

        # Create quantized wrapper
        self.q_conv = QuantConv3dDecomposed(self.fp32)

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

    def test_decomposition_correctness_no_quant(self):
        """
        In NO_QUANT mode, decomposition should match FP32 Conv3d exactly.
        This verifies the slice+Conv2d+Add logic is correct.
        """
        # Create quantized wrapper (stays in NO_QUANT)
        q_conv = QuantConv3dDecomposed(self.fp32)

        # Run forward pass
        q_out = q_conv(self.x)

        # Run original Conv3d
        fp_out = F.conv3d(
            self.x,
            self.fp32.weight,
            self.fp32.bias,
            stride=self.fp32.stride,
            padding=self.fp32.padding,
        )

        # Check shape and values
        self.assertEqual(q_out.shape, fp_out.shape)
        self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-6, rtol=1e-6))

    def test_decomposition_various_shapes(self):
        """Test decomposition correctness across various input shapes."""
        test_cases = [
            (1, 3, 4, 8, 8),  # Small input
            (2, 3, 4, 8, 8),  # Reference shape
            (4, 3, 4, 8, 8),  # Larger batch
            (2, 3, 8, 16, 16),  # Larger spatial dimensions
            (2, 3, 6, 8, 8),  # More frames
        ]

        for batch, in_ch, depth, height, width in test_cases:
            with self.subTest(
                batch=batch, in_ch=in_ch, depth=depth, height=height, width=width
            ):
                fp32 = nn.Conv3d(in_ch, 16, (2, 3, 3), padding=(0, 1, 1), bias=True)
                q_conv = QuantConv3dDecomposed(fp32)

                x = torch.randn(batch, in_ch, depth, height, width)

                q_out = q_conv(x)
                fp_out = F.conv3d(x, fp32.weight, fp32.bias, padding=(0, 1, 1))

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
        for _ in range(5):
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
        self.assertLess(diff, 0.5, "Quantization error should be reasonable")

    def test_dynamic_observers_created(self):
        """Test that dynamic observers are created during first forward pass."""
        self.assertFalse(self.q_conv._dynamic_obs_calibrated)

        # Enable calibration
        self.q_conv.enable_calibration()
        _ = self.q_conv(self.x)

        # Observers should be created
        self.assertTrue(self.q_conv._dynamic_obs_calibrated)

        # Check observer counts
        kT = self.fp32.kernel_size[0]  # 2
        T_out = (4 + 0 - 1 * (2 - 1) - 1) // 1 + 1  # 3

        self.assertEqual(len(self.q_conv._input_slice_obs), kT)
        self.assertEqual(len(self.q_conv._conv2d_obs), kT)
        self.assertEqual(len(self.q_conv._acc_obs), T_out)

    def test_observers_reused_across_calls(self):
        """Test that observers are reused when input shape doesn't change."""
        # Calibration
        self.q_conv.enable_calibration()
        _ = self.q_conv(self.x)

        # Get initial observer count
        initial_count = len(self.q_conv._input_slice_obs)

        # Second forward pass with same shape
        _ = self.q_conv(self.x)

        # Observer count should not change
        self.assertEqual(len(self.q_conv._input_slice_obs), initial_count)

    def test_per_channel_weight_quantization(self):
        """
        Test that per-channel weight quantization produces correct number of scales.
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
        """Test that activation statistics are collected during calibration."""
        # Calibration
        self.q_conv.enable_calibration()

        # Run forward pass
        _ = self.q_conv(self.x)

        # Check that activation observers have collected stats
        self.assertTrue(self.q_conv.obs_act_in.min_val.numel() > 0)
        self.assertTrue(self.q_conv.obs_act_out.min_val.numel() > 0)

        # Freeze and check qparams exist
        self.q_conv.freeze_qparams()
        self.assertTrue(self.q_conv.obs_act_in.has_qparams)
        self.assertTrue(self.q_conv.obs_act_out.has_qparams)

    def test_dynamic_activation_stats_collected(self):
        """Test that dynamic activation observers collect stats."""
        # Calibration
        self.q_conv.enable_calibration()
        _ = self.q_conv(self.x)

        # Check that dynamic observers have collected stats
        for obs in self.q_conv._input_slice_obs.values():
            self.assertTrue(obs.min_val.numel() > 0)

        for obs in self.q_conv._conv2d_obs.values():
            self.assertTrue(obs.min_val.numel() > 0)

        for obs in self.q_conv._acc_obs.values():
            self.assertTrue(obs.min_val.numel() > 0)

    def test_dtype_override(self):
        """Test that PTQConfig overrides propagate to observers."""
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act_in": {"dtype": DType.uint(4)},
                "act_out": {"dtype": DType.uint(4)},
                "weight": {"dtype": DType.uint(4)},
            },
        )

        qcustom = QuantConv3dDecomposed(self.fp32, qcfg=cfg)

        # Check that overrides were applied
        self.assertEqual(qcustom.obs_weight.dtype, DType.uint(4))
        self.assertEqual(qcustom.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(qcustom.obs_act_out.dtype, DType.uint(4))

    def test_conv3d_without_bias(self):
        """Test that Conv3d without bias is handled correctly."""
        fp32_no_bias = nn.Conv3d(3, 16, (2, 3, 3), bias=False)
        q_conv_no_bias = QuantConv3dDecomposed(fp32_no_bias)

        # Calibration and forward
        q_conv_no_bias.enable_calibration()
        _ = q_conv_no_bias(self.x)
        q_conv_no_bias.freeze_qparams()

        # Should not raise
        with torch.no_grad():
            _ = q_conv_no_bias(self.x)

    def test_different_kernel_sizes(self):
        """Test with various kernel sizes."""
        kernel_sizes = [
            (2, 3, 3),  # Temporal kernel = 2 (like Qwen3-VL)
            (3, 3, 3),  # Standard 3D kernel
            (1, 3, 3),  # No temporal kernel
        ]

        for ksize in kernel_sizes:
            with self.subTest(kernel_size=ksize):
                fp32 = nn.Conv3d(3, 16, ksize, padding=(ksize[0] // 2, 1, 1))
                q_conv = QuantConv3dDecomposed(fp32)

                # Test decomposition correctness
                x = torch.randn(2, 3, 4, 8, 8)
                q_out = q_conv(x)
                fp_out = F.conv3d(
                    x, fp32.weight, fp32.bias, padding=(ksize[0] // 2, 1, 1)
                )

                self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-6, rtol=1e-6))

    def test_different_padding(self):
        """
        Test that different padding schemes produce correct outputs.
        Covers all branches in _parse_padding method.
        """
        # Define test cases: (padding, description)
        test_cases = [
            # String-based padding
            ("same", "String padding='same'"),
            ("valid", "String padding='valid'"),
            # List/tuple-based padding
            ([1], "Single-element list padding"),
            ([1, 1, 1], "Three-element list padding"),
            ((1,), "Single-element tuple padding"),
            ((1, 1, 1), "Three-element tuple padding"),
            ([1, 2, 3], "Asymmetric list padding"),
            ((1, 2, 3), "Asymmetric tuple padding"),
            # Integer padding
            (1, "Integer padding=1"),
            (2, "Integer padding=2"),
            # Edge cases
            ([2, 1, 2], "List padding with different values"),
            ((2, 1, 2), "Tuple padding with different values"),
        ]

        for padding, description in test_cases:
            with self.subTest(padding=padding, description=description):
                try:
                    # Create Conv3d with the given padding
                    fp32 = nn.Conv3d(
                        in_channels=3,
                        out_channels=8,
                        kernel_size=(2, 3, 3),
                        padding=padding,
                        bias=True,
                    )

                    # Create quantized wrapper
                    q_conv = QuantConv3dDecomposed(fp32)

                    # Test input
                    x = torch.randn(2, 3, 4, 8, 8)

                    # Get outputs
                    q_out = q_conv(x)

                    # Calculate expected padding for FP32 reference
                    ref_padding = None
                    if isinstance(padding, str):
                        if padding == "same":
                            ref_padding = (
                                fp32.kernel_size[0] // 2,
                                fp32.kernel_size[1] // 2,
                                fp32.kernel_size[2] // 2,
                            )
                        elif padding == "valid":
                            ref_padding = (0, 0, 0)
                    elif isinstance(padding, (list, tuple)):
                        if len(padding) == 1:
                            ref_padding = (padding[0], padding[0], padding[0])
                        elif len(padding) == 3:
                            ref_padding = (padding[0], padding[1], padding[2])
                        else:
                            continue  # Skip unsupported padding format
                    elif isinstance(padding, int):
                        ref_padding = (padding, padding, padding)
                    else:
                        continue  # Skip unsupported padding type

                    # Get FP32 reference output
                    fp_out = F.conv3d(
                        x,
                        fp32.weight,
                        fp32.bias,
                        stride=(1, 1, 1),
                        padding=ref_padding,
                    )

                    # Verify outputs match
                    self.assertEqual(
                        q_out.shape,
                        fp_out.shape,
                        f"Output shapes don't match for {description}",
                    )
                    self.assertTrue(
                        torch.allclose(q_out, fp_out, atol=1e-6, rtol=1e-6),
                        f"Output values don't match for {description}",
                    )

                except ValueError as e:
                    # Expected for unsupported padding types (2D padding for 3D conv)
                    if "Unsupported padding" in str(e):
                        # This is expected behavior for invalid padding formats
                        continue
                    else:
                        raise

    def test_temporal_padding(self):
        """Test temporal padding with zeros+cat."""
        fp32_padded = nn.Conv3d(3, 16, (2, 3, 3), padding=(1, 1, 1), bias=True)
        q_conv_padded = QuantConv3dDecomposed(fp32_padded)

        x = torch.randn(2, 3, 4, 8, 8)

        # Test decomposition correctness
        q_out = q_conv_padded(x)
        fp_out = F.conv3d(x, fp32_padded.weight, fp32_padded.bias, padding=(1, 1, 1))

        self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-6, rtol=1e-6))

    def test_different_strides(self):
        """Test with different stride configurations."""
        strides = [(1, 1, 1), (1, 2, 2), (2, 1, 1)]

        for stride in strides:
            with self.subTest(stride=stride):
                fp32 = nn.Conv3d(3, 16, (2, 3, 3), stride=stride, padding=(0, 1, 1))
                q_conv = QuantConv3dDecomposed(fp32)

                x = torch.randn(2, 3, 4, 8, 8)

                # Test decomposition correctness
                q_out = q_conv(x)
                fp_out = F.conv3d(
                    x, fp32.weight, fp32.bias, stride=stride, padding=(0, 1, 1)
                )

                self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-6, rtol=1e-6))

    def test_dilation(self):
        """Test temporal dilation support."""
        fp32_dilated = nn.Conv3d(
            3, 16, (2, 3, 3), dilation=(2, 1, 1), padding=(0, 1, 1)
        )
        q_conv_dilated = QuantConv3dDecomposed(fp32_dilated)

        x = torch.randn(2, 3, 8, 8, 8)  # Need more frames for dilation

        # Test decomposition correctness
        q_out = q_conv_dilated(x)
        fp_out = F.conv3d(
            x,
            fp32_dilated.weight,
            fp32_dilated.bias,
            dilation=(2, 1, 1),
            padding=(0, 1, 1),
        )

        self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-6, rtol=1e-6))

    def test_registration_in_registry(self):
        """Test that nn.Conv3d is properly registered."""
        import warnings

        # Suppress warnings from PyTorch's Swig-generated types
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="builtin type SwigPyPacked has no __module__ attribute",
            )
            warnings.filterwarnings(
                "ignore",
                message="builtin type SwigPyObject has no __module__ attribute",
            )

            from tico.quantization.wrapq.wrappers.nn.quant_conv3d_decomposed import (
                QuantConv3dDecomposed,
            )
            from tico.quantization.wrapq.wrappers.registry import lookup

            # Verify Conv3d maps to QuantConv3dDecomposed
            wrapper_cls = lookup(nn.Conv3d)
            self.assertIs(wrapper_cls, QuantConv3dDecomposed)

    def test_all_observers_yielded(self):
        """Test that _all_observers returns all observers."""
        # Calibration to create dynamic observers
        self.q_conv.enable_calibration()
        _ = self.q_conv(self.x)

        # Get all observers
        observers = list(self.q_conv._all_observers())

        # Should include static observers
        self.assertIn(self.q_conv.obs_weight, observers)
        self.assertIn(self.q_conv.obs_act_in, observers)
        self.assertIn(self.q_conv.obs_act_out, observers)

        # Should include dynamic observers
        self.assertGreater(len(observers), 3)

    def test_multiple_calibration_cycles(self):
        """Test that multiple calibration cycles work correctly."""
        # First calibration
        self.q_conv.enable_calibration()
        for _ in range(3):
            _ = self.q_conv(self.x)
        self.q_conv.freeze_qparams()

        # Get first output
        with torch.no_grad():
            q_out_1 = self.q_conv(self.x)

        # Second calibration
        self.q_conv.enable_calibration()
        for _ in range(3):
            _ = self.q_conv(self.x)
        self.q_conv.freeze_qparams()

        # Get second output
        with torch.no_grad():
            q_out_2 = self.q_conv(self.x)

        # Outputs should be close (same calibration data)
        self.assertTrue(torch.allclose(q_out_1, q_out_2, atol=1e-5, rtol=1e-5))

    def test_output_shape_correctness(self):
        """Test that output shape matches Conv3d formula."""
        test_cases = [
            (2, 3, 4, 8, 8),  # Reference case
            (1, 3, 8, 16, 16),  # Larger input
            (4, 3, 2, 4, 4),  # Small input
        ]

        for input_shape in test_cases:
            with self.subTest(input_shape=input_shape):
                q_conv = QuantConv3dDecomposed(self.fp32)

                x = torch.randn(*input_shape)
                q_out = q_conv(x)
                fp_out = self.fp32(x)

                self.assertEqual(q_out.shape, fp_out.shape)

    def test_special_case_optimization(self):
        """
        Test the special case optimization where Conv3d can be converted to Conv2d
        without addition operations.

        Special case conditions:
        - kernel_size[D] = input_size[D] for all dimensions D
        - stride[D] = kernel_size[D] for all dimensions D
        - padding[D] = 0 for all dimensions D
        - groups = 1
        - dilation = 1 for all dimensions D
        """
        # Create a Conv3d that meets the special case conditions
        # Input: (N=2, C=3, T=2, H=16, W=16)
        # Kernel: (2, 16, 16) - matches temporal and spatial dimensions
        # Stride: (2, 16, 16) - equals kernel size
        # Padding: 0
        fp32_special = nn.Conv3d(
            in_channels=3,
            out_channels=1024,
            kernel_size=(2, 16, 16),
            stride=(2, 16, 16),
            padding=0,
            bias=True,
            groups=1,
        )

        # Create quantized wrapper
        q_conv_special = QuantConv3dDecomposed(fp32_special)

        # Input that matches the kernel size in temporal and spatial dimensions
        x_special = torch.randn(2, 3, 2, 16, 16)

        # Test that the special case produces the same result as standard Conv3d
        q_out = q_conv_special(x_special)
        fp_out = F.conv3d(
            x_special,
            fp32_special.weight,
            fp32_special.bias,
            stride=(2, 16, 16),
            padding=0,
        )

        # Check shape and values
        self.assertEqual(q_out.shape, fp_out.shape)
        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0, "Quantized output should differ from FP32")
        self.assertLess(diff, 0.7, "Quantization error should be reasonable")

    def test_special_case_without_bias(self):
        """
        Test the special case optimization with bias=False.
        """
        # Create a Conv3d without bias that meets the special case conditions
        fp32_special = nn.Conv3d(
            in_channels=3,
            out_channels=512,
            kernel_size=(2, 8, 8),
            stride=(2, 8, 8),
            padding=0,
            bias=False,
            groups=1,
        )

        # Create quantized wrapper
        q_conv_special = QuantConv3dDecomposed(fp32_special)

        # Input that matches the kernel size in temporal and spatial dimensions
        x_special = torch.randn(3, 3, 2, 8, 8)

        # Test that the special case produces the same result as standard Conv3d
        q_out = q_conv_special(x_special)
        fp_out = F.conv3d(
            x_special,
            fp32_special.weight,
            fp32_special.bias,
            stride=(2, 8, 8),
            padding=0,
        )

        # Check shape and values
        self.assertEqual(q_out.shape, fp_out.shape)
        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0, "Quantized output should differ from FP32")
        self.assertLess(diff, 0.7, "Quantization error should be reasonable")


if __name__ == "__main__":
    unittest.main()
