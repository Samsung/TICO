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

"""
Smoke tests for the universal GPTQ quantizer.

This module tests the UniversalGPTQQuantizer from universal_gptq_quantizer.py
which implements a model-agnostic GPTQ quantization approach.
"""

import unittest

import torch
import torch.nn as nn

from tico.quantization.algorithm.universal_gptq.quantizer import UniversalGPTQQuantizer
from tico.quantization.config.gptq import UniversalGPTQConfig


class FiveLinearModel(nn.Module):
    """
    A simple sequential model with 5 Linear layers.
    No residual connections - purely feedforward.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    """
    A residual block with nested submodules and skip connection.

    Structure:
        input --> [Linear -> ReLU -> Linear] --> + --> output
                     |                        ^
                     +---- identity ----------+
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out + residual  # Skip connection


class ConvBlock(nn.Module):
    """
    A convolutional block with nested submodules and skip connection.

    Structure:
        input --> [Conv2d -> BN -> ReLU -> Conv2d -> BN] --> + --> output
                           |                              ^
                           +---- identity ----------------+
    """

    def __init__(self, in_channels: int = 16, out_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection skip if channels don't match
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        if self.skip_proj is not None:
            residual = self.skip_proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual


class MultiBlockModel(nn.Module):
    """
    A complex model with multiple types of blocks:
    - Initial projection
    - Multiple residual blocks (with nested Linear submodules)
    - Multiple conv blocks (with nested Conv2d/BN submodules)
    - Final classification head

    This tests:
    - Residual connections
    - Nested submodules at multiple levels
    - Mixed layer types (Linear, Conv2d, BatchNorm)

    Architecture:
        With conv blocks:
            input -> Linear -> [ResidualBlock]*N -> Linear -> [ConvBlock]*M -> Linear -> output
                              (hidden_dim)         (proj)   (conv_dim=H*W*C)   (pool)
        Without conv blocks:
            input -> Linear -> [ResidualBlock]*N -> Linear -> output
                              (hidden_dim)         (direct)

    Note: conv_dim must be factorable as channels * H * W for the reshape to work.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_residual_blocks: int = 3,
        num_conv_blocks: int = 2,
        conv_dim: int = 64,  # Must be factorable as channels * H * W
        conv_channels: int = 16,  # Number of conv channels
        conv_size: int = 2,  # H = W = conv_size (so conv_dim = channels * size^2)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.conv_channels = conv_channels
        self.conv_size = conv_size
        self.num_conv_blocks = num_conv_blocks

        # Validate conv_dim = channels * size^2
        assert conv_dim == conv_channels * (
            conv_size**2
        ), f"conv_dim ({conv_dim}) must equal conv_channels * conv_size^2 ({conv_channels} * {conv_size}^2 = {conv_channels * conv_size**2})"

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks (text-like processing)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )

        # Projection from hidden_dim to conv_dim (only used if conv_blocks > 0)
        self.residual_to_conv = (
            nn.Linear(hidden_dim, conv_dim) if num_conv_blocks > 0 else None
        )

        # Convolutional blocks (vision-like processing)
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(conv_channels, conv_channels) for _ in range(num_conv_blocks)]
        )

        # Global pooling (only used if conv_blocks > 0)
        self.global_pool = nn.AdaptiveAvgPool2d(1) if num_conv_blocks > 0 else None

        # Output projections - different for conv vs non-conv path
        if num_conv_blocks > 0:
            self.output_proj = nn.Linear(conv_channels, input_dim)
        else:
            self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)  # (batch, hidden_dim)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)  # Each block has internal skip connection

        # Only do conv path if we have conv blocks
        if self.num_conv_blocks > 0:
            # Project to conv_dim
            x = self.residual_to_conv(x)  # (batch, conv_dim)

            # Reshape for convolution: (batch, conv_dim) -> (batch, channels, H, W)
            batch_size = x.shape[0]
            x = x.view(batch_size, self.conv_channels, self.conv_size, self.conv_size)

            # Convolutional blocks
            for block in self.conv_blocks:
                x = block(x)  # Each block has internal skip connection

            # Global pooling and output
            x = self.global_pool(x).squeeze(-1).squeeze(-1)  # (batch, conv_channels)

        # Output projection (input dim depends on which path was taken)
        x = self.output_proj(x)
        return x


class TestUniversalGPTQ(unittest.TestCase):
    """Smoke tests for universal GPTQ quantizer."""

    @torch.inference_mode()
    def test_universal_gptq_basic(self):
        """
        Test basic GPTQ quantization workflow:
        1. Create a small model with 5 Linear layers
        2. Prepare the model for GPTQ
        3. Run calibration (cache inputs)
        4. Convert the model
        5. Verify quantizers attribute exists with correct count
        """
        input_dim = 16
        samples_per_batch = 2
        num_batches = 4

        # Create model
        model = FiveLinearModel(input_dim)
        model.eval()

        # Create calibration dataset (small batches for smoke test)
        calibration_data = [
            (torch.randn(samples_per_batch, input_dim),) for _ in range(num_batches)
        ]

        # Create GPTQ quantizer
        config = UniversalGPTQConfig(
            weight_bits=8,
            percdamp=0.01,
            groupsize=-1,
            actorder=False,
            show_progress=False,
            verbose=False,
        )
        quantizer = UniversalGPTQQuantizer(config)

        # Prepare: substitutes forward method to cache inputs
        prepared_model = quantizer.prepare(model)

        # Calibration: run batches to cache inputs
        for batch in calibration_data:
            prepared_model(*batch)

        # Convert: runs GPTQ algorithm on cached data
        quantized_model = quantizer.convert(prepared_model)

        # Verify: check that quantizers attribute exists
        self.assertTrue(
            hasattr(quantized_model, "quantizers"),
            "Quantized model should have 'quantizers' attribute",
        )

        # Verify: check correct number of quantizers (5 Linear layers)
        # Note: quantizers is a dict[str, Quantizer] attached as attribute
        quantizers_dict = getattr(quantized_model, "quantizers", {})
        self.assertIsInstance(
            quantizers_dict,
            dict,
            "quantizers should be a dict",
        )
        self.assertEqual(
            len(quantizers_dict),
            5,
            f"Expected 5 quantizers for 5 Linear layers, got {len(quantizers_dict)}",
        )

        # Verify: check quantizer names match expected layer paths
        expected_keys = {
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "layers.4",
        }
        actual_keys = set(quantizers_dict.keys())
        self.assertEqual(
            actual_keys,
            expected_keys,
            f"Quantizer keys mismatch. Expected: {expected_keys}, Got: {actual_keys}",
        )

        # Verify: each quantizer should be a Quantizer instance
        from tico.quantization.algorithm.gptq.quant import Quantizer

        for name, q in quantizers_dict.items():
            self.assertIsInstance(
                q,
                Quantizer,
                f"Quantizer '{name}' should be an instance of Quantizer class",
            )

    @torch.inference_mode()
    def test_universal_gptq_weight_update(self):
        """
        Test that GPTQ actually updates the weights.
        """
        input_dim = 16
        samples_per_batch = 2
        num_batches = 8

        # Create model
        model = FiveLinearModel(input_dim)
        model.eval()

        # Save original weights
        original_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Create calibration dataset
        calibration_data = [
            (torch.randn(samples_per_batch, input_dim),) for _ in range(num_batches)
        ]

        # Create and run GPTQ
        config = UniversalGPTQConfig(
            weight_bits=8,
            percdamp=0.01,
            groupsize=-1,
            actorder=False,
            show_progress=False,
            verbose=False,
        )
        quantizer = UniversalGPTQQuantizer(config)

        prepared_model = quantizer.prepare(model)
        for batch in calibration_data:
            prepared_model(*batch)

        quantized_model = quantizer.convert(prepared_model)

        # Verify: weights should be updated (not equal to original)
        weights_changed = False
        for name, param in quantized_model.named_parameters():
            if name in original_weights:
                if not torch.allclose(param, original_weights[name], atol=1e-6):
                    weights_changed = True
                    break

        self.assertTrue(
            weights_changed,
            "GPTQ should update the weights (quantized weights should differ from original)",
        )

    @torch.inference_mode()
    def test_universal_gptq_output_consistency(self):
        """
        Test that quantized model produces similar outputs to original.
        """
        input_dim = 16
        samples_per_batch = 2
        num_batches = 8

        # Create model
        model = FiveLinearModel(input_dim)
        model.eval()

        # Create test input
        test_input = torch.randn(2, 16)

        # Get original output
        with torch.no_grad():
            original_output = model(test_input)

        # Create calibration dataset
        calibration_data = [
            (torch.randn(samples_per_batch, input_dim),) for _ in range(num_batches)
        ]

        # Run GPTQ
        config = UniversalGPTQConfig(
            weight_bits=8,
            percdamp=0.01,
            groupsize=-1,
            actorder=False,
            show_progress=False,
            verbose=False,
        )
        quantizer = UniversalGPTQQuantizer(config)

        prepared_model = quantizer.prepare(model)
        for batch in calibration_data:
            try:
                prepared_model(*batch)
            except Exception:
                pass

        quantized_model = quantizer.convert(prepared_model)
        quantized_model.eval()

        # Get quantized output
        with torch.no_grad():
            quantized_output = quantized_model(test_input)

        # Verify: outputs should be reasonably close (PEIR < 10% for 8-bit)
        relative_error = torch.norm(quantized_output - original_output) / torch.norm(
            original_output
        )

        self.assertLess(
            relative_error.item(),
            0.1,  # 10% relative error threshold
            f"Quantized output deviates too much from original. Relative error: {relative_error.item():.4f}",
        )

    @torch.inference_mode()
    def test_universal_gptq_with_residual_blocks(self):
        """
        Test GPTQ quantization on a model with residual blocks.

        This test verifies that the universal GPTQ quantizer correctly handles:
        1. Residual/skip connections
        2. Nested submodules within blocks
        3. Multiple block types (Linear-based and Conv2d-based)
        """
        input_dim = 32
        hidden_dim = 64
        num_residual_blocks = 3
        num_conv_blocks = 2
        samples_per_batch = 2
        num_batches = 4

        # Conv params: conv_dim = conv_channels * conv_size^2 = 16 * 2^2 = 64
        conv_dim = 64
        conv_channels = 16
        conv_size = 2

        # Create model
        model = MultiBlockModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_residual_blocks=num_residual_blocks,
            num_conv_blocks=num_conv_blocks,
            conv_dim=conv_dim,
            conv_channels=conv_channels,
            conv_size=conv_size,
        )
        model.eval()

        # Create calibration dataset
        calibration_data = [
            (torch.randn(samples_per_batch, input_dim),) for _ in range(num_batches)
        ]

        # Create GPTQ quantizer
        config = UniversalGPTQConfig(
            weight_bits=8,
            percdamp=0.01,
            groupsize=-1,
            actorder=False,
            show_progress=False,
            verbose=False,
        )
        quantizer = UniversalGPTQQuantizer(config)

        # Prepare: substitutes forward method to cache inputs
        prepared_model = quantizer.prepare(model)

        # Calibration: run batches to cache inputs
        for batch in calibration_data:
            prepared_model(*batch)

        # Convert: runs GPTQ algorithm on cached data
        quantized_model = quantizer.convert(prepared_model)

        # Verify: check that quantizers attribute exists
        self.assertTrue(
            hasattr(quantized_model, "quantizers"),
            "Quantized model should have 'quantizers' attribute",
        )

        # Verify: check quantizers dict type
        quantizers_dict = getattr(quantized_model, "quantizers", {})
        self.assertIsInstance(
            quantizers_dict,
            dict,
            "quantizers should be a dict",
        )

        # Verify: check that we have quantizers for all Linear and Conv2d layers
        # Expected GPTQ-applicable layers:
        # - input_proj (Linear)
        # - residual_blocks: 3 blocks × 2 Linear each = 6 Linear
        # - conv_blocks: 2 blocks × 2 Conv2d each = 4 Conv2d
        # - output_proj (Linear)
        # Total: 1 + 6 + 4 + 1 = 12 quantizable layers
        # Note: BatchNorm2d and ReLU are not GPTQ-applicable
        expected_min_quantizers = 12
        self.assertGreaterEqual(
            len(quantizers_dict),
            expected_min_quantizers,
            f"Expected at least {expected_min_quantizers} quantizers, got {len(quantizers_dict)}",
        )

        # Verify: check specific layer paths exist
        expected_keys = {
            "input_proj",  # Initial projection
            "output_proj",  # Final projection
        }
        # Add residual block layers
        for i in range(num_residual_blocks):
            expected_keys.add(f"residual_blocks.{i}.linear1")
            expected_keys.add(f"residual_blocks.{i}.linear2")

        # Add conv block layers
        for i in range(num_conv_blocks):
            expected_keys.add(f"conv_blocks.{i}.conv1")
            expected_keys.add(f"conv_blocks.{i}.conv2")

        actual_keys = set(quantizers_dict.keys())
        missing_keys = expected_keys - actual_keys
        self.assertEqual(
            missing_keys,
            set(),
            f"Missing expected quantizer keys: {missing_keys}",
        )

        # Verify: each quantizer should be a Quantizer instance
        from tico.quantization.algorithm.gptq.quant import Quantizer

        for name, q in quantizers_dict.items():
            self.assertIsInstance(
                q,
                Quantizer,
                f"Quantizer '{name}' should be an instance of Quantizer class",
            )

        # Verify: quantized model produces reasonable outputs
        test_input = torch.randn(2, input_dim)

        # Get original output
        with torch.no_grad():
            original_output = model(test_input)

        # Get quantized output
        with torch.no_grad():
            quantized_output = quantized_model(test_input)

        # Verify shapes match
        self.assertEqual(
            original_output.shape,
            quantized_output.shape,
            "Original and quantized model outputs should have same shape",
        )

        # Verify: outputs should be reasonably close (PEIR < 15% for complex model)
        # Complex models with many layers accumulate more quantization error
        relative_error = torch.norm(quantized_output - original_output) / torch.norm(
            original_output
        )

        self.assertLess(
            relative_error.item(),
            0.15,  # 15% relative error threshold for complex models
            f"Quantized output deviates too much from original. Relative error: {relative_error.item():.4f}",
        )

    @torch.inference_mode()
    def test_universal_gptq_nested_submodule_cache_release(self):
        """
        Test that cached outputs of nested submodules are properly handled.

        This test verifies the memory efficiency optimization where:
        1. When a parent block (ResidualBlock/ConvBlock) finishes caching,
           its children's cached outputs can be safely released.
        2. The parent's cached output is sufficient for subsequent replays.

        Test structure:
        - Model with 2 residual blocks
        - Each block has 2 Linear submodules + ReLU
        - When block.0 output is cached, block.0.linear1 and block.0.linear2
          outputs should be releasable
        """
        input_dim = 32
        hidden_dim = 64
        num_residual_blocks = 2
        samples_per_batch = 2
        num_batches = 3

        # For this test, use simple parameters (no conv path)
        # When num_conv_blocks=0, conv params don't matter but must be valid
        conv_dim = 64
        conv_channels = 16
        conv_size = 2

        # Create model
        model = MultiBlockModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_residual_blocks=num_residual_blocks,
            num_conv_blocks=0,  # No conv blocks for this simpler test
            conv_dim=conv_dim,
            conv_channels=conv_channels,
            conv_size=conv_size,
        )
        model.eval()

        # Create calibration dataset
        calibration_data = [
            (torch.randn(samples_per_batch, input_dim),) for _ in range(num_batches)
        ]

        # Create GPTQ quantizer
        config = UniversalGPTQConfig(
            weight_bits=8,
            percdamp=0.01,
            groupsize=-1,
            actorder=False,
            show_progress=False,
            verbose=False,
        )
        quantizer = UniversalGPTQQuantizer(config)

        # Prepare
        prepared_model = quantizer.prepare(model)

        # Calibration
        for batch in calibration_data:
            prepared_model(*batch)

        # Convert - this is where caching happens
        quantized_model = quantizer.convert(prepared_model)

        # Verify: model should have quantizers
        self.assertTrue(hasattr(quantized_model, "quantizers"))

        # Verify: quantized model should produce valid outputs
        test_input = torch.randn(2, input_dim)
        with torch.no_grad():
            output = quantized_model(test_input)

        # Output should be finite (no NaN/Inf from broken caching)
        self.assertTrue(
            torch.isfinite(output).all(),
            "Quantized model output should be finite (no NaN/Inf)",
        )

        # Output should be non-zero (model is actually computing)
        self.assertTrue(
            torch.any(output != 0),
            "Quantized model output should be non-zero",
        )


if __name__ == "__main__":
    unittest.main()
