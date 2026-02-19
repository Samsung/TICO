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
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.nn.quant_layernorm import QuantLayerNorm
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_block import (
    QuantQwen3VLVisionBlock,
)


skip_msg = "transformers not installed — skipping Qwen3VLVisionBlock tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLVisionBlock(unittest.TestCase):
    fp_block: torch.nn.Module
    hidden_size: int
    num_heads: int
    head_dim: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

        # Use smaller sizes for testing
        cfg = Qwen3VLVisionConfig(
            hidden_size=64,
            num_heads=4,
        )

        # Ensure eager attention implementation so outputs are deterministic
        # and do not require GPU flash attention kernels.
        # Some versions use `_attn_implementation`, others expose `attn_implementation`.
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"

        cls.fp_block = Qwen3VLVisionBlock(cfg)
        cls.hidden_size = cfg.hidden_size
        cls.num_heads = cfg.num_heads
        cls.head_dim = cls.hidden_size // cls.num_heads

    def _rand_rope(self, seq_len):
        """Helper to create dummy rotary position embeddings"""
        emb = torch.randn(seq_len, self.head_dim)
        return emb.cos(), emb.sin()

    def _create_test_inputs(self, num_patches=32):
        """Helper to create test inputs for VisionBlock."""
        hidden_states = torch.randn(num_patches, self.hidden_size)
        # For testing, use a single chunk (no splitting) to avoid position embedding mismatch
        cu_seqlens = torch.tensor([0, num_patches])
        position_embeddings = self._rand_rope(num_patches)
        return hidden_states, cu_seqlens, position_embeddings

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        self.assertIs(q_block._mode, Mode.NO_QUANT)

        q_block.enable_calibration()
        self.assertIs(q_block._mode, Mode.CALIB)

        # Run forward pass during calibration
        hidden_states, cu_seqlens, pos_emb = self._create_test_inputs()
        _ = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)

        q_block.freeze_qparams()
        self.assertIs(q_block._mode, Mode.QUANT)

    def test_forward_diff(self):
        """
        Test that quantized output is acceptably close to FP32 reference.
        """
        torch.manual_seed(42)
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        q_block.enable_calibration()

        # Calibrate with multiple inputs
        for _ in range(4):
            hidden_states, cu_seqlens, pos_emb = self._create_test_inputs()
            _ = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)

        q_block.freeze_qparams()

        hidden_states, cu_seqlens, pos_emb = self._create_test_inputs()
        with torch.no_grad():
            q_out = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)
            fp_out = self.fp_block(
                hidden_states, cu_seqlens, position_embeddings=pos_emb
            )

        self.assertEqual(fp_out.shape, q_out.shape)
        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.7)  # acceptably close

    def test_registration_in_registry(self):
        """
        Test that Qwen3VLVisionBlock is properly registered.
        """
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_block import (
            QuantQwen3VLVisionBlock,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

        wrapper_cls = lookup(Qwen3VLVisionBlock)
        self.assertIs(wrapper_cls, QuantQwen3VLVisionBlock)

    def test_output_shape(self):
        """
        Test that output shape is preserved.
        Input: (num_patches, hidden_size)
        Output: (num_patches, hidden_size)
        """
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        q_block.enable_calibration()

        num_patches = 32
        hidden_states, cu_seqlens, pos_emb = self._create_test_inputs(num_patches)
        _ = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)

        q_block.freeze_qparams()

        with torch.no_grad():
            q_out = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)
            fp_out = self.fp_block(
                hidden_states, cu_seqlens, position_embeddings=pos_emb
            )

        expected_shape = (num_patches, self.hidden_size)
        self.assertEqual(q_out.shape, expected_shape)
        self.assertEqual(fp_out.shape, expected_shape)

    def test_residual_connection_preservation(self):
        """
        Test that residual connections are preserved (output close to input + transformation).
        """
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        q_block.enable_calibration()

        hidden_states, cu_seqlens, pos_emb = self._create_test_inputs()
        _ = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)

        q_block.freeze_qparams()

        with torch.no_grad():
            # Save input
            input_copy = hidden_states.clone()

            # Run forward pass
            output = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)

        # Output should be different from input (transformation applied)
        self.assertFalse(torch.equal(output, input_copy))

        # But shape should be preserved
        self.assertEqual(output.shape, input_copy.shape)

    def test_different_num_patches(self):
        """
        Test that quantization works correctly with different numbers of patches.
        """
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        q_block.enable_calibration()

        # Calibrate with one size
        calibrate_hidden, calibrate_cu, pos_emb = self._create_test_inputs(32)
        for _ in range(3):
            _ = q_block(calibrate_hidden, calibrate_cu, position_embeddings=pos_emb)
        q_block.freeze_qparams()

        # Test with different sizes
        for num_patches in [16, 32, 64]:
            hidden_states, cu_seqlens, pos_emb = self._create_test_inputs(num_patches)
            with torch.no_grad():
                q_out = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)
                fp_out = self.fp_block(
                    hidden_states, cu_seqlens, position_embeddings=pos_emb
                )

            self.assertEqual(q_out.shape[0], num_patches)
            self.assertEqual(q_out.shape[1], self.hidden_size)
            diff = (fp_out - q_out).abs().mean().item()
            self.assertLess(diff, 0.7)

    def test_observer_count(self):
        """
        Test that the wrapper has the correct number of observers.
        - 3 local observers (input, post_attn, output)
        """
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        q_block.enable_calibration()

        # Calibrate to ensure observers are initialized
        hidden_states, cu_seqlens, pos_emb = self._create_test_inputs()
        _ = q_block(hidden_states, cu_seqlens, position_embeddings=pos_emb)

        q_block.freeze_qparams()

        observers = list(q_block._all_observers())
        # Should have 3 local + submodules' observers
        self.assertEqual(len(observers), 3)
