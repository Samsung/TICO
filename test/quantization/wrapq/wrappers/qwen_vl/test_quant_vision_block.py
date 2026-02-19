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
from tico.quantization.wrapq.wrappers.nn.quant_layernorm import QuantLayerNorm
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_block import (
    QuantQwen3VLVisionBlock,
)


trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping Qwen3VLVisionBlock tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantQwen3VLVisionBlock(unittest.TestCase):
    fp_block: torch.nn.Module
    hidden_size: int
    num_attention_heads: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

        # Use smaller sizes for testing
        cfg = Qwen3VLVisionConfig(
            hidden_size=64,
            num_attention_heads=4,
            intermediate_size=256,
        )

        cls.fp_block = Qwen3VLVisionBlock(cfg)
        cls.hidden_size = cfg.hidden_size
        cls.num_attention_heads = cfg.num_attention_heads

    def _create_test_inputs(self, num_patches=32):
        """Helper to create test inputs for VisionBlock."""
        hidden_states = torch.randn(num_patches, self.hidden_size)
        cu_seqlens = torch.arange(0, num_patches + 1, 8)
        return hidden_states, cu_seqlens

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        self.assertIs(q_block._mode, Mode.NO_QUANT)

        q_block.enable_calibration()
        self.assertIs(q_block._mode, Mode.CALIB)

        # Run forward pass during calibration
        hidden_states, cu_seqlens = self._create_test_inputs()
        _ = q_block(hidden_states, cu_seqlens)

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
            hidden_states, cu_seqlens = self._create_test_inputs()
            _ = q_block(hidden_states, cu_seqlens)

        q_block.freeze_qparams()

        hidden_states, cu_seqlens = self._create_test_inputs()
        with torch.no_grad():
            q_out = q_block(hidden_states, cu_seqlens)
            fp_out = self.fp_block(hidden_states, cu_seqlens)

        self.assertEqual(fp_out.shape, q_out.shape)
        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.7)  # acceptably close

    def test_module_override(self):
        """
        PTQConfig overrides should propagate to wrapped submodules.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "attn": {
                    "weight": {"dtype": DType.uint(4)},
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                },
                "mlp": {
                    "weight": {"dtype": DType.uint(4)},
                },
                "norm1": {
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                },
            },
        )
        q_block = QuantQwen3VLVisionBlock(self.fp_block, qcfg=cfg)

        # Check norm1
        q_norm1 = q_block.norm1.wrapped
        self.assertIsInstance(q_norm1, QuantLayerNorm)
        self.assertEqual(q_norm1.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_norm1.obs_act_out.dtype, DType.uint(4))

        # Check norm2
        q_norm2 = q_block.norm2.wrapped
        self.assertIsInstance(q_norm2, QuantLayerNorm)

        # Check attn and mlp are wrapped
        self.assertIsNotNone(q_block.attn.wrapped)
        self.assertIsNotNone(q_block.mlp.wrapped)

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
        hidden_states, cu_seqlens = self._create_test_inputs(num_patches)
        _ = q_block(hidden_states, cu_seqlens)

        q_block.freeze_qparams()

        with torch.no_grad():
            q_out = q_block(hidden_states, cu_seqlens)
            fp_out = self.fp_block(hidden_states, cu_seqlens)

        expected_shape = (num_patches, self.hidden_size)
        self.assertEqual(q_out.shape, expected_shape)
        self.assertEqual(fp_out.shape, expected_shape)

    def test_residual_connection_preservation(self):
        """
        Test that residual connections are preserved (output close to input + transformation).
        """
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        q_block.enable_calibration()

        hidden_states, cu_seqlens = self._create_test_inputs()
        _ = q_block(hidden_states, cu_seqlens)

        q_block.freeze_qparams()

        with torch.no_grad():
            # Save input
            input_copy = hidden_states.clone()

            # Run forward pass
            output = q_block(hidden_states, cu_seqlens)

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
        calibrate_hidden, calibrate_cu = self._create_test_inputs(32)
        for _ in range(3):
            _ = q_block(calibrate_hidden, calibrate_cu)
        q_block.freeze_qparams()

        # Test with different sizes
        for num_patches in [16, 32, 64]:
            hidden_states, cu_seqlens = self._create_test_inputs(num_patches)
            with torch.no_grad():
                q_out = q_block(hidden_states, cu_seqlens)
                fp_out = self.fp_block(hidden_states, cu_seqlens)

            self.assertEqual(q_out.shape[0], num_patches)
            self.assertEqual(q_out.shape[1], self.hidden_size)
            diff = (fp_out - q_out).abs().mean().item()
            self.assertLess(diff, 0.7)

    def test_observer_count(self):
        """
        Test that the wrapper has the correct number of observers.
        - 3 local observers (input, post_attn, output)
        - Observers from norm1 (QuantLayerNorm)
        - Observers from norm2 (QuantLayerNorm)
        - Observers from attn (QuantQwen3VLVisionAttention)
        - Observers from mlp (QuantQwen3VLVisionMLP)
        """
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        q_block.enable_calibration()

        # Calibrate to ensure observers are initialized
        hidden_states, cu_seqlens = self._create_test_inputs()
        _ = q_block(hidden_states, cu_seqlens)

        q_block.freeze_qparams()

        observers = list(q_block._all_observers())
        # Should have 3 local + submodules' observers
        self.assertGreater(len(observers), 3)

    def test_subgraph_export(self):
        """
        Test that quantized block can be exported to Circle format.
        """
        q_block = QuantQwen3VLVisionBlock(self.fp_block).eval()
        hidden_states, cu_seqlens = self._create_test_inputs(16)

        # Calibrate and freeze
        q_block.enable_calibration()
        _ = q_block(hidden_states, cu_seqlens)
        q_block.freeze_qparams()

        self.assertIs(q_block._mode, Mode.QUANT)

        # Export to Circle
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "vision_block.circle"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                exported = tico.convert(q_block, (hidden_states, cu_seqlens))
            exported.save(path)
            self.assertTrue(path.exists())

    def test_with_position_embeddings(self):
        """
        Test that the block works correctly with position embeddings.
        """
        q_block = QuantQwen3VLVisionBlock(self.fp_block)
        q_block.enable_calibration()

        num_patches = 32
        hidden_states, cu_seqlens = self._create_test_inputs(num_patches)

        # Create dummy position embeddings
        head_dim = self.hidden_size // self.num_attention_heads
        pos_emb = torch.randn(num_patches, head_dim)

        _ = q_block(hidden_states, cu_seqlens, rotary_pos_emb=pos_emb)

        q_block.freeze_qparams()

        with torch.no_grad():
            q_out = q_block(hidden_states, cu_seqlens, rotary_pos_emb=pos_emb)
            fp_out = self.fp_block(hidden_states, cu_seqlens, rotary_pos_emb=pos_emb)

        self.assertEqual(q_out.shape, fp_out.shape)
        diff = (fp_out - q_out).abs().mean().item()
        self.assertLess(diff, 0.7)
