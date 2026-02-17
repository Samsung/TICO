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
import unittest

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_rotary_embedding import (
    QuantQwen3VLTextRotaryEmbedding,
)


trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping Qwen3VLTextRotaryEmbedding tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantQwen3VLTextRotaryEmbedding(unittest.TestCase):
    fp_rope: torch.nn.Module
    hidden_size: int
    head_dim: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
        )

        # Use smaller config for testing
        cfg = Qwen3VLTextConfig(
            hidden_size=32,  # Smaller for testing
            num_attention_heads=4,
            max_position_embeddings=512,
        )
        cls.fp_rope = Qwen3VLTextRotaryEmbedding(cfg)
        cls.hidden_size = cfg.hidden_size
        cls.head_dim = (
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )  # 8

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        self.assertIs(q_rope._mode, Mode.NO_QUANT)

        q_rope.enable_calibration()
        self.assertIs(q_rope._mode, Mode.CALIB)

        # Run forward pass during calibration
        x = torch.randn(2, 64, self.head_dim)
        position_ids = torch.arange(64).unsqueeze(0).expand(2, -1)
        _ = q_rope(x, position_ids)

        q_rope.freeze_qparams()
        self.assertIs(q_rope._mode, Mode.QUANT)

    def test_quantised_output_close(self):
        """
        Test that quantized outputs (cos, sin) are acceptably close to FP32 reference.
        """
        torch.manual_seed(42)
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        # Calibrate with different sequence lengths
        for seq_len in [32, 64, 128]:
            x = torch.randn(2, seq_len, self.head_dim)
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)
            _ = q_rope(x, position_ids)

        q_rope.freeze_qparams()

        seq_len = 64
        x = torch.randn(2, seq_len, self.head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            q_cos, q_sin = q_rope(x, position_ids)
            fp_cos, fp_sin = self.fp_rope(x, position_ids)

        diff_cos = (fp_cos - q_cos).abs().mean().item()
        diff_sin = (fp_sin - q_sin).abs().mean().item()

        self.assertGreater(diff_cos, 0.0)  # not identical
        self.assertGreater(diff_sin, 0.0)
        self.assertLess(diff_cos, 0.4)  # acceptably close
        self.assertLess(diff_sin, 0.4)
        self.assertEqual(fp_cos.shape, q_cos.shape)
        self.assertEqual(fp_sin.shape, q_sin.shape)

    def test_output_shape(self):
        """
        Test that output shapes are correct: (batch_size, seq_len, head_dim)
        """
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        seq_len = 64
        x = torch.randn(2, seq_len, self.head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)
        _ = q_rope(x, position_ids)

        q_rope.freeze_qparams()

        with torch.no_grad():
            q_cos, q_sin = q_rope(x, position_ids)

        expected_shape = (2, seq_len, self.head_dim)
        self.assertEqual(q_cos.shape, expected_shape)
        self.assertEqual(q_sin.shape, expected_shape)

    def test_output_range(self):
        """
        Test that cos and sin outputs are in valid range [-1, 1].
        """
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        seq_len = 64
        x = torch.randn(2, seq_len, self.head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)
        _ = q_rope(x, position_ids)

        q_rope.freeze_qparams()

        with torch.no_grad():
            q_cos, q_sin = q_rope(x, position_ids)

        # Check ranges (with some tolerance for quantization error)
        self.assertLessEqual(q_cos.max(), 1.01)
        self.assertGreaterEqual(q_cos.min(), -1.01)
        self.assertLessEqual(q_sin.max(), 1.01)
        self.assertGreaterEqual(q_sin.min(), -1.01)

    def test_different_sequence_lengths(self):
        """
        Test that quantization works correctly with different sequence lengths.
        Calibrate with maximum length to cover full range.
        """
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        # Calibrate with MAXIMUM length
        max_seq_len = 256
        for _ in range(3):
            x = torch.randn(2, max_seq_len, self.head_dim)
            position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(2, -1)
            _ = q_rope(x, position_ids)

        q_rope.freeze_qparams()

        # Test with different lengths
        for seq_len in [32, 64, 128, 256]:
            x = torch.randn(2, seq_len, self.head_dim)
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)

            with torch.no_grad():
                q_cos, q_sin = q_rope(x, position_ids)
                fp_cos, fp_sin = self.fp_rope(x, position_ids)

            diff_cos = (fp_cos - q_cos).abs().mean().item()
            diff_sin = (fp_sin - q_sin).abs().mean().item()

            self.assertLess(diff_cos, 0.4)
            self.assertLess(diff_sin, 0.4)
            self.assertEqual(q_cos.shape[0], 2)
            self.assertEqual(q_cos.shape[1], seq_len)
            self.assertEqual(q_cos.shape[2], self.head_dim)

    def test_dtype_override(self):
        """
        PTQConfig overrides should affect the observers.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "cos": {"dtype": DType.uint(4)},
                "sin": {"dtype": DType.uint(4)},
            },
        )
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope, qcfg=cfg)

        self.assertEqual(q_rope.obs_cos.dtype, DType.uint(4))
        self.assertEqual(q_rope.obs_sin.dtype, DType.uint(4))

    def test_activation_stats_collected(self):
        """
        Test that activation statistics are properly collected during calibration.
        """
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        # Run forward pass to collect stats
        seq_len = 64
        x = torch.randn(2, seq_len, self.head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)
        _ = q_rope(x, position_ids)

        # Check that observers have collected stats
        self.assertTrue(
            q_rope.obs_cos.has_qparams or q_rope.obs_cos.min_val.numel() > 0
        )
        self.assertTrue(
            q_rope.obs_sin.has_qparams or q_rope.obs_sin.min_val.numel() > 0
        )

        # Freeze and check qparams exist
        q_rope.freeze_qparams()
        self.assertTrue(q_rope.obs_cos.has_qparams)
        self.assertTrue(q_rope.obs_sin.has_qparams)

    def test_observer_count(self):
        """
        Test that the wrapper has the correct number of observers.
        6 observers: inv_freq, freqs, freqs_mrope, emb, cos, sin
        """
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)

        observers = list(q_rope._all_observers())
        self.assertEqual(len(observers), 6)

    def test_registration_in_registry(self):
        """
        Test that Qwen3VLTextRotaryEmbedding is properly registered.
        """
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_rotary_embedding import (
            QuantQwen3VLTextRotaryEmbedding,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
        )

        wrapper_cls = lookup(Qwen3VLTextRotaryEmbedding)
        self.assertIs(wrapper_cls, QuantQwen3VLTextRotaryEmbedding)

    def test_no_learnable_parameters(self):
        """
        Test that the wrapper has no learnable parameters (only buffers).
        """
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)

        # Check that there are no parameters
        params = list(q_rope.parameters())
        self.assertEqual(len(params), 0)

        # Check that inv_freq is a buffer, not a parameter
        self.assertIsInstance(q_rope.inv_freq, torch.Tensor)
        self.assertIn("inv_freq", q_rope._buffers)

    def test_cos_sin_relationship(self):
        """
        Test that cos² + sin² = 1 (unit circle property).
        Quantization error should be small enough to preserve this property approximately.
        """
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        seq_len = 64
        x = torch.randn(2, seq_len, self.head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)
        _ = q_rope(x, position_ids)

        q_rope.freeze_qparams()

        with torch.no_grad():
            q_cos, q_sin = q_rope(x, position_ids)

        # Check unit circle property
        unit_circle = q_cos.pow(2) + q_sin.pow(2)
        # Allow some deviation due to quantization error
        self.assertGreaterEqual(unit_circle.min(), 0.95)
        self.assertLessEqual(unit_circle.max(), 1.05)

    def test_different_batch_sizes(self):
        """
        Test that quantization works correctly with different batch sizes.
        """
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        seq_len = 64
        # Calibrate with batch size 2
        for _ in range(3):
            x = torch.randn(2, seq_len, self.head_dim)
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)
            _ = q_rope(x, position_ids)

        q_rope.freeze_qparams()

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, seq_len, self.head_dim)
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

            with torch.no_grad():
                q_cos, q_sin = q_rope(x, position_ids)
                fp_cos, fp_sin = self.fp_rope(x, position_ids)

            diff_cos = (fp_cos - q_cos).abs().mean().item()
            diff_sin = (fp_sin - q_sin).abs().mean().item()

            self.assertLess(diff_cos, 0.4)
            self.assertLess(diff_sin, 0.4)
            self.assertEqual(q_cos.shape[0], batch_size)

    def test_mrope_semantic_equivalence(self):
        """
        Test that QuantQwen3VLTextRotaryEmbedding.apply_interleaved_mrope produces identical output
        to the original Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope.
        """
        torch.manual_seed(42)

        # Create test freqs tensor
        batch_size = 2
        seq_len = 64
        head_dim = self.head_dim
        freqs = torch.randn(3, batch_size, seq_len, head_dim // 2)

        # Call original implementation
        freqs_t_original = self.fp_rope.apply_interleaved_mrope(
            freqs, self.fp_rope.mrope_section
        )

        # Call new implementation
        q_rope = QuantQwen3VLTextRotaryEmbedding(self.fp_rope)
        freqs_t_new = q_rope.apply_interleaved_mrope(freqs, q_rope.mrope_section)

        # Compare outputs
        self.assertEqual(freqs_t_original.shape, freqs_t_new.shape)

        # Check exact equality (should be identical)
        torch.testing.assert_close(
            freqs_t_original,
            freqs_t_new,
            rtol=1e-5,
            atol=1e-5,
            msg="MRoPE implementations produce different outputs",
        )

        # Also check with different input shapes
        test_configs = [
            (1, 32, head_dim),  # Single sample, shorter sequence
            (4, 128, head_dim),  # Larger batch, longer sequence
            (2, 256, head_dim),  # Very long sequence
        ]

        for bs, sl, hd in test_configs:
            freqs = torch.randn(3, bs, sl, hd // 2)

            freqs_t_original = self.fp_rope.apply_interleaved_mrope(
                freqs, self.fp_rope.mrope_section
            )
            freqs_t_new = q_rope.apply_interleaved_mrope(freqs, q_rope.mrope_section)

            self.assertEqual(freqs_t_original.shape, freqs_t_new.shape)
            self.assertTrue(
                torch.equal(freqs_t_original, freqs_t_new),
                f"MRoPE implementations differ for shape (3, {bs}, {sl}, {hd//2})",
            )
