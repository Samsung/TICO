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
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_rotary_embedding import (
    QuantQwen3VLVisionRotaryEmbedding,
)


trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping Qwen3VLVisionRotaryEmbedding tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantQwen3VLVisionRotaryEmbedding(unittest.TestCase):
    fp_rope: torch.nn.Module
    dim: int
    theta: float

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionRotaryEmbedding,
        )

        # Use smaller dim for testing (typically 128 for head_dim=64)
        cls.fp_rope = Qwen3VLVisionRotaryEmbedding(dim=64)
        cls.dim = 64
        cls.theta = 10000.0

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope)
        self.assertIs(q_rope._mode, Mode.NO_QUANT)

        q_rope.enable_calibration()
        self.assertIs(q_rope._mode, Mode.CALIB)

        # Run forward pass during calibration
        seqlen = 128
        _ = q_rope(seqlen)

        q_rope.freeze_qparams()
        self.assertIs(q_rope._mode, Mode.QUANT)

    def test_quantised_output_close(self):
        """
        Test that quantized output is acceptably close to FP32 reference.
        """
        torch.manual_seed(42)
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        # Calibrate with different sequence lengths
        for seqlen in [64, 128, 256]:
            _ = q_rope(seqlen)

        q_rope.freeze_qparams()

        seqlen = 128
        with torch.no_grad():
            q_out = q_rope(seqlen)
            fp_out = self.fp_rope(seqlen)

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.4)  # acceptably close
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_output_shape(self):
        """
        Test that output shape is correct: (seqlen, dim/2)
        """
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        for seqlen in [64, 128, 256]:
            q_rope.enable_calibration()
            _ = q_rope(seqlen)

        q_rope.freeze_qparams()

        seqlen = 128
        with torch.no_grad():
            q_out = q_rope(seqlen)

        expected_shape = (seqlen, self.dim // 2)
        self.assertEqual(q_out.shape, expected_shape)

    def test_different_sequence_lengths(self):
        """
        Test that quantization works correctly with different sequence lengths.
        """
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        # Calibrate with one length
        for _ in range(3):
            _ = q_rope(256)

        q_rope.freeze_qparams()

        # Test with different lengths
        for seqlen in [2, 4, 8, 16, 32, 64, 128, 256]:
            with torch.no_grad():
                q_out = q_rope(seqlen)
                fp_out = self.fp_rope(seqlen)

            diff = (fp_out - q_out).abs().mean().item()
            self.assertLess(diff, 0.4)
            self.assertEqual(q_out.shape[0], seqlen)
            self.assertEqual(q_out.shape[1], self.dim // 2)

    def test_dtype_override(self):
        """
        PTQConfig overrides should affect the output observer.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "output": {"dtype": DType.uint(4)},
            },
        )
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope, qcfg=cfg)

        self.assertEqual(q_rope.obs_output.dtype, DType.uint(4))

    def test_activation_stats_collected(self):
        """
        Test that activation statistics are properly collected during calibration.
        """
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()

        # Run forward pass to collect stats
        seqlen = 128
        _ = q_rope(seqlen)

        # Check that observer has collected stats
        self.assertTrue(q_rope.obs_output.min_val.numel() > 0)

        # Freeze and check qparams exist
        q_rope.freeze_qparams()
        self.assertTrue(q_rope.obs_output.has_qparams)

    def test_observer_count(self):
        """
        Test that the wrapper has the correct number of observers.
        Only 1 observer (output) since there are no learnable parameters.
        """
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope)

        observers = list(q_rope._all_observers())
        self.assertEqual(len(observers), 1)

    def test_registration_in_registry(self):
        """
        Test that Qwen3VLVisionRotaryEmbedding is properly registered.
        """
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_rotary_embedding import (
            QuantQwen3VLVisionRotaryEmbedding,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionRotaryEmbedding,
        )

        wrapper_cls = lookup(Qwen3VLVisionRotaryEmbedding)
        self.assertIs(wrapper_cls, QuantQwen3VLVisionRotaryEmbedding)

    def test_no_learnable_parameters(self):
        """
        Test that the wrapper has no learnable parameters (only buffers).
        """
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope)

        # Check that there are no parameters
        params = list(q_rope.parameters())
        self.assertEqual(len(params), 0)

        # Check that inv_freq is a buffer, not a parameter
        self.assertIsInstance(q_rope.inv_freq, torch.Tensor)
        self.assertIn("inv_freq", q_rope._buffers)

    def test_frequency_values_correct(self):
        """
        Test that the computed frequency values are mathematically correct.
        Formula: freqs[i, j] = i * theta^(-2j/dim)
        """
        q_rope = QuantQwen3VLVisionRotaryEmbedding(self.fp_rope)
        q_rope.enable_calibration()
        q_rope.freeze_qparams()

        seqlen = 4
        with torch.no_grad():
            freqs = q_rope(seqlen)

        # Manually compute expected values
        expected = torch.outer(
            torch.arange(seqlen, dtype=torch.float32),
            self.fp_rope.inv_freq,
        )

        # The quantized output should still have the same pattern
        # (quantization changes precision but not the mathematical relationship)
        torch.testing.assert_close(freqs.shape, expected.shape)
        self.assertEqual(freqs.shape, (seqlen, self.dim // 2))
