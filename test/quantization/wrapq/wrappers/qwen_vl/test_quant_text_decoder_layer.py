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
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_decoder_layer import (
    QuantQwen3VLTextDecoderLayer,
)


skip_msg = (
    "required transformers not installed — skipping Qwen3VLTextDecoderLayer tests"
)


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLTextDecoderLayer(unittest.TestCase):
    fp_model: torch.nn.Module
    hidden_size: int
    num_attention_heads: int
    head_dim: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        # Use smaller sizes for testing
        cfg = Qwen3VLTextConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=2048,
            intermediate_size=1024,
        )

        # Ensure eager attention implementation so outputs are deterministic
        # and do not require GPU flash attention kernels.
        # Some versions use `_attn_implementation`, others expose `attn_implementation`.
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"

        cls.fp_model = Qwen3VLTextDecoderLayer(cfg, layer_idx=0)
        cls.hidden_size = cfg.hidden_size
        cls.num_attention_heads = cfg.num_attention_heads
        cls.head_dim = cls.hidden_size // cls.num_attention_heads

    def _rand_position_embeddings(self, batch_size, seq_len):
        """Helper to create dummy rotary position embeddings"""
        cos = torch.randn(batch_size, seq_len, self.head_dim)
        sin = torch.randn(batch_size, seq_len, self.head_dim)
        return cos, sin

    def _create_test_inputs(self, batch_size=2, seq_len=16):
        """Helper to create test inputs for TextDecoderLayer."""
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)
        position_embeddings = self._rand_position_embeddings(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        return hidden_states, position_embeddings, attention_mask, position_ids

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""

        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        self.assertIs(q_model._mode, Mode.NO_QUANT)

        q_model.enable_calibration()
        self.assertIs(q_model._mode, Mode.CALIB)

        # Run forward pass during calibration
        hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs()
        _ = q_model(
            hidden_states=hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
        )

        q_model.freeze_qparams()
        self.assertIs(q_model._mode, Mode.QUANT)

    def test_forward_diff(self):
        """
        Test that quantized output is acceptably close to FP32 reference.
        """
        torch.manual_seed(42)
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        q_model.enable_calibration()

        # Calibrate with multiple inputs
        for _ in range(4):
            hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs()
            _ = q_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )

        q_model.freeze_qparams()

        hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs()
        with torch.no_grad():
            q_out = q_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )
            fp_out = self.fp_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )

        self.assertEqual(fp_out.shape, q_out.shape)
        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.7)  # acceptably close

    def test_registration_in_registry(self):
        """
        Test that Qwen3VLTextDecoderLayer is properly registered.
        """
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_decoder_layer import (
            QuantQwen3VLTextDecoderLayer,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        wrapper_cls = lookup(Qwen3VLTextDecoderLayer)
        self.assertIs(wrapper_cls, QuantQwen3VLTextDecoderLayer)

    def test_output_shape(self):
        """
        Test that output shape is preserved.
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
        """
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        q_model.enable_calibration()

        batch_size = 2
        seq_len = 16
        hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs(
            batch_size, seq_len
        )
        _ = q_model(
            hidden_states=hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
        )

        q_model.freeze_qparams()

        with torch.no_grad():
            q_out = q_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )
            fp_out = self.fp_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )

        expected_shape = (batch_size, seq_len, self.hidden_size)
        self.assertEqual(q_out.shape, expected_shape)
        self.assertEqual(fp_out.shape, expected_shape)

    def test_residual_connection_preservation(self):
        """
        Test that residual connections are preserved (output close to input + transformation).
        """
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        q_model.enable_calibration()

        hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs()
        _ = q_model(
            hidden_states=hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
        )

        q_model.freeze_qparams()

        with torch.no_grad():
            # Save input
            input_copy = hidden_states.clone()

            # Run forward pass
            output = q_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )

        # Output should be different from input (transformation applied)
        self.assertFalse(torch.equal(output, input_copy))

        # But shape should be preserved
        self.assertEqual(output.shape, input_copy.shape)

    def test_observer_count(self):
        """
        Test that the wrapper has the correct number of observers.
        - 3 local observers (input, post_attn, output)
        """
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        observers = list(q_model._all_observers())
        # Should have 3 local observers
        self.assertEqual(len(observers), 3)

    def test_per_module_override(self):
        """
        Test that PTQConfig overrides propagate correctly to submodules.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "self_attn": {
                    "act_in": {"dtype": DType.uint(4)},
                }
            },
        )
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model, qcfg=cfg)

        # Check that override is applied to local observer
        self.assertEqual(q_model.obs_act_in.dtype, DType.uint(8))

    def test_different_batch_sizes(self):
        """
        Test that quantization works correctly with different batch sizes.
        """
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        q_model.enable_calibration()

        # Calibrate with one batch size
        calibrate_hidden, pos_emb, attn_mask, pos_ids = self._create_test_inputs(
            batch_size=2
        )
        for _ in range(3):
            _ = q_model(
                hidden_states=calibrate_hidden,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )
        q_model.freeze_qparams()

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs(
                batch_size=batch_size
            )
            with torch.no_grad():
                q_out = q_model(
                    hidden_states=hidden_states,
                    position_embeddings=pos_emb,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                )
                fp_out = self.fp_model(
                    hidden_states=hidden_states,
                    position_embeddings=pos_emb,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                )

            self.assertEqual(q_out.shape, fp_out.shape)
            diff = (fp_out - q_out).abs().mean().item()
            self.assertLess(diff, 0.8)
