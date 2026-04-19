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
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_model import (
    QuantQwen3VLTextModel,
)

skip_msg = "required transformers not installed — skipping Qwen3VLTextModel tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLTextModel(unittest.TestCase):
    fp_model: torch.nn.Module
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        # Use smaller sizes for testing
        cfg = Qwen3VLTextConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=1024,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
            use_cache=True,
        )

        # Ensure eager attention implementation so outputs are deterministic
        # and do not require GPU flash attention kernels.
        # Some versions use `_attn_implementation`, others expose `attn_implementation`.
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"

        cls.fp_model = Qwen3VLTextModel(cfg)
        cls.vocab_size = cfg.vocab_size
        cls.hidden_size = cfg.hidden_size
        cls.num_hidden_layers = cfg.num_hidden_layers

    def _create_test_inputs(self, batch_size=2, seq_len=16):
        """Helper to create test inputs for TextModel."""
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        return input_ids

    def _create_padding_mask(self, batch_size=2, seq_len=16, valid_len=12):
        """
        Create a 2D padding mask with trailing padded positions.
        """
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask[:, :valid_len] = 1
        return attention_mask

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        q_model = QuantQwen3VLTextModel(self.fp_model)
        self.assertIs(q_model._mode, Mode.NO_QUANT)

        q_model.enable_calibration()
        self.assertIs(q_model._mode, Mode.CALIB)

        # Run forward pass during calibration
        input_ids = self._create_test_inputs()
        _ = q_model(input_ids=input_ids, use_cache=False)

        q_model.freeze_qparams()
        self.assertIs(q_model._mode, Mode.QUANT)

    def test_forward_diff(self):
        """
        Test that quantized output is acceptably close to FP32 reference.
        """
        torch.manual_seed(42)
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        # Calibrate with multiple inputs
        for _ in range(4):
            input_ids = self._create_test_inputs()
            _ = q_model(input_ids=input_ids, use_cache=False)

        q_model.freeze_qparams()

        input_ids = self._create_test_inputs()
        with torch.no_grad():
            q_out = q_model(input_ids=input_ids, use_cache=False)
            fp_out = self.fp_model(input_ids=input_ids, use_cache=False)

        # Extract last_hidden_state
        q_hidden = q_out.last_hidden_state
        fp_hidden = fp_out.last_hidden_state

        self.assertEqual(fp_hidden.shape, q_hidden.shape)
        diff = (fp_hidden - q_hidden).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.7)  # acceptably close

    def test_registration_in_registry(self):
        """
        Test that Qwen3VLTextModel is properly registered.
        """
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_model import (
            QuantQwen3VLTextModel,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        wrapper_cls = lookup(Qwen3VLTextModel)
        self.assertIs(wrapper_cls, QuantQwen3VLTextModel)

    def test_output_shape(self):
        """
        Test that output shape is preserved.
        Input: (batch_size, seq_len)
        Output last_hidden_state: (batch_size, seq_len, hidden_size)
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        batch_size = 2
        seq_len = 16
        input_ids = self._create_test_inputs(batch_size, seq_len)
        _ = q_model(input_ids=input_ids, use_cache=False)

        q_model.freeze_qparams()

        with torch.no_grad():
            q_out = q_model(input_ids=input_ids, use_cache=False)
            fp_out = self.fp_model(input_ids=input_ids, use_cache=False)

        q_hidden = q_out.last_hidden_state
        fp_hidden = fp_out.last_hidden_state

        expected_shape = (batch_size, seq_len, self.hidden_size)
        self.assertEqual(q_hidden.shape, expected_shape)
        self.assertEqual(fp_hidden.shape, expected_shape)

    def test_embedding_layer_quantization(self):
        """
        Test that embedding layer is properly wrapped and quantized.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        input_ids = self._create_test_inputs()
        _ = q_model(input_ids=input_ids, use_cache=False)

        q_model.freeze_qparams()

        # Check that embed_tokens is wrapped
        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        self.assertIsInstance(q_model.embed_tokens, PTQWrapper)

    def test_rotary_emb_not_wrapped(self):
        """
        Test that rotary_emb is NOT wrapped (saved as-is).
        """
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
        )

        q_model = QuantQwen3VLTextModel(self.fp_model)

        # rotary_emb should be the original module, not a wrapper
        self.assertIsInstance(q_model.rotary_emb, Qwen3VLTextRotaryEmbedding)

        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        self.assertNotIsInstance(q_model.rotary_emb, PTQWrapper)

    def test_layers_wrapped(self):
        """
        Test that all decoder layers are properly wrapped with PTQWrapper.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)

        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        self.assertEqual(len(q_model.layers), self.num_hidden_layers)
        for layer in q_model.layers:
            self.assertIsInstance(layer, PTQWrapper)

    def test_norm_wrapped(self):
        """
        Test that final normalization layer is properly wrapped.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)

        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        self.assertIsInstance(q_model.norm, PTQWrapper)

    def test_different_batch_sizes(self):
        """
        Test that quantization works correctly with different batch sizes.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        # Calibrate with one batch size
        calibrate_input = self._create_test_inputs(batch_size=2)
        for _ in range(3):
            _ = q_model(input_ids=calibrate_input, use_cache=False)
        q_model.freeze_qparams()

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            input_ids = self._create_test_inputs(batch_size=batch_size)
            with torch.no_grad():
                q_out = q_model(input_ids=input_ids, use_cache=False)
                fp_out = self.fp_model(input_ids=input_ids, use_cache=False)

            q_hidden = q_out.last_hidden_state
            fp_hidden = fp_out.last_hidden_state

            self.assertEqual(q_hidden.shape[0], batch_size)
            self.assertEqual(fp_hidden.shape[0], batch_size)
            diff = (fp_hidden - q_hidden).abs().mean().item()
            self.assertLess(diff, 0.8)

    def test_different_sequence_lengths(self):
        """
        Test that quantization works correctly with different sequence lengths.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        # Calibrate with one sequence length
        calibrate_input = self._create_test_inputs(seq_len=16)
        for _ in range(3):
            _ = q_model(input_ids=calibrate_input, use_cache=False)
        q_model.freeze_qparams()

        # Test with different sequence lengths
        for seq_len in [8, 16, 32]:
            input_ids = self._create_test_inputs(seq_len=seq_len)
            with torch.no_grad():
                q_out = q_model(input_ids=input_ids, use_cache=False)
                fp_out = self.fp_model(input_ids=input_ids, use_cache=False)

            q_hidden = q_out.last_hidden_state
            fp_hidden = fp_out.last_hidden_state

            self.assertEqual(q_hidden.shape[1], seq_len)
            self.assertEqual(fp_hidden.shape[1], seq_len)
            diff = (fp_hidden - q_hidden).abs().mean().item()
            self.assertLess(diff, 0.8)

    def test_observer_count(self):
        """
        Test that the wrapper has the correct number of observers.
        - 5 local observers (embeds, cos, sin, hidden_states, visual_embeds)
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        observers = list(q_model._all_observers())
        # Should have 6 local observers
        self.assertEqual(len(observers), 5 + len(q_model.layers))

    def test_per_module_override(self):
        """
        Test that PTQConfig overrides propagate correctly to submodules.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={"inputs_embeds": {"dtype": DType.uint(4)}},
        )
        q_model = QuantQwen3VLTextModel(self.fp_model, qcfg=cfg)

        # Check that override is applied
        # The embed_tokens weight should have the override
        self.assertEqual(q_model.obs_inputs_embeds.dtype, DType.uint(4))

    def test_deepstack_injection(self):
        """
        Test that DeepStack visual feature injection works correctly.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        batch_size = 2
        seq_len = 16
        input_ids = self._create_test_inputs(batch_size, seq_len)

        # Calibrate without DeepStack
        _ = q_model(input_ids=input_ids, use_cache=False)

        q_model.freeze_qparams()

        # Test with DeepStack features
        visual_pos_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        visual_pos_masks[:, :4] = True  # First 4 positions get visual features

        # Create visual embeddings for first layer only
        visual_embeds = [
            torch.randn(
                8, self.hidden_size
            )  # 8 visual tokens (batch_size * 4 positions)
        ]

        with torch.no_grad():
            q_out = q_model(
                input_ids=input_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=visual_embeds,
                use_cache=False,
            )
            fp_out = self.fp_model(
                input_ids=input_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=visual_embeds,
                use_cache=False,
            )

        q_hidden = q_out.last_hidden_state
        fp_hidden = fp_out.last_hidden_state

        self.assertEqual(q_hidden.shape, fp_hidden.shape)
        # Visual positions should have different values due to DeepStack injection
        self.assertFalse(
            torch.equal(
                q_hidden[visual_pos_masks],
                fp_hidden[visual_pos_masks],
            )
        )

    def test_inputs_embeds_path(self):
        """
        Test that the model works correctly when inputs_embeds are provided instead of input_ids.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        # Calibrate with input_ids
        input_ids = self._create_test_inputs()
        _ = q_model(input_ids=input_ids, use_cache=False)

        q_model.freeze_qparams()

        # Test with inputs_embeds
        batch_size = 2
        seq_len = 16
        inputs_embeds = torch.randn(batch_size, seq_len, self.hidden_size)

        with torch.no_grad():
            q_out = q_model(inputs_embeds=inputs_embeds, use_cache=False)
            fp_out = self.fp_model(inputs_embeds=inputs_embeds, use_cache=False)

        q_hidden = q_out.last_hidden_state
        fp_hidden = fp_out.last_hidden_state

        self.assertEqual(q_hidden.shape, fp_hidden.shape)
        diff = (fp_hidden - q_hidden).abs().mean().item()
        self.assertLess(diff, 1.0)

    def test_no_cache_mode(self):
        """
        Test that the model works correctly with use_cache=False.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        input_ids = self._create_test_inputs()
        _ = q_model(input_ids=input_ids, use_cache=False)

        q_model.freeze_qparams()

        with torch.no_grad():
            q_out = q_model(input_ids=input_ids, use_cache=False)
            fp_out = self.fp_model(input_ids=input_ids, use_cache=False)

        q_hidden = q_out.last_hidden_state
        fp_hidden = fp_out.last_hidden_state

        self.assertEqual(q_hidden.shape, fp_hidden.shape)
        diff = (fp_hidden - q_hidden).abs().mean().item()
        self.assertLess(diff, 0.7)

        # past_key_values should be None when use_cache=False
        self.assertIsNone(q_out.past_key_values)
        self.assertIsNone(fp_out.past_key_values)

    def test_return_dict_false_with_cache(self):
        """
        Test that the wrapper returns a tuple of hidden states and cache when
        return_dict=False and use_cache=True.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        input_ids = self._create_test_inputs(batch_size=1, seq_len=8)
        _ = q_model(input_ids=input_ids, use_cache=False)

        q_model.freeze_qparams()

        with torch.no_grad():
            q_out = q_model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=False,
            )

        self.assertIsInstance(q_out, tuple)
        self.assertEqual(len(q_out), 2)
        self.assertEqual(q_out[0].shape, (1, 8, self.hidden_size))
        self.assertIsNotNone(q_out[1])

    def test_attention_mask_2d_prefill(self):
        """
        Test that a 2D padding mask is accepted and produces the expected output
        shape during prefill.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        input_ids = self._create_test_inputs(batch_size=2, seq_len=16)
        attention_mask = self._create_padding_mask(
            batch_size=2,
            seq_len=16,
            valid_len=12,
        )

        _ = q_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        q_model.freeze_qparams()

        with torch.no_grad():
            q_out = q_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            fp_out = self.fp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        self.assertEqual(q_out.last_hidden_state.shape, fp_out.last_hidden_state.shape)
        diff = (fp_out.last_hidden_state - q_out.last_hidden_state).abs().mean().item()
        self.assertLess(diff, 0.8)

    def test_attention_mask_bool_prefill(self):
        """
        Test that a 2D boolean padding mask is normalized correctly.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        input_ids = self._create_test_inputs(batch_size=2, seq_len=16)
        attention_mask = self._create_padding_mask(
            batch_size=2,
            seq_len=16,
            valid_len=10,
        ).bool()

        _ = q_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        q_model.freeze_qparams()

        with torch.no_grad():
            q_out = q_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        self.assertEqual(
            q_out.last_hidden_state.shape,
            (2, 16, self.hidden_size),
        )

    def test_attention_mask_4d_additive_passthrough(self):
        """
        Test that a 4D additive mask is accepted directly.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        input_ids = self._create_test_inputs(batch_size=1, seq_len=8)
        additive_mask = torch.zeros(1, 1, 8, 8)
        additive_mask[..., :, 6:] = -120.0

        _ = q_model(
            input_ids=input_ids,
            attention_mask=additive_mask,
            use_cache=False,
        )
        q_model.freeze_qparams()

        with torch.no_grad():
            q_out = q_model(
                input_ids=input_ids,
                attention_mask=additive_mask,
                use_cache=False,
            )

        self.assertEqual(q_out.last_hidden_state.shape, (1, 8, self.hidden_size))

    def test_attention_mask_decode_with_cache(self):
        """
        Test that decoding with cache and a 2D attention mask works correctly.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)
        q_model.enable_calibration()

        prefill_ids = self._create_test_inputs(batch_size=1, seq_len=8)
        _ = q_model(input_ids=prefill_ids, use_cache=False)
        q_model.freeze_qparams()

        with torch.no_grad():
            prefill_out = q_model(
                input_ids=prefill_ids,
                use_cache=True,
            )

        decode_ids = self._create_test_inputs(batch_size=1, seq_len=1)
        decode_attention_mask = torch.ones(1, 9, dtype=torch.long)

        with torch.no_grad():
            decode_out = q_model(
                input_ids=decode_ids,
                attention_mask=decode_attention_mask,
                past_key_values=prefill_out.past_key_values,
                use_cache=True,
            )

        self.assertEqual(decode_out.last_hidden_state.shape, (1, 1, self.hidden_size))
        self.assertIsNotNone(decode_out.past_key_values)

    def test_normalize_attention_mask_shapes(self):
        """
        Test that the normalized additive mask has the expected shape for
        prefill and decode.
        """
        q_model = QuantQwen3VLTextModel(self.fp_model)

        prefill_embeds = torch.randn(2, 6, self.hidden_size)

        mask_prefill = q_model._normalize_attention_mask(
            attention_mask=None,
            input_embeds=prefill_embeds,
            past_key_values=None,
        )
        self.assertEqual(mask_prefill.shape, (1, 1, 6, 6))

        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=q_model.config)
        dummy_k = torch.randn(
            2,
            q_model.config.num_key_value_heads,
            4,
            q_model.config.head_dim,
        )
        dummy_v = torch.randn(
            2,
            q_model.config.num_key_value_heads,
            4,
            q_model.config.head_dim,
        )
        cache.update(dummy_k, dummy_v, 0)

        decode_embeds = torch.randn(2, 1, self.hidden_size)
        mask_decode = q_model._normalize_attention_mask(
            attention_mask=torch.ones(2, 5, dtype=torch.long),
            input_embeds=decode_embeds,
            past_key_values=cache,
        )
        self.assertEqual(mask_decode.shape, (2, 1, 1, 5))
