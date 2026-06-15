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

import inspect
import unittest

import torch
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLTextDecoderLayerDecodeExportAdapter,
    Qwen3VLTextDecoderLayerPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_decoder_layer import (
    QuantQwen3VLTextDecoderLayer,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


skip_msg = (
    "required transformers not installed — skipping Qwen3VLTextDecoderLayer tests"
)


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLTextDecoderLayer(unittest.TestCase):
    fp_model: torch.nn.Module
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        # Use smaller sizes for testing.
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
        cls.num_key_value_heads = cfg.num_key_value_heads
        cls.head_dim = cfg.head_dim

    def _rand_position_embeddings(self, batch_size, seq_len):
        """Create dummy rotary position embeddings."""
        cos = torch.randn(batch_size, seq_len, self.head_dim)
        sin = torch.randn(batch_size, seq_len, self.head_dim)
        return cos, sin

    def _create_test_inputs(self, batch_size=2, seq_len=16):
        """Create synthetic inputs for Qwen3-VL text decoder-layer tests."""
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)
        position_embeddings = self._rand_position_embeddings(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        return hidden_states, position_embeddings, attention_mask, position_ids

    def _create_decode_inputs(self, batch_size=2, past_len=4):
        """Create synthetic single-token decode inputs and static tuple caches."""
        hidden_states = torch.randn(batch_size, 1, self.hidden_size)
        position_embeddings = self._rand_position_embeddings(batch_size, 1)
        attention_mask = torch.zeros(batch_size, 1, 1, past_len + 1)
        position_ids = torch.full((batch_size, 1), past_len, dtype=torch.long)
        past_key = torch.randn(
            batch_size,
            self.num_key_value_heads,
            past_len,
            self.head_dim,
        )
        past_value = torch.randn_like(past_key)
        return (
            hidden_states,
            position_embeddings,
            attention_mask,
            position_ids,
            (
                past_key,
                past_value,
            ),
        )

    def _collect_cache_calibration(self, q_model: QuantQwen3VLTextDecoderLayer) -> None:
        """Collect observer statistics for the static KV-cache path."""
        hidden_states, pos_emb, attn_mask, pos_ids, past = self._create_decode_inputs(
            batch_size=2,
            past_len=2,
        )
        _ = q_model(
            hidden_states=hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=past,
            use_cache=True,
            cache_output_mode="present",
        )

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT -> CALIB -> QUANT."""
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        self.assertIs(q_model._mode, Mode.NO_QUANT)

        q_model.enable_calibration()
        self.assertIs(q_model._mode, Mode.CALIB)

        # Run forward pass during calibration.
        hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs()
        _ = q_model(
            hidden_states=hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
        )

        self._collect_cache_calibration(q_model)
        q_model.freeze_qparams()
        self.assertIs(q_model._mode, Mode.QUANT)

    def test_forward_diff(self):
        """Test that quantized output is acceptably close to the FP32 reference."""
        torch.manual_seed(42)
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        q_model.enable_calibration()

        # Calibrate with multiple inputs.
        for _ in range(4):
            hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs()
            _ = q_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )

        self._collect_cache_calibration(q_model)
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
        """Test that Qwen3VLTextDecoderLayer is properly registered."""
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        wrapper_cls = lookup(Qwen3VLTextDecoderLayer)
        self.assertIs(wrapper_cls, QuantQwen3VLTextDecoderLayer)

    def test_output_shape(self):
        """Test that the decoder layer preserves hidden-state shape."""
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

        self._collect_cache_calibration(q_model)
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

    def test_public_cache_argument_is_past_key_values(self):
        """Verify that decoder-layer cache APIs use only the plural argument."""
        layer_params = inspect.signature(
            QuantQwen3VLTextDecoderLayer.forward
        ).parameters
        self.assertIn("past_key_values", layer_params)
        self.assertNotIn("past_key_value", layer_params)

        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        decode_adapter = q_model.as_export_module("decode")
        adapter_params = inspect.signature(decode_adapter.forward).parameters
        self.assertIn("past_key_values", adapter_params)
        self.assertNotIn("past_key_value", adapter_params)

    def test_forward_with_cache_returns_delta_kv_tuple(self):
        """Validate decoder-layer delta KV output in tuple-return mode."""
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model, return_type="tuple")
        q_model.enable_calibration()

        hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs(
            batch_size=2,
            seq_len=4,
        )
        _ = q_model(
            hidden_states=hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=True,
            cache_output_mode="delta",
        )
        self._collect_cache_calibration(q_model)
        q_model.freeze_qparams()

        with torch.no_grad():
            outputs = q_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=True,
                cache_output_mode="delta",
            )

        hidden_out, cache_delta = outputs
        new_k, new_v = cache_delta
        self.assertEqual(hidden_out.shape, (2, 4, self.hidden_size))
        self.assertEqual(new_k.shape, (2, self.num_key_value_heads, 4, self.head_dim))
        self.assertEqual(new_v.shape, (2, self.num_key_value_heads, 4, self.head_dim))

    def test_decode_with_static_tuple_cache_returns_delta_kv(self):
        """Validate static tuple-cache decode with delta-only cache output."""
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model, return_type="tuple")
        hidden_states, pos_emb, attn_mask, pos_ids, past = self._create_decode_inputs(
            batch_size=2,
            past_len=3,
        )

        q_model.enable_calibration()
        _ = q_model(
            hidden_states=hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=past,
            use_cache=True,
            cache_output_mode="delta",
        )
        self._collect_cache_calibration(q_model)
        q_model.freeze_qparams()

        with torch.no_grad():
            outputs = q_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=past,
                use_cache=True,
                cache_output_mode="delta",
            )

        hidden_out, cache_delta = outputs
        new_k, new_v = cache_delta
        self.assertEqual(hidden_out.shape, (2, 1, self.hidden_size))
        self.assertEqual(new_k.shape, (2, self.num_key_value_heads, 1, self.head_dim))
        self.assertEqual(new_v.shape, (2, self.num_key_value_heads, 1, self.head_dim))

    def test_as_export_module_prefill_and_decode_contracts(self):
        """Validate the static runtime adapter input and output contracts."""
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)

        prefill_adapter = q_model.as_export_module("prefill", return_kv=True)
        self.assertIsInstance(
            prefill_adapter,
            Qwen3VLTextDecoderLayerPrefillExportAdapter,
        )

        B, S = 1, 4
        hidden_states = torch.randn(B, S, self.hidden_size)
        pos_emb = self._rand_position_embeddings(B, S)
        attn_mask = torch.zeros(B, 1, S, S)

        with torch.no_grad():
            hidden_out, new_k, new_v = prefill_adapter(
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=pos_emb,
            )

        self.assertEqual(hidden_out.shape, (B, S, self.hidden_size))
        self.assertEqual(new_k.shape, (B, self.num_key_value_heads, S, self.head_dim))
        self.assertEqual(new_v.shape, (B, self.num_key_value_heads, S, self.head_dim))

        decode_adapter = q_model.as_export_module("decode", return_kv=True)
        self.assertIsInstance(
            decode_adapter,
            Qwen3VLTextDecoderLayerDecodeExportAdapter,
        )

        hidden_decode = torch.randn(B, 1, self.hidden_size)
        pos_decode = self._rand_position_embeddings(B, 1)
        attn_decode = torch.zeros(B, 1, 1, S + 1)

        with torch.no_grad():
            hidden_next, delta_k, delta_v = decode_adapter(
                hidden_states=hidden_decode,
                attention_mask=attn_decode,
                position_embeddings=pos_decode,
                past_key_values=(new_k, new_v),
            )

        self.assertEqual(hidden_next.shape, (B, 1, self.hidden_size))
        self.assertEqual(delta_k.shape, (B, self.num_key_value_heads, 1, self.head_dim))
        self.assertEqual(delta_v.shape, (B, self.num_key_value_heads, 1, self.head_dim))

    def test_as_export_module_return_kv_false_returns_hidden_only(self):
        """Validate hidden-only adapter output when KV return is disabled."""
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        prefill_adapter = q_model.as_export_module("prefill", return_kv=False)

        B, S = 1, 3
        hidden_states = torch.randn(B, S, self.hidden_size)
        pos_emb = self._rand_position_embeddings(B, S)
        attn_mask = torch.zeros(B, 1, S, S)

        with torch.no_grad():
            hidden_out = prefill_adapter(
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=pos_emb,
            )

        self.assertIsInstance(hidden_out, torch.Tensor)
        self.assertEqual(hidden_out.shape, (B, S, self.hidden_size))

    def test_as_export_module_rejects_unknown_mode(self):
        """Reject unsupported export adapter modes."""
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        with self.assertRaises(ValueError):
            _ = q_model.as_export_module("train")  # type: ignore[arg-type]

    def test_residual_connection_preservation(self):
        """Test that residual connections preserve shape and transform values."""
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        q_model.enable_calibration()

        hidden_states, pos_emb, attn_mask, pos_ids = self._create_test_inputs()
        _ = q_model(
            hidden_states=hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
        )

        self._collect_cache_calibration(q_model)
        q_model.freeze_qparams()

        with torch.no_grad():
            input_copy = hidden_states.clone()
            output = q_model(
                hidden_states=hidden_states,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=pos_ids,
            )

        # Output should be different from input because transformations are applied.
        self.assertFalse(torch.equal(output, input_copy))

        # Shape should be preserved.
        self.assertEqual(output.shape, input_copy.shape)

    def test_observer_count(self):
        """
        Test that the wrapper has the correct number of local observers.

        The attention and MLP submodules own their own observers.
        """
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        observers = list(q_model._all_observers())
        self.assertEqual(len(observers), 3)

    def test_per_module_override(self):
        """Test that PTQConfig overrides propagate correctly to submodules."""
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "self_attn": {
                    "act_in": {"dtype": DType.uint(4)},
                }
            },
        )
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model, qcfg=cfg)

        # The local input observer keeps the top-level dtype override.
        self.assertEqual(q_model.obs_act_in.dtype, DType.uint(8))

    def test_different_batch_sizes(self):
        """Test that quantization works correctly with different batch sizes."""
        q_model = QuantQwen3VLTextDecoderLayer(self.fp_model)
        q_model.enable_calibration()

        # Calibrate with one batch size.
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
        self._collect_cache_calibration(q_model)
        q_model.freeze_qparams()

        # Test with different batch sizes.
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
