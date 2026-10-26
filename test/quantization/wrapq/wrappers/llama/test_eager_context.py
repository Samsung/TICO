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

import io
import math
import unittest
from typing import Any
from unittest.mock import patch

import torch

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)
from tico.quantization.wrapq.wrappers.llama.quant_model import QuantLlamaModel

from test.quantization.quant_spec_helpers import make_affine_ptq_config


@unittest.skipUnless(has_transformers_for("llama"), "Llama support is required")
class TestLlamaEagerContext(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def _model(self, max_positions=32, *, scaled_rope=False):
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel

        kwargs: dict[str, Any] = {}
        if scaled_rope:
            kwargs["rope_scaling"] = {"rope_type": "linear", "factor": 2.0}
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=max_positions,
            attention_dropout=0.0,
            **kwargs,
        )
        config._attn_implementation = "eager"
        return LlamaModel(config).eval()

    def _wrapper(self, model, profile="reference_eval"):
        return QuantLlamaModel(
            model,
            qcfg=make_affine_ptq_config(
                dtype=DType.int(16), model_args={"profile": profile}
            ),
        ).eval()

    def test_native_context_does_not_allocate_native_masks_or_rope_tables(self):
        """A large native context must not allocate a quadratic mask at prepare time."""
        native_length = 131072
        model = self._model(native_length)
        original_full = torch.full

        def guarded_full(size, *args, **kwargs):
            self.assertLess(
                math.prod(size),
                1000000,
                "Wrapping must not allocate a native-context attention mask",
            )
            return original_full(size, *args, **kwargs)

        with patch("torch.full", side_effect=guarded_full):
            wrapped = self._wrapper(model)

        self.assertEqual(wrapped.config.max_position_embeddings, native_length)
        for name, buffer in wrapped.named_buffers():
            self.assertNotIn(native_length, buffer.shape, name)
            self.assertLess(buffer.numel(), 1000000, name)

        inputs = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            expected = model(inputs, use_cache=False).last_hidden_state
            actual = wrapped(inputs, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_scaled_rope_and_batched_position_ids_match_pretrained_model(self):
        """Both profiles must preserve scaled RoPE and caller-provided positions."""
        model = self._model(scaled_rope=True)
        inputs = torch.randint(0, 64, (2, 12))
        positions = torch.stack((torch.arange(12), torch.arange(12) + 3))
        attention_mask = torch.ones_like(inputs)
        attention_mask[1, :2] = 0
        with torch.no_grad():
            expected = model(
                inputs,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=False,
            ).last_hidden_state
            for profile in ("reference_eval", "npu_export"):
                with self.subTest(profile=profile):
                    actual = self._wrapper(model, profile)(
                        inputs,
                        attention_mask=attention_mask,
                        position_ids=positions,
                        use_cache=False,
                    ).last_hidden_state
                    # All-masked padding queries are outside the scored token set.
                    torch.testing.assert_close(
                        actual[attention_mask.bool()],
                        expected[attention_mask.bool()],
                        atol=2e-6,
                        rtol=2e-5,
                    )

    def test_prefill_decode_matches_full_pretrained_forward(self):
        """Growing eager caches must use the actual key length and token position."""
        model = self._model(scaled_rope=True)
        inputs = torch.randint(0, 64, (1, 11))
        with torch.no_grad():
            expected = model(inputs, use_cache=False).last_hidden_state[:, -1:]
            for profile in ("reference_eval", "npu_export"):
                with self.subTest(profile=profile):
                    wrapped = self._wrapper(model, profile)
                    prefill = wrapped(inputs[:, :7], use_cache=True)
                    cache = prefill.past_key_values
                    for index in range(7, inputs.shape[1]):
                        result = wrapped(
                            inputs[:, index : index + 1],
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cache = result.past_key_values
                    self.assertEqual(cache.get_seq_length(), inputs.shape[1])
                    torch.testing.assert_close(
                        result.last_hidden_state, expected, atol=2e-6, rtol=2e-5
                    )

    def test_standalone_eager_layer_uses_pretrained_rotary_state(self):
        """Standalone eager calls must preserve the resolved pretrained rotary state."""
        model = self._model()
        model.rotary_emb.inv_freq.mul_(0.5)
        wrapped = self._wrapper(model)
        positions = torch.arange(9).unsqueeze(0)
        hidden = torch.zeros(1, 9, 16)
        expected = model.rotary_emb(hidden, positions)
        layer = wrapped.layers[0].wrapped
        actual = layer._slice_rope(
            start=0, seq_len=9, device=torch.device("cpu"), dtype=torch.float32
        )
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_longer_evaluation_keeps_frozen_qparams(self):
        """Short calibration must not become an eager sequence limit or recalibrate."""
        wrapped = self._wrapper(self._model())
        wrapped.enable_calibration()
        with torch.no_grad():
            prefill = wrapped(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
            wrapped(
                torch.tensor([[5]]),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
        wrapped.freeze_qparams()
        snapshots = {
            name: {key: value.clone() for key, value in observer.named_buffers()}
            for name, observer in wrapped.named_observers()
        }

        with torch.no_grad():
            result = wrapped(torch.arange(12).unsqueeze(0), use_cache=False)
        self.assertEqual(tuple(result.last_hidden_state.shape), (1, 12, 16))
        self.assertTrue(torch.isfinite(result.last_hidden_state).all())
        self.assertEqual(wrapped.config.max_position_embeddings, 32)
        self.assertIs(wrapped._mode, Mode.QUANT)
        for name, observer in wrapped.named_observers():
            self.assertFalse(observer.enabled, name)
            for key, value in observer.named_buffers():
                torch.testing.assert_close(
                    value, snapshots[name][key], atol=0, rtol=0, equal_nan=True
                )

    def test_explicit_static_shapes_preserve_prefill_and_decode(self):
        """Fixed export inputs must retain the eager result at valid positions."""
        model = self._model(scaled_rope=True)
        layer = QuantLlamaDecoderLayer(
            model.layers[0],
            qcfg=make_affine_ptq_config(model_args={"profile": "npu_export"}),
            layer_idx=0,
            return_type="tuple",
        ).eval()
        static_length = 8
        valid_length = 3
        hidden = torch.randn(1, static_length, 16)
        rope = layer._slice_rope(
            start=0,
            seq_len=static_length,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        mask = layer._build_causal_mask(
            q_len=static_length,
            k_len=static_length,
            past_len=0,
            device=torch.device("cpu"),
        )
        mask[:, :, valid_length:] = -120.0
        prefill = layer.as_export_module("prefill", return_kv=True)
        with torch.no_grad():
            expected, expected_cache = layer(hidden[:, :valid_length], use_cache=True)
            actual, keys, values = prefill(hidden, mask, rope)
        torch.testing.assert_close(
            actual[:, :valid_length], expected, atol=1e-6, rtol=1e-5
        )

        past_key = torch.zeros(1, 1, static_length - 1, 8)
        past_value = torch.zeros_like(past_key)
        past_key[:, :, :valid_length] = keys[:, :, :valid_length]
        past_value[:, :, :valid_length] = values[:, :, :valid_length]
        next_hidden = torch.randn(1, 1, 16)
        next_rope = layer._slice_rope(
            start=valid_length,
            seq_len=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        next_mask = torch.full((1, 1, static_length), -120.0)
        next_mask[:, :, :valid_length] = 0
        next_mask[:, :, -1] = 0
        decode = layer.as_export_module("decode", return_kv=True)
        with torch.no_grad():
            expected_decode, _ = layer(
                next_hidden,
                past_key_value=expected_cache,
                use_cache=True,
            )
            actual_decode = decode(
                next_hidden, next_mask, next_rope, (past_key, past_value)
            )
        torch.testing.assert_close(
            actual_decode[0], expected_decode, atol=1e-6, rtol=1e-5
        )
        self.assertEqual(actual_decode[1].shape[2], 1)

        exported = torch.export.export(
            decode, (next_hidden, next_mask, next_rope, (past_key, past_value))
        )
        exported_result = exported.module()(
            next_hidden, next_mask, next_rope, (past_key, past_value)
        )
        for actual_tensor, expected_tensor in zip(exported_result, actual_decode):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_legacy_checkpoint_rebuilds_rope_without_changing_recorded_limit(self):
        """Legacy checkpoints can rebuild RoPE from their preserved configuration."""
        model = self._model(16, scaled_rope=True)
        wrapped = self._wrapper(model)
        del wrapped.rotary_emb
        checkpoint = io.BytesIO()
        torch.save(wrapped, checkpoint)
        checkpoint.seek(0)
        loaded = torch.load(checkpoint, weights_only=False)
        inputs = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            actual = loaded(inputs, use_cache=False).last_hidden_state
            expected = model(inputs, use_cache=False).last_hidden_state
        self.assertEqual(loaded.config.max_position_embeddings, 16)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
