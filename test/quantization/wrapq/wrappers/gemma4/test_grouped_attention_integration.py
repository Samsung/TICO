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

"""Cached integration regressions shared by the two grouped attention variants."""

import copy
import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
    QuantGemma4TextModel,
)

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _has_gemma4,
    _make_text_config,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import (
    _make_text_config as _make_model_config,
    _make_text_model,
)


@unittest.skipUnless(
    _has_gemma4(), "required transformers Gemma4 modules are not installed"
)
class TestGroupedAttentionIntegration(unittest.TestCase):
    """Check cache offsets and head order through the actual wrapper/model."""

    def test_batch_two_append_prefill_padding_and_sliding(self):
        """Appending multiple tokens retains padding, causality and cache order."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(705)
        for layer_type, equal_kv in (
            ("full_attention", False),
            ("sliding_attention", False),
            ("full_attention", True),
        ):
            with self.subTest(layer_type=layer_type, equal_kv=equal_kv):
                config = _make_text_config(
                    num_attention_heads=24,
                    num_key_value_heads=8,
                    num_global_key_value_heads=8,
                    hidden_size=96,
                    num_hidden_layers=2,
                    layer_types=[layer_type, "full_attention"],
                    sliding_window=3,
                    attention_k_eq_v=equal_kv,
                )
                fp_attention = Gemma4TextAttention(config, layer_idx=0).eval()
                attentions = [
                    QuantGemma4TextAttention(
                        copy.deepcopy(fp_attention),
                        qcfg=PTQConfig(
                            model_args={"attention": {"layout": layout, "rope": "hf"}}
                        ),
                    ).eval()
                    for layout in ("batched", "unrolled")
                ]
                past = (
                    torch.randn(2, 8, 4, 4),
                    torch.randn(2, 8, 4, 4),
                )
                hidden = torch.randn(2, 3, 96)
                angles = torch.randn(2, 3, 4)
                padding = torch.ones(2, 7, dtype=torch.bool)
                padding[0, 1] = False
                padding[1, 2] = False
                outputs = []
                with torch.no_grad():
                    for attention in attentions:
                        outputs.append(
                            attention(
                                hidden,
                                (angles.cos(), angles.sin()),
                                attention_mask=padding,
                                past_key_value=past,
                                use_cache=True,
                                cache_output_mode="present",
                            )
                        )
                for actual, expected in zip(outputs[1][:2], outputs[0][:2]):
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                for actual, expected, old in zip(outputs[1][2], outputs[0][2], past):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(actual[:, :, :4], old, atol=0, rtol=0)
                    self.assertEqual(actual.shape, (2, 8, 7, 4))
                weights = outputs[1][1]
                self.assertEqual(weights.shape, (2, 24, 3, 7))
                for batch, pad_position in ((0, 1), (1, 2)):
                    self.assertTrue(
                        torch.equal(
                            weights[batch, :, :, pad_position], torch.zeros(24, 3)
                        )
                    )
                self.assertTrue(
                    torch.equal(weights[:, :, 0, 5:], torch.zeros(2, 24, 2))
                )

    def test_tiny_model_prefill_then_three_decode_steps(self):
        """Actual sliding/shared-KV model execution matches the same FP weights."""
        torch.manual_seed(706)
        config = _make_model_config(
            num_attention_heads=24,
            num_key_value_heads=8,
            num_global_key_value_heads=8,
            hidden_size=96,
            layer_types=["sliding_attention", "full_attention"] * 2,
            sliding_window=4,
            num_kv_shared_layers=2,
            max_position_embeddings=32,
        )
        fp_model = _make_text_model(config)
        quant_model = QuantGemma4TextModel(copy.deepcopy(fp_model)).eval()
        ids = torch.randint(1, config.vocab_size, (1, 8))
        actual_cache = expected_cache = None
        start = 0
        for end in (5, 6, 7, 8):
            with self.subTest(end=end), torch.no_grad():
                kwargs = {
                    "input_ids": ids[:, start:end],
                    "attention_mask": torch.ones_like(ids[:, :end]),
                    "use_cache": True,
                    "return_shared_kv_states": True,
                }
                actual = quant_model(past_key_values=actual_cache, **kwargs)
                expected = fp_model(past_key_values=expected_cache, **kwargs)
                torch.testing.assert_close(
                    actual.last_hidden_state,
                    expected.last_hidden_state,
                    atol=1e-5,
                    rtol=1e-5,
                )
                for layer_type in set(config.layer_types):
                    for actual_kv, expected_kv in zip(
                        actual.shared_kv_states[layer_type],
                        expected.shared_kv_states[layer_type],
                    ):
                        torch.testing.assert_close(
                            actual_kv, expected_kv, atol=1e-5, rtol=1e-5
                        )
                actual_cache = actual.past_key_values
                expected_cache = expected.past_key_values
                self.assertEqual(actual_cache.get_seq_length(), end)
            start = end


if __name__ == "__main__":
    unittest.main()
