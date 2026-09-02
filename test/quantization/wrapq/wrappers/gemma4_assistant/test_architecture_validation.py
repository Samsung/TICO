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

"""Tests for the Gemma4 assistant architecture validator."""

import unittest
from types import SimpleNamespace

from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
    assistant_layer_type_head_dim,
    validate_gemma4_assistant_architecture,
)


def _make_text_config(**overrides) -> SimpleNamespace:
    """Create a duck-typed assistant text config with valid defaults."""
    values = dict(
        vocab_size=64,
        hidden_size=32,
        head_dim=16,
        global_head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        enable_moe_block=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_config(text_overrides=None, **overrides) -> SimpleNamespace:
    """Create a duck-typed assistant config with valid defaults."""
    values = dict(
        text_config=_make_text_config(**(text_overrides or {})),
        backbone_hidden_size=24,
        use_ordered_embeddings=True,
        num_centroids=8,
        centroid_intermediate_top_k=2,
        tie_word_embeddings=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestGemma4AssistantArchitectureValidation(unittest.TestCase):
    """Reject unsupported assistant architectures with actionable errors."""

    def test_valid_all_shared_kv_assistant_is_accepted(self):
        validate_gemma4_assistant_architecture(_make_config())

    def test_moe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "enable_moe_block"):
            validate_gemma4_assistant_architecture(
                _make_config(text_overrides={"enable_moe_block": True})
            )

    def test_nonzero_ple_dim_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hidden_size_per_layer_input"):
            validate_gemma4_assistant_architecture(
                _make_config(text_overrides={"hidden_size_per_layer_input": 64})
            )

    def test_nonzero_ple_vocab_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vocab_size_per_layer_input"):
            validate_gemma4_assistant_architecture(
                _make_config(text_overrides={"vocab_size_per_layer_input": 64})
            )

    def test_partially_shared_layers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_kv_shared_layers"):
            validate_gemma4_assistant_architecture(
                _make_config(text_overrides={"num_kv_shared_layers": 1})
            )

    def test_unsupported_layer_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "layer types"):
            validate_gemma4_assistant_architecture(
                _make_config(
                    text_overrides={
                        "layer_types": ["chunked_attention", "full_attention"]
                    }
                )
            )

    def test_invalid_centroid_divisibility_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible by num_centroids"):
            validate_gemma4_assistant_architecture(_make_config(num_centroids=7))

    def test_invalid_centroid_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "centroid_intermediate_top_k"):
            validate_gemma4_assistant_architecture(
                _make_config(centroid_intermediate_top_k=9)
            )

    def test_missing_sliding_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sliding_window"):
            validate_gemma4_assistant_architecture(
                _make_config(text_overrides={"sliding_window": 0})
            )

    def test_layer_type_head_dims(self):
        text_config = _make_text_config()
        self.assertEqual(
            assistant_layer_type_head_dim(text_config, "full_attention"), 32
        )
        self.assertEqual(
            assistant_layer_type_head_dim(text_config, "sliding_attention"), 16
        )
        with self.assertRaisesRegex(ValueError, "layer type"):
            assistant_layer_type_head_dim(text_config, "chunked_attention")


if __name__ == "__main__":
    unittest.main()
