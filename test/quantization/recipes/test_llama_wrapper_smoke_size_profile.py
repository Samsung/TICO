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

"""Tests for Llama wrapper-smoke size profiles."""

import unittest

from tico.quantization.recipes.debug.wrapper_smoke.cases.llama import (
    _has_llama,
    LlamaAttentionDecodeCase,
    LlamaAttentionPrefillCase,
    LlamaDecoderLayerDecodeCase,
    LlamaDecoderLayerPrefillCase,
    LlamaMLPCase,
    LlamaStaticRuntimeShape,
)


def _profile_cfg(profile: str, *, max_seq: int = 2_048) -> dict:
    """Build a minimal wrapper-smoke config for one Llama profile."""
    return {
        "debug": {
            "wrapper_smoke": {
                "llama": {
                    "size_profile": profile,
                    "static_runtime": {"max_seq": max_seq},
                }
            }
        }
    }


_LLAMA_CASE_TYPES = (
    LlamaMLPCase,
    LlamaAttentionPrefillCase,
    LlamaAttentionDecodeCase,
    LlamaDecoderLayerPrefillCase,
    LlamaDecoderLayerDecodeCase,
)
_LLAMA_AVAILABILITY = _has_llama()


class TestLlamaWrapperSmokeProfileValidation(unittest.TestCase):
    """Validate profile selection without allocating target-width modules."""

    def test_all_bounded_cases_accept_both_llama3_2_3b_profiles(self):
        """Every existing Llama module case should support both large profiles."""
        for case_type in _LLAMA_CASE_TYPES:
            for profile in (
                "llama3_2_3b_dims",
                "llama3_2_3b_static_runtime",
            ):
                with self.subTest(case=case_type.__name__, profile=profile):
                    case_type().validate_config(_profile_cfg(profile))

    def test_unknown_profile_is_rejected(self):
        """Profile typos should not silently fall back to tiny dimensions."""
        with self.assertRaisesRegex(ValueError, "Unsupported Llama"):
            LlamaMLPCase().validate_config(_profile_cfg("llama_3b"))

    def test_static_shape_requires_a_decodeable_capacity(self):
        """Static decode needs one current token and at least one cache slot."""
        self.assertEqual(LlamaStaticRuntimeShape().max_seq, 2_048)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            LlamaStaticRuntimeShape(max_seq=1)

    def test_static_shape_override_is_cached_by_the_case(self):
        """The configured runtime capacity should drive prefill and decode shapes."""
        case = LlamaAttentionDecodeCase()
        case.validate_config(_profile_cfg("llama3_2_3b_static_runtime", max_seq=1_024))
        self.assertEqual(case._decode_max_seq(16), 1_024)
        self.assertEqual(case._prefill_seq_len(16), 1_024)

    def test_profile_specific_circle_filenames_are_distinct(self):
        """Large-profile artifacts should not overwrite tiny artifacts."""
        case = LlamaMLPCase()
        self.assertEqual(case.export_filename({}), "llama_mlp.q.circle")
        self.assertEqual(
            case.export_filename(_profile_cfg("llama3_2_3b_dims")),
            "llama_mlp.llama3_2_3b_dims.q.circle",
        )
        self.assertEqual(
            case.export_filename(_profile_cfg("llama3_2_3b_static_runtime")),
            "llama_mlp.llama3_2_3b_static_runtime.q.circle",
        )


@unittest.skipUnless(
    _LLAMA_AVAILABILITY.available,
    _LLAMA_AVAILABILITY.reason or "Llama is unavailable",
)
class TestLlamaWrapperSmokeConfigDimensions(unittest.TestCase):
    """Validate tiny compatibility and Llama 3.2-3B target dimensions."""

    def test_tiny_remains_the_default_profile(self):
        """An omitted profile should preserve the existing tiny dimensions."""
        config = LlamaMLPCase()._make_config({}, tiny_max_seq=16)
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(config.intermediate_size, 32)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.num_key_value_heads, 1)
        self.assertEqual(config.head_dim, 8)
        self.assertEqual(config.max_position_embeddings, 16)

    def test_dims_profile_uses_llama3_2_3b_widths_and_one_layer(self):
        """The dimensions profile should avoid constructing all 28 layers."""
        config = LlamaMLPCase()._make_config(
            _profile_cfg("llama3_2_3b_dims"), tiny_max_seq=16
        )
        self.assertEqual(config.vocab_size, 128_256)
        self.assertEqual(config.hidden_size, 3_072)
        self.assertEqual(config.intermediate_size, 8_192)
        self.assertEqual(config.num_hidden_layers, 1)
        self.assertEqual(config.num_attention_heads, 24)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.max_position_embeddings, 16)
        self.assertEqual(config.rms_norm_eps, 1e-5)

    def test_static_profile_uses_runtime_context_capacity(self):
        """Static wrappers should allocate masks and caches for 2,048 tokens."""
        config = LlamaAttentionDecodeCase()._make_config(
            _profile_cfg("llama3_2_3b_static_runtime"), tiny_max_seq=16
        )
        self.assertEqual(config.hidden_size, 3_072)
        self.assertEqual(config.max_position_embeddings, 2_048)


if __name__ == "__main__":
    unittest.main()
