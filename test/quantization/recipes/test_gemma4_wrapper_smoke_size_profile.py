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

"""Tests for Gemma4 wrapper-smoke size profiles."""

import unittest

from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    _has_gemma4,
    Gemma4TextAttentionCase,
    Gemma4TextMLPCase,
    Gemma4TextModelCase,
    Gemma4TextScaledWordEmbeddingCase,
    Gemma4VisionAttentionCase,
    Gemma4VisionEncoderCase,
    Gemma4VisionModelCase,
    Gemma4VisionPatchEmbedderCase,
)


def _profile_cfg(profile: str) -> dict:
    """Build a minimal wrapper-smoke config for one size profile."""
    return {
        "debug": {
            "wrapper_smoke": {
                "gemma4": {
                    "size_profile": profile,
                }
            }
        }
    }


_GEMMA4_AVAILABILITY = _has_gemma4()


@unittest.skipUnless(
    _GEMMA4_AVAILABILITY.available,
    _GEMMA4_AVAILABILITY.reason or "Gemma4 is unavailable",
)
class TestGemma4WrapperSmokeSizeProfile(unittest.TestCase):
    """Validate tiny compatibility and bounded E2B-width configurations."""

    def test_tiny_remains_the_default_profile(self):
        """An omitted profile should preserve the current tiny dimensions."""
        text_cfg = Gemma4TextMLPCase()._make_text_config({})
        vision_cfg = Gemma4VisionAttentionCase()._make_vision_config({})

        self.assertEqual(text_cfg.hidden_size, 64)
        self.assertEqual(text_cfg.intermediate_size, 128)
        self.assertEqual(vision_cfg.hidden_size, 32)
        self.assertEqual(vision_cfg.intermediate_size, 64)

    def test_e2b_text_dimensions_preserve_case_topology(self):
        """Text cases should use E2B widths without constructing 35 layers."""
        cfg = _profile_cfg("e2b_dims")
        text_cfg = Gemma4TextAttentionCase()._make_text_config(
            cfg,
            layer_types=("sliding_attention", "full_attention"),
        )

        self.assertEqual(text_cfg.hidden_size, 1_536)
        self.assertEqual(text_cfg.intermediate_size, 6_144)
        self.assertEqual(text_cfg.num_attention_heads, 8)
        self.assertEqual(text_cfg.num_key_value_heads, 1)
        self.assertEqual(text_cfg.head_dim, 256)
        self.assertEqual(text_cfg.global_head_dim, 512)
        self.assertEqual(text_cfg.num_hidden_layers, 2)
        self.assertEqual(
            text_cfg.layer_types,
            ["sliding_attention", "full_attention"],
        )
        self.assertEqual(text_cfg.hidden_size_per_layer_input, 0)

    def test_e2b_vision_dimensions_keep_one_layer(self):
        """Vision module cases should use E2B widths and one smoke layer."""
        vision_cfg = Gemma4VisionAttentionCase()._make_vision_config(
            _profile_cfg("e2b_dims")
        )

        self.assertEqual(vision_cfg.hidden_size, 768)
        self.assertEqual(vision_cfg.intermediate_size, 3_072)
        self.assertEqual(vision_cfg.num_attention_heads, 12)
        self.assertEqual(vision_cfg.num_key_value_heads, 12)
        self.assertEqual(vision_cfg.head_dim, 64)
        self.assertEqual(vision_cfg.num_hidden_layers, 1)
        self.assertEqual(vision_cfg.patch_size, 16)
        self.assertEqual(vision_cfg.position_embedding_size, 10_240)

    def test_e2b_patch_embedder_uses_original_patch_dimensions(self):
        """Patch embedding should use original patch and output widths."""
        vision_cfg = Gemma4VisionPatchEmbedderCase()._make_vision_patch_embedder_config(
            _profile_cfg("e2b_dims")
        )

        self.assertEqual(vision_cfg.hidden_size, 768)
        self.assertEqual(vision_cfg.patch_size, 16)
        self.assertEqual(vision_cfg.position_embedding_size, 10_240)

    def test_bounded_vision_composites_accept_e2b_dims(self):
        """One-layer vision composites should support original-width export."""
        cfg = _profile_cfg("e2b_dims")
        Gemma4VisionEncoderCase().validate_config(cfg)
        Gemma4VisionModelCase().validate_config(cfg)

    def test_unsupported_composite_case_fails_before_model_build(self):
        """Composite model cases should reject E2B width explicitly."""
        cfg = _profile_cfg("e2b_dims")
        with self.assertRaisesRegex(ValueError, "does not support"):
            Gemma4TextModelCase().validate_config(cfg)

        with self.assertRaisesRegex(ValueError, "does not support"):
            Gemma4TextScaledWordEmbeddingCase().build(cfg)

    def test_unknown_profile_is_rejected(self):
        """Typos should not silently fall back to tiny dimensions."""
        with self.assertRaisesRegex(ValueError, "Unsupported Gemma4"):
            Gemma4TextMLPCase()._make_text_config(_profile_cfg("full"))

    def test_e2b_circle_filename_is_distinct(self):
        """Original-width artifacts should not overwrite tiny artifacts."""
        case = Gemma4TextMLPCase()
        self.assertEqual(case.export_filename({}), "gemma4_text_mlp.q.circle")
        self.assertEqual(
            case.export_filename(_profile_cfg("e2b_dims")),
            "gemma4_text_mlp.e2b_dims.q.circle",
        )


if __name__ == "__main__":
    unittest.main()
