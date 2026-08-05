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

"""Tests for Qwen3-VL wrapper-smoke size profiles."""

import unittest

from tico.quantization.recipes.debug.wrapper_smoke.cases.qwen3_vl import (
    _has_qwen3_vl,
    Qwen3VLStaticRuntimeShape,
    QwenForConditionalGenerationCase,
    QwenModelCase,
    QwenTextAttentionDecodeCase,
    QwenTextAttentionPrefillCase,
    QwenTextDecoderLayerDecodeCase,
    QwenTextDecoderLayerPrefillCase,
    QwenTextMLPCase,
    QwenTextModelCase,
    QwenVisionAttentionCase,
    QwenVisionBlockCase,
    QwenVisionMLPCase,
    QwenVisionModelCase,
    QwenVisionPatchEmbedCase,
    QwenVisionPatchMergerCase,
)


def _profile_cfg(
    profile: str,
    *,
    max_seq: int = 2_048,
    grid_thw: tuple[int, int, int] = (1, 54, 72),
    visual_capacity: int = 1_000,
    non_visual_tokens: int = 14,
    visual_start_idx: int = 4,
) -> dict:
    """Build a minimal wrapper-smoke config for one Qwen3-VL profile."""
    return {
        "debug": {
            "wrapper_smoke": {
                "qwen3_vl": {
                    "size_profile": profile,
                    "static_runtime": {
                        "max_seq": max_seq,
                        "grid_thw": list(grid_thw),
                        "visual_capacity": visual_capacity,
                        "non_visual_tokens": non_visual_tokens,
                        "visual_start_idx": visual_start_idx,
                    },
                }
            }
        }
    }


_QWEN_SUPPORTED_CASE_TYPES = (
    QwenTextAttentionPrefillCase,
    QwenTextAttentionDecodeCase,
    QwenTextMLPCase,
    QwenTextDecoderLayerPrefillCase,
    QwenTextDecoderLayerDecodeCase,
    QwenVisionAttentionCase,
    QwenVisionMLPCase,
    QwenVisionBlockCase,
    QwenVisionPatchEmbedCase,
    QwenVisionPatchMergerCase,
    QwenVisionModelCase,
)
_QWEN_AVAILABILITY = _has_qwen3_vl()


class TestQwen3VLWrapperSmokeProfileValidation(unittest.TestCase):
    """Validate profile selection and fixed runtime-shape calculations."""

    def test_default_static_runtime_contract(self):
        """The default shape should match the Qwen3-VL-4B TICO runtime contract."""
        shape = Qwen3VLStaticRuntimeShape()
        self.assertEqual(shape.max_seq, 2_048)
        self.assertEqual(shape.grid_thw, (1, 54, 72))
        self.assertEqual(shape.num_patch_tokens, 3_888)
        self.assertEqual(shape.num_visual_tokens, 972)
        self.assertEqual(shape.valid_seq_len, 986)
        self.assertEqual(shape.visual_capacity, 1_000)
        self.assertEqual(shape.visual_arena_start, 1_048)
        self.assertEqual(shape.visual_start_idx, 4)

    def test_static_runtime_rejects_incompatible_shapes(self):
        """Invalid merge grids and visual arenas should fail before model build."""
        with self.assertRaisesRegex(ValueError, "height must be divisible"):
            Qwen3VLStaticRuntimeShape(grid_thw=(1, 53, 72))
        with self.assertRaisesRegex(ValueError, "visual_capacity"):
            Qwen3VLStaticRuntimeShape(visual_capacity=900)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            Qwen3VLStaticRuntimeShape(max_seq=900)
        with self.assertRaisesRegex(ValueError, "non_visual_tokens"):
            Qwen3VLStaticRuntimeShape(non_visual_tokens=-1)
        with self.assertRaisesRegex(ValueError, "visual_start_idx"):
            Qwen3VLStaticRuntimeShape(visual_start_idx=15)

    def test_supported_cases_accept_both_qwen3_vl_4b_profiles(self):
        """Bounded text/vision modules should accept dimensions and static profiles."""
        for case_type in _QWEN_SUPPORTED_CASE_TYPES:
            for profile in (
                "qwen3_vl_4b_dims",
                "qwen3_vl_4b_static_runtime",
            ):
                with self.subTest(case=case_type.__name__, profile=profile):
                    case_type().validate_config(_profile_cfg(profile))

    def test_embedding_and_full_model_cases_reject_real_width_profiles(self):
        """Cases with full embeddings or LM heads should fail before allocation."""
        for case_type in (
            QwenTextModelCase,
            QwenModelCase,
            QwenForConditionalGenerationCase,
        ):
            with self.subTest(case=case_type.__name__):
                with self.assertRaisesRegex(ValueError, "does not support"):
                    case_type().validate_config(_profile_cfg("qwen3_vl_4b_dims"))

    def test_unknown_profile_is_rejected(self):
        """Profile typos should not silently fall back to tiny dimensions."""
        with self.assertRaisesRegex(ValueError, "Unsupported Qwen3-VL"):
            QwenTextMLPCase().validate_config(_profile_cfg("qwen3_vl_4b"))

    def test_static_shape_override_is_cached_by_the_case(self):
        """A smaller valid test grid should remain configurable."""
        cfg = _profile_cfg(
            "qwen3_vl_4b_static_runtime",
            max_seq=128,
            grid_thw=(1, 8, 8),
            visual_capacity=16,
            non_visual_tokens=8,
            visual_start_idx=4,
        )
        case = QwenVisionAttentionCase()
        case.validate_config(cfg)
        shape = case._static_runtime_shape()
        assert shape is not None
        self.assertEqual(shape.num_patch_tokens, 64)
        self.assertEqual(shape.num_visual_tokens, 16)
        self.assertEqual(shape.valid_seq_len, 24)
        self.assertEqual(case._vision_grid_tuple((1, 2, 2)), (1, 8, 8))

    def test_profile_specific_circle_filenames_are_distinct(self):
        """Large-profile artifacts should not overwrite tiny artifacts."""
        case = QwenTextMLPCase()
        self.assertEqual(case.export_filename({}), "qwen3_vl_text_mlp.q.circle")
        self.assertEqual(
            case.export_filename(_profile_cfg("qwen3_vl_4b_dims")),
            "qwen3_vl_text_mlp.qwen3_vl_4b_dims.q.circle",
        )
        self.assertEqual(
            case.export_filename(_profile_cfg("qwen3_vl_4b_static_runtime")),
            "qwen3_vl_text_mlp.qwen3_vl_4b_static_runtime.q.circle",
        )


@unittest.skipUnless(
    _QWEN_AVAILABILITY.available,
    _QWEN_AVAILABILITY.reason or "Qwen3-VL is unavailable",
)
class TestQwen3VLWrapperSmokeConfigDimensions(unittest.TestCase):
    """Validate tiny compatibility and Qwen3-VL-4B target dimensions."""

    def test_tiny_remains_the_default_profile(self):
        """An omitted profile should preserve the current synthetic configs."""
        text_cfg = QwenTextMLPCase()._make_text_config({})
        vision_cfg = QwenVisionBlockCase()._make_vision_config({})
        self.assertEqual(text_cfg.hidden_size, 64)
        self.assertEqual(text_cfg.intermediate_size, 128)
        self.assertEqual(text_cfg.num_attention_heads, 2)
        self.assertEqual(text_cfg.num_key_value_heads, 2)
        self.assertEqual(text_cfg.head_dim, 32)
        self.assertEqual(vision_cfg.hidden_size, 64)
        self.assertEqual(vision_cfg.num_heads, 4)
        self.assertEqual(vision_cfg.depth, 2)
        self.assertEqual(vision_cfg.out_hidden_size, 64)

    def test_dims_profile_uses_qwen3_vl_4b_text_widths(self):
        """The text config should keep one layer while copying 4B dimensions."""
        text_cfg = QwenTextMLPCase()._make_text_config(_profile_cfg("qwen3_vl_4b_dims"))
        self.assertEqual(text_cfg.vocab_size, 151_936)
        self.assertEqual(text_cfg.hidden_size, 2_560)
        self.assertEqual(text_cfg.intermediate_size, 9_728)
        self.assertEqual(text_cfg.num_hidden_layers, 1)
        self.assertEqual(text_cfg.num_attention_heads, 32)
        self.assertEqual(text_cfg.num_key_value_heads, 8)
        self.assertEqual(text_cfg.head_dim, 128)
        self.assertEqual(text_cfg.max_position_embeddings, 128)

    def test_dims_profile_uses_bounded_qwen3_vl_4b_vision_widths(self):
        """The vision config should copy 4B widths but retain one smoke layer."""
        vision_cfg = QwenVisionBlockCase()._make_vision_config(
            _profile_cfg("qwen3_vl_4b_dims")
        )
        self.assertEqual(vision_cfg.hidden_size, 1_024)
        self.assertEqual(vision_cfg.intermediate_size, 4_096)
        self.assertEqual(vision_cfg.num_heads, 16)
        self.assertEqual(vision_cfg.depth, 1)
        self.assertEqual(vision_cfg.out_hidden_size, 2_560)
        self.assertEqual(vision_cfg.patch_size, 16)
        self.assertEqual(vision_cfg.temporal_patch_size, 2)
        self.assertEqual(vision_cfg.spatial_merge_size, 2)
        self.assertEqual(vision_cfg.num_position_embeddings, 2_304)

    def test_static_profile_uses_runtime_text_capacity(self):
        """Static text wrappers should use a 2,048-token mask/cache capacity."""
        text_cfg = QwenTextAttentionDecodeCase()._make_text_config(
            _profile_cfg("qwen3_vl_4b_static_runtime")
        )
        self.assertEqual(text_cfg.hidden_size, 2_560)
        self.assertEqual(text_cfg.max_position_embeddings, 2_048)


if __name__ == "__main__":
    unittest.main()
