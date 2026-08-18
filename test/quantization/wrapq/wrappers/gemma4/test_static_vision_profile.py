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

"""Tests for canonical Gemma4 static vision profiles."""

import unittest
from types import SimpleNamespace

import torch

from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import (
    build_gemma4_static_vision_profile,
    canonicalize_gemma4_static_vision_model_args,
    DEFAULT_GEMMA4_STATIC_VISION_PROFILE,
    get_gemma4_static_vision_profile,
)


class TestGemma4StaticVisionProfile(unittest.TestCase):
    """Validate geometry, text fusion, and named-profile expansion."""

    def setUp(self) -> None:
        """Load the canonical E2B profile used by static export and runtime."""
        self.profile = get_gemma4_static_vision_profile(
            DEFAULT_GEMMA4_STATIC_VISION_PROFILE
        )

    def test_default_e2b_profile_matches_66_by_36_patch_grid(self) -> None:
        """The default profile should use a 66-by-36 row-major patch grid."""
        self.assertEqual(self.profile.name, "e2b_66x36_264")
        self.assertEqual(self.profile.visual_start_idx, 1)
        self.assertEqual(self.profile.visual_end_idx, 265)
        self.assertEqual(self.profile.num_visual_tokens, 264)
        self.assertEqual(self.profile.patch_grid_height, 36)
        self.assertEqual(self.profile.patch_grid_width, 66)
        self.assertEqual(self.profile.soft_grid_height, 12)
        self.assertEqual(self.profile.soft_grid_width, 22)
        self.assertEqual(self.profile.num_valid_patches, 2376)
        self.assertEqual(self.profile.num_patches, 2520)
        self.assertEqual(self.profile.num_padding_patches, 144)
        self.assertEqual(self.profile.image_height, 576)
        self.assertEqual(self.profile.image_width, 1056)
        self.assertEqual(self.profile.patch_vector_size, 768)
        self.assertEqual(
            self.profile.position_ids_sha256(),
            "245f8e683f7c4f08a07cd32b8cd24e0f496656b070eeb5095f5e2a7161894432",
        )

    def test_previous_e2b_profile_remains_available_by_name(self) -> None:
        """The previous 57-by-42 profile should remain explicitly selectable."""
        profile = get_gemma4_static_vision_profile("e2b_57x42_266")
        self.assertEqual(profile.num_visual_tokens, 266)
        self.assertEqual(profile.patch_grid_height, 42)
        self.assertEqual(profile.patch_grid_width, 57)
        self.assertEqual(profile.image_height, 672)
        self.assertEqual(profile.image_width, 912)

    def test_position_ids_are_row_major_with_a_padding_suffix(self) -> None:
        """Position IDs should encode x-fastest coordinates and trailing padding."""
        position_ids = self.profile.build_image_position_ids()
        self.assertEqual(tuple(position_ids.shape), (1, 2520, 2))
        torch.testing.assert_close(position_ids[0, 0], torch.tensor([0, 0]))
        torch.testing.assert_close(position_ids[0, 65], torch.tensor([65, 0]))
        torch.testing.assert_close(position_ids[0, 66], torch.tensor([0, 1]))
        torch.testing.assert_close(position_ids[0, 2375], torch.tensor([65, 35]))
        self.assertTrue(torch.all(position_ids[0, 2376:] == -1))

    def test_position_validation_rejects_a_different_layout(self) -> None:
        """Runtime processor coordinates must match the export profile exactly."""
        changed = self.profile.build_image_position_ids()
        changed[0, 0, 0] = 7
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.profile.validate_image_position_ids(changed)

    def test_text_fusion_validation_accepts_the_exact_slot_span(self) -> None:
        """The processed prompt should expose 264 contiguous image-token slots."""
        image_token_id = 42
        input_ids = torch.zeros(1, 266, dtype=torch.long)
        input_ids[:, 1:265] = image_token_id
        self.profile.validate_text_input_ids(
            input_ids,
            image_token_id=image_token_id,
        )

    def test_text_fusion_validation_rejects_a_shifted_span(self) -> None:
        """A one-token shift should fail before fixed-slot multimodal fusion."""
        image_token_id = 42
        input_ids = torch.zeros(1, 267, dtype=torch.long)
        input_ids[:, 2:266] = image_token_id
        with self.assertRaisesRegex(ValueError, "expected image-token span"):
            self.profile.validate_text_input_ids(
                input_ids,
                image_token_id=image_token_id,
            )

    def test_named_profile_expands_model_args(self) -> None:
        """Named profiles should be the single source for explicit vision fields."""
        model_args = canonicalize_gemma4_static_vision_model_args(
            {"vision": {"profile": DEFAULT_GEMMA4_STATIC_VISION_PROFILE}}
        )
        vision = model_args["vision"]
        self.assertEqual(vision["profile"], "e2b_66x36_264")
        self.assertEqual(vision["visual_start_idx"], 1)
        self.assertEqual(vision["num_visual_tokens"], 264)
        self.assertEqual(vision["max_soft_tokens"], 280)
        self.assertEqual(vision["patch_grid_height"], 36)
        self.assertEqual(vision["patch_grid_width"], 66)
        self.assertNotIn("image_height", vision)
        self.assertNotIn("image_width", vision)

    def test_processor_outputs_match_the_complete_profile_contract(self) -> None:
        """Canonical processor tensors should pass one combined contract check."""
        image_token_id = 42
        input_ids = torch.zeros(1, 266, dtype=torch.long)
        input_ids[:, 1:265] = image_token_id
        outputs = {
            "input_ids": input_ids,
            "pixel_values": torch.zeros(1, 2520, 768),
            "image_position_ids": self.profile.build_image_position_ids(),
            "num_soft_tokens_per_image": torch.tensor([264]),
        }

        self.profile.validate_processor_outputs(
            outputs,
            image_token_id=image_token_id,
        )

    def test_processor_settings_match_the_named_profile(self) -> None:
        """Tokenizer and image-processor defaults should be profile-compatible."""
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(add_bos_token=False),
            image_processor=SimpleNamespace(
                patch_size=16,
                pooling_kernel_size=3,
                max_soft_tokens=280,
            ),
            image_seq_length=280,
        )
        self.profile.validate_processor(processor)

        processor.tokenizer.add_bos_token = True
        with self.assertRaisesRegex(ValueError, "add_bos_token"):
            self.profile.validate_processor(processor)

    def test_profile_build_validates_the_model_geometry(self) -> None:
        """The expanded profile should agree with the Gemma4 vision config."""
        config = SimpleNamespace(
            patch_size=16,
            pooling_kernel_size=3,
            default_output_length=280,
        )
        profile = build_gemma4_static_vision_profile(
            {"vision": {"profile": DEFAULT_GEMMA4_STATIC_VISION_PROFILE}},
            vision_config=config,
            max_seq_len=2048,
        )
        self.assertEqual(profile, self.profile)

    def test_profile_build_rejects_a_conflicting_model_config(self) -> None:
        """A checkpoint with different vision geometry must not reuse the profile."""
        config = SimpleNamespace(
            patch_size=14,
            pooling_kernel_size=3,
            default_output_length=280,
        )
        with self.assertRaisesRegex(ValueError, "patch_size"):
            build_gemma4_static_vision_profile(
                {"vision": {"profile": DEFAULT_GEMMA4_STATIC_VISION_PROFILE}},
                vision_config=config,
                max_seq_len=2048,
            )


if __name__ == "__main__":
    unittest.main()
