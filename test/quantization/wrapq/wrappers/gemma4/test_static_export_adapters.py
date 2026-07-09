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

from tico.quantization.wrapq.wrappers.gemma4.quant_model import QuantGemma4Model
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    dynamic_placeholder_fuse,
    fixed_slot_fuse,
    validate_static_visual_layout,
)


class Gemma4StaticExportAdapterUtilityTest(unittest.TestCase):
    """Test utility functions used by static Gemma4 export adapters."""

    def test_fixed_slot_fuse_replaces_expected_range(self) -> None:
        """Fixed-slot fusion should replace exactly the configured visual range."""
        text = torch.zeros(1, 6, 2)
        visual = torch.ones(1, 2, 2)

        fused = fixed_slot_fuse(
            text,
            visual,
            visual_start_idx=2,
            num_visual_tokens=2,
        )

        self.assertEqual(tuple(fused.shape), tuple(text.shape))
        self.assertTrue(torch.equal(fused[:, :2], torch.zeros(1, 2, 2)))
        self.assertTrue(torch.equal(fused[:, 2:4], torch.ones(1, 2, 2)))
        self.assertTrue(torch.equal(fused[:, 4:], torch.zeros(1, 2, 2)))

    def test_fixed_slot_fuse_rejects_wrong_visual_length(self) -> None:
        """Fixed-slot fusion should reject mismatched visual token counts."""
        text = torch.zeros(1, 6, 2)
        visual = torch.ones(1, 2, 2)

        with self.assertRaisesRegex(ValueError, "expected 3, got 2"):
            fixed_slot_fuse(
                text,
                visual,
                visual_start_idx=2,
                num_visual_tokens=3,
            )

    def test_fixed_slot_fuse_rejects_out_of_range_slot(self) -> None:
        """Fixed-slot fusion should reject visual slots that exceed max sequence length."""
        text = torch.zeros(1, 6, 2)
        visual = torch.ones(1, 2, 2)

        with self.assertRaisesRegex(ValueError, "Invalid visual slot range"):
            fixed_slot_fuse(
                text,
                visual,
                visual_start_idx=5,
                num_visual_tokens=2,
            )

    def test_dynamic_placeholder_fuse_replaces_mask_positions(self) -> None:
        """Dynamic fusion should replace the actual image placeholder positions."""
        text = torch.zeros(1, 6, 2)
        visual = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        image_mask = torch.tensor([[False, True, False, True, False, False]])

        fused = dynamic_placeholder_fuse(text, visual, image_mask)

        expected = text.clone()
        expected[:, 1, :] = visual[:, 0, :]
        expected[:, 3, :] = visual[:, 1, :]
        self.assertTrue(torch.equal(fused, expected))

    def test_dynamic_placeholder_fuse_rejects_token_mismatch(self) -> None:
        """Dynamic fusion should reject placeholder and feature count mismatches."""
        text = torch.zeros(1, 6, 2)
        visual = torch.ones(1, 2, 2)
        image_mask = torch.tensor([[False, True, False, False, False, False]])

        with self.assertRaisesRegex(ValueError, "placeholder count"):
            dynamic_placeholder_fuse(text, visual, image_mask)

    def test_dynamic_and_fixed_fusion_match_for_valid_static_layout(self) -> None:
        """Dynamic and fixed fusion should agree for a valid static visual span."""
        text = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
        visual = torch.full((1, 2, 2), 99.0)
        image_mask = torch.tensor([[False, False, True, True, False, False]])

        validate_static_visual_layout(
            image_mask,
            visual_start_idx=2,
            num_visual_tokens=2,
            seq_len=6,
        )
        dynamic = dynamic_placeholder_fuse(text, visual, image_mask)
        static = fixed_slot_fuse(
            text,
            visual,
            visual_start_idx=2,
            num_visual_tokens=2,
        )

        torch.testing.assert_close(dynamic, static)

    def test_validate_static_visual_layout_rejects_wrong_span(self) -> None:
        """Static validation should reject non-contiguous or wrong-start masks."""
        image_mask = torch.tensor([[False, True, False, True, False, False]])

        with self.assertRaisesRegex(ValueError, "static visual-token span"):
            validate_static_visual_layout(
                image_mask,
                visual_start_idx=2,
                num_visual_tokens=2,
                seq_len=6,
            )

    def test_force_export_flag_enables_static_branch_for_unit_tests(self) -> None:
        """force_export should allow tests to exercise the static fusion branch."""
        model = object.__new__(QuantGemma4Model)

        model.force_export = False
        self.assertFalse(model._uses_static_fusion())

        model.force_export = True
        self.assertTrue(model._uses_static_fusion())

    def test_static_layout_validation_is_opt_in_for_eager_path(self) -> None:
        """Dynamic eager validation should not enforce static spans by default."""
        model = object.__new__(QuantGemma4Model)
        model.validate_static_layout = False
        model.visual_start_idx = 2
        model.num_visual_tokens = 2
        image_mask = torch.tensor([[False, True, False, True, False, False]])

        model._validate_static_image_layout(image_mask, seq_len=6)

    def test_static_layout_validation_rejects_wrong_span_when_enabled(self) -> None:
        """Opt-in static validation should reject non-static eager inputs."""
        model = object.__new__(QuantGemma4Model)
        model.validate_static_layout = True
        model.visual_start_idx = 2
        model.num_visual_tokens = 2
        image_mask = torch.tensor([[False, True, False, True, False, False]])

        with self.assertRaisesRegex(ValueError, "static visual-token span"):
            model._validate_static_image_layout(image_mask, seq_len=6)

    def test_static_visual_token_count_validation_has_clear_error(self) -> None:
        """Static token-count validation should report image-size config hints."""
        model = object.__new__(QuantGemma4Model)
        model.num_visual_tokens = 3
        image_embeds = torch.zeros(1, 2, 4)

        with self.assertRaisesRegex(ValueError, "image_height"):
            model._validate_static_visual_token_count(image_embeds)


if __name__ == "__main__":
    unittest.main()
