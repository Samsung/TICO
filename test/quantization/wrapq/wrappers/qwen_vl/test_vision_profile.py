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

from tico.quantization.wrapq.wrappers.qwen_vl.vision_profile import Qwen3VLVisionProfile


class TestQwen3VLVisionProfile(unittest.TestCase):
    """Validate fixed Qwen3-VL vision profile normalization and naming."""

    def test_normalizes_tuple_and_tensor_values(self):
        """Equivalent concrete grid values should create the same profile."""
        expected = Qwen3VLVisionProfile(2, 4, 6)

        self.assertEqual(
            Qwen3VLVisionProfile.from_grid_thw((2, 4, 6)),
            expected,
        )
        self.assertEqual(
            Qwen3VLVisionProfile.from_grid_thw(
                torch.tensor([[2, 4, 6]], dtype=torch.long)
            ),
            expected,
        )

    def test_derives_static_metadata_and_artifact_name(self):
        """The profile should own static split sizes and artifact identity."""
        profile = Qwen3VLVisionProfile(2, 4, 6)

        self.assertEqual(profile.grid_thw, (2, 4, 6))
        self.assertEqual(profile.num_patch_tokens, 48)
        self.assertEqual(profile.attention_split_sizes, (24, 24))
        self.assertEqual(profile.num_visual_tokens(2), 12)
        self.assertEqual(profile.key, "t2_h4_w6")
        self.assertEqual(
            profile.circle_filename("q"),
            "vision_prefill_t2_h4_w6.q.circle",
        )
        self.assertEqual(
            profile.stage_filename("deepstack_fusion_0", "q"),
            "deepstack_fusion_0_t2_h4_w6.q.circle",
        )

    def test_rejects_non_integer_and_non_positive_dimensions(self):
        """Deployment profiles should contain positive integer dimensions."""
        for value in ((1.0, 4, 4), (True, 4, 4)):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    Qwen3VLVisionProfile.from_grid_thw(value)

        with self.assertRaises(ValueError):
            Qwen3VLVisionProfile.from_grid_thw((1, 0, 4))
        with self.assertRaises(TypeError):
            Qwen3VLVisionProfile.from_grid_thw(torch.ones(1, 3))
        with self.assertRaises(ValueError):
            Qwen3VLVisionProfile.from_grid_thw(torch.ones(2, 3, dtype=torch.long))

    def test_rejects_grid_incompatible_with_spatial_merge(self):
        """The model merger should divide both spatial profile dimensions."""
        profile = Qwen3VLVisionProfile(1, 5, 6)

        with self.assertRaisesRegex(ValueError, "divisible"):
            profile.validate_spatial_merge_size(2)


if __name__ == "__main__":
    unittest.main()
