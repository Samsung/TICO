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

from tico.quantization.wrapq.wrappers.gemma4.utils import fixed_slot_fuse


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

    def test_fixed_slot_fuse_warns_on_wrong_visual_length(self) -> None:
        """Fixed-slot fusion should warn on mismatched visual token counts but still work."""
        text = torch.zeros(1, 6, 2)
        visual = torch.ones(1, 2, 2)

        # Should warn but not raise - uses actual visual token count
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fused = fixed_slot_fuse(
                text,
                visual,
                visual_start_idx=2,
                num_visual_tokens=3,  # Mismatch: visual has 2 tokens, config expects 3
            )
            # Verify warning was issued
            self.assertEqual(len(w), 1)
            self.assertIn("Visual token count mismatch", str(w[0].message))

        # Fusion should succeed using actual token count (2)
        self.assertEqual(tuple(fused.shape), (1, 6, 2))
        # Visual tokens inserted at positions [2:4] (actual count = 2)
        self.assertTrue(torch.equal(fused[:, 2:4], torch.ones(1, 2, 2)))

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


if __name__ == "__main__":
    unittest.main()
