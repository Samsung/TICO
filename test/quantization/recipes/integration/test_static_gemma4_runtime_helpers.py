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

import importlib.util
import unittest

import torch

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers is required for static Gemma4 runtime helpers"
)
class TestStaticGemma4RuntimeHelpers(unittest.TestCase):
    """Test helper functions for StaticGemma4Runtime."""

    def test_normalize_valid_token_mask_with_attention_mask(self):
        """_normalize_valid_token_mask should convert attention mask to boolean."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _normalize_valid_token_mask,
        )

        input_ids = torch.tensor([[4, 5, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])

        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(valid.tolist(), [[True, True, False, False]])

    def test_normalize_valid_token_mask_without_attention_mask(self):
        """_normalize_valid_token_mask should use pad_token_id when no attention mask."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _normalize_valid_token_mask,
        )

        input_ids = torch.tensor([[4, 5, 0, 0]])

        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask=None,
            pad_token_id=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(valid.tolist(), [[True, True, False, False]])

    def test_validate_padding_layout_right_padding(self):
        """_validate_padding_layout should accept right-padded sequences."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        valid_token_mask = torch.tensor([[True, True, False, False]])

        # Should not raise
        _validate_padding_layout(
            torch.tensor([[4, 5, 0, 0]]),
            valid_token_mask,
            padding_side="right",
        )

    def test_validate_padding_layout_right_padding_invalid(self):
        """_validate_padding_layout should reject invalid right-padded sequences."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        # Invalid: True after False (not proper right padding)
        valid_token_mask = torch.tensor([[True, False, True, False]])

        with self.assertRaises(ValueError):
            _validate_padding_layout(
                torch.tensor([[4, 0, 5, 0]]),
                valid_token_mask,
                padding_side="right",
            )

    def test_build_position_ids_from_valid_token_mask(self):
        """_build_position_ids_from_valid_token_mask should generate sequential IDs."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _build_position_ids_from_valid_token_mask,
        )

        valid_token_mask = torch.tensor([[True, True, False, False]])

        position_ids = _build_position_ids_from_valid_token_mask(valid_token_mask)

        self.assertEqual(position_ids.tolist(), [[0, 1, 2, 3]])

    def test_gather_last_token_logits_right_padding(self):
        """_gather_last_token_logits should select last valid token for each row."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _gather_last_token_logits,
        )

        # Shape: (batch=2, seq=4, vocab=3)
        logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        valid = torch.tensor(
            [
                [True, True, False, False],  # Last valid at index 1
                [True, True, True, False],  # Last valid at index 2
            ]
        )

        gathered = _gather_last_token_logits(logits, valid, padding_side="right")

        self.assertTrue(torch.equal(gathered[0], logits[0, 1]))
        self.assertTrue(torch.equal(gathered[1], logits[1, 2]))

    def test_build_gemma4_rope_templates(self):
        """_build_gemma4_rope_templates should build per-layer-type RoPE tables."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _build_gemma4_rope_templates,
        )

        # Mock config
        class MockConfig:
            hidden_size = 64
            num_attention_heads = 4
            layer_types = ["full_attention", "sliding_attention"]

        config = MockConfig()
        max_seq = 16
        device = torch.device("cpu")

        rope_tables = _build_gemma4_rope_templates(config, max_seq, device)

        # Should have entries for both layer types
        self.assertIn("full_attention", rope_tables)
        self.assertIn("sliding_attention", rope_tables)

        # Check shapes: (1, max_seq, head_dim)
        for layer_type, (cos, sin) in rope_tables.items():
            self.assertEqual(cos.shape, (1, max_seq, 16))  # head_dim = 64/4 = 16
            self.assertEqual(sin.shape, (1, max_seq, 16))

    def test_build_gemma4_prefill_masks_full_attention(self):
        """_build_gemma4_prefill_masks should build causal mask for full_attention."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _build_gemma4_prefill_masks,
        )

        valid_token_mask = torch.tensor([[True, True, True, False]])
        layer_types = ["full_attention"]
        device = torch.device("cpu")
        dtype = torch.float32

        masks = _build_gemma4_prefill_masks(
            valid_token_mask,
            layer_types,
            sliding_window=None,
            device=device,
            dtype=dtype,
        )

        self.assertIn("full_attention", masks)
        mask = masks["full_attention"]

        # Shape: (batch, 1, seq, seq)
        self.assertEqual(mask.shape, (1, 1, 4, 4))

        # Should be lower triangular (causal)
        # Query 0 can attend to key 0
        # Query 1 can attend to keys 0, 1
        # Query 2 can attend to keys 0, 1, 2
        # Invalid tokens (index 3) should be masked
        self.assertEqual(mask[0, 0, 0, 0].item(), 0.0)  # Not masked
        self.assertLess(mask[0, 0, 0, 1].item(), -100)  # Masked (future)
        # Query 1 (valid) should attend to key 0 (valid, causal)
        self.assertEqual(mask[0, 0, 1, 0].item(), 0.0)  # Not masked
        # Query 3 (invalid) should be fully masked
        self.assertLess(mask[0, 0, 3, 0].item(), -100)  # Masked (invalid query)

    def test_build_gemma4_prefill_masks_sliding_window(self):
        """_build_gemma4_prefill_masks should build sliding window causal mask."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _build_gemma4_prefill_masks,
        )

        valid_token_mask = torch.tensor([[True, True, True, True]])
        layer_types = ["sliding_attention"]
        sliding_window = 2
        device = torch.device("cpu")
        dtype = torch.float32

        masks = _build_gemma4_prefill_masks(
            valid_token_mask,
            layer_types,
            sliding_window=sliding_window,
            device=device,
            dtype=dtype,
        )

        self.assertIn("sliding_attention", masks)
        mask = masks["sliding_attention"]

        # Query 0 can attend to keys in [0, 0] (window=2, but only key 0 exists before)
        # Query 1 can attend to keys in [0, 1]
        # Query 2 can attend to keys in [1, 2] (sliding window)
        # Query 3 can attend to keys in [2, 3] (sliding window)

        # Query 2 should NOT attend to key 0 (outside window)
        self.assertLess(mask[0, 0, 2, 0].item(), -100)
        # Query 2 should attend to key 1 (inside window)
        self.assertEqual(mask[0, 0, 2, 1].item(), 0.0)
        # Query 2 should attend to key 2 (causal)
        self.assertEqual(mask[0, 0, 2, 2].item(), 0.0)

    def test_build_gemma4_decode_masks_full_attention(self):
        """_build_gemma4_decode_masks should allow all past tokens for full_attention."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _build_gemma4_decode_masks,
        )

        batch_size = 1
        past_len = 3
        max_seq = 8
        layer_types = ["full_attention"]
        sliding_window = None
        device = torch.device("cpu")
        dtype = torch.float32

        masks = _build_gemma4_decode_masks(
            batch_size, past_len, max_seq, layer_types, sliding_window, device, dtype
        )

        self.assertIn("full_attention", masks)
        mask = masks["full_attention"]

        # Shape: (batch, 1, max_seq)
        self.assertEqual(mask.shape, (1, 1, 8))

        # Positions [0:4] should be visible (past_len + 1 = 4)
        # Positions [4:8] should be masked (future)
        for i in range(past_len + 1):
            self.assertEqual(mask[0, 0, i].item(), 0.0)

    def test_build_gemma4_decode_masks_sliding_window(self):
        """_build_gemma4_decode_masks should limit visible tokens for sliding_attention."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _build_gemma4_decode_masks,
        )

        batch_size = 1
        past_len = 5
        max_seq = 8
        layer_types = ["sliding_attention"]
        sliding_window = 3
        device = torch.device("cpu")
        dtype = torch.float32

        masks = _build_gemma4_decode_masks(
            batch_size, past_len, max_seq, layer_types, sliding_window, device, dtype
        )

        self.assertIn("sliding_attention", masks)
        mask = masks["sliding_attention"]

        # Sliding window: visible range is [past_len - sliding_window + 1 : past_len + 1]
        # = [5 - 3 + 1 : 5 + 1] = [3 : 6]
        # Positions 0, 1, 2 should be masked (outside window)
        # Positions 3, 4, 5 should be visible
        # Positions 6, 7 should be masked (future)

        for i in range(3):
            self.assertLess(mask[0, 0, i].item(), -100)  # Masked

        for i in range(3, 6):
            self.assertEqual(mask[0, 0, i].item(), 0.0)  # Visible

        for i in range(6, 8):
            self.assertLess(mask[0, 0, i].item(), -100)  # Masked (future)

    def test_apply_logit_softcapping(self):
        """_apply_logit_softcapping should bound logits to [-softcap, softcap]."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _apply_logit_softcapping,
        )

        logits = torch.tensor([[-100.0, -10.0, 0.0, 10.0, 100.0]])
        softcap = 5.0

        softcapped = _apply_logit_softcapping(logits, softcap)

        # All values should be bounded to [-5, 5]
        self.assertTrue(torch.all(softcapped >= -softcap))
        self.assertTrue(torch.all(softcapped <= softcap))

        # Extreme values should be close to softcap
        self.assertAlmostEqual(softcapped[0, 0].item(), -softcap, places=5)
        self.assertAlmostEqual(softcapped[0, 4].item(), softcap, places=5)

    def test_apply_logit_softcapping_none(self):
        """_apply_logit_softcapping should return logits unchanged if softcap is None."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _apply_logit_softcapping,
        )

        logits = torch.tensor([[1.0, 2.0, 3.0]])

        result = _apply_logit_softcapping(logits, None)

        self.assertTrue(torch.equal(result, logits))

    def test_gather_rope_by_position_ids(self):
        """_gather_rope_by_position_ids should gather RoPE at specified positions."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _gather_rope_by_position_ids,
        )

        # Create mock RoPE tables
        max_seq = 8
        head_dim = 16
        cos_full = torch.randn(1, max_seq, head_dim)
        sin_full = torch.randn(1, max_seq, head_dim)
        rope_tables = {"full_attention": (cos_full, sin_full)}

        # Position IDs: batch=2, seq=3
        position_ids = torch.tensor(
            [
                [0, 2, 4],
                [1, 3, 5],
            ]
        )
        layer_types = ["full_attention"]

        result = _gather_rope_by_position_ids(rope_tables, position_ids, layer_types)

        self.assertIn("full_attention", result)
        cos, sin = result["full_attention"]

        # Output shape should be (batch, seq, head_dim)
        self.assertEqual(cos.shape, (2, 3, 16))
        self.assertEqual(sin.shape, (2, 3, 16))

        # Verify gathered values match source
        for b in range(2):
            for s in range(3):
                pos = position_ids[b, s].item()
                self.assertTrue(torch.allclose(cos[b, s], cos_full[0, pos]))
                self.assertTrue(torch.allclose(sin[b, s], sin_full[0, pos]))


if __name__ == "__main__":
    unittest.main()
