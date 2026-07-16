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

"""Unit tests for CPU helper functions in static_gemma4_runtime.

These tests exercise the pure-Python helper functions
(`_normalize_valid_token_mask`, `_validate_padding_layout`) without
requiring a real Gemma4 model or processor.
"""

import importlib.util
import unittest

import torch

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers is required for static runtime helpers"
)
class TestNormalizeValidTokenMask(unittest.TestCase):
    """Tests for _normalize_valid_token_mask."""

    def test_with_attention_mask(self):
        """When attention_mask is provided, it should be converted to bool."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _normalize_valid_token_mask,
        )

        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([[True, True, True, False, False]])
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.dtype, torch.bool)

    def test_without_attention_mask_uses_pad_token_id(self):
        """When attention_mask is None, derive from pad_token_id comparison."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _normalize_valid_token_mask,
        )

        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        result = _normalize_valid_token_mask(
            input_ids,
            None,
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([[True, True, True, False, False]])
        self.assertTrue(torch.equal(result, expected))

    def test_without_attention_mask_no_pad_token_id(self):
        """When both attention_mask and pad_token_id are None, all valid."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _normalize_valid_token_mask,
        )

        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        result = _normalize_valid_token_mask(
            input_ids,
            None,
            pad_token_id=None,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([[True, True, True, True, True]])
        self.assertTrue(torch.equal(result, expected))

    def test_shape_mismatch_raises(self):
        """attention_mask with wrong shape should raise ValueError."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _normalize_valid_token_mask,
        )

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 0]])
        with self.assertRaisesRegex(ValueError, "attention_mask shape"):
            _normalize_valid_token_mask(
                input_ids,
                attention_mask,
                pad_token_id=0,
                device=torch.device("cpu"),
            )

    def test_batched_input(self):
        """Should handle batched (2D) input correctly."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _normalize_valid_token_mask,
        )

        input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        result = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([[True, True, False, False], [True, True, True, True]])
        self.assertTrue(torch.equal(result, expected))


@unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers is required for static runtime helpers"
)
class TestValidatePaddingLayout(unittest.TestCase):
    """Tests for _validate_padding_layout."""

    def test_right_padding_valid(self):
        """Right-padded layout (valid then pad) should pass."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        valid_token_mask = torch.tensor([[True, True, True, False, False]])
        # Should not raise
        _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")

    def test_right_padding_no_padding(self):
        """Fully valid sequence (no padding) should pass for right padding."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 2, 3]])
        valid_token_mask = torch.tensor([[True, True, True]])
        _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")

    def test_right_padding_invalid(self):
        """Non-contiguous valid tokens should raise for right padding."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 0, 3, 0, 0]])
        valid_token_mask = torch.tensor([[True, False, True, False, False]])
        with self.assertRaisesRegex(ValueError, "Right padding expected"):
            _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")

    def test_unsupported_padding_side_raises(self):
        """Unsupported padding_side should raise ValueError."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        valid_token_mask = torch.tensor([[True, True, True, False, False]])
        with self.assertRaisesRegex(ValueError, "Unsupported padding_side"):
            _validate_padding_layout(input_ids, valid_token_mask, padding_side="left")

    def test_batched_right_padding_valid(self):
        """Batched right-padded layout should pass."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        valid_token_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")

    def test_batched_right_padding_invalid(self):
        """Batched layout with one bad row should raise."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 2, 0, 0], [3, 0, 5, 0]])
        valid_token_mask = torch.tensor(
            [[True, True, False, False], [True, False, True, False]]
        )
        with self.assertRaisesRegex(ValueError, "Right padding expected"):
            _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")


if __name__ == "__main__":
    unittest.main()
