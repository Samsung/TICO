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


class TestAllocateEmptyCache(unittest.TestCase):
    """Tests for StaticGemma4Runtime._allocate_empty_cache.

    Verifies that per-layer-type head_dim and num_kv_heads are correctly
    selected when allocating KV caches.  Gemma4 uses different head dimensions
    for sliding vs. full-attention layers.
    """

    def test_per_layer_type_head_dim(self):
        """Sliding layers use head_dim; full-attention layers use global_head_dim."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        # Build a minimal mock runtime with just the attributes
        # _allocate_empty_cache accesses.
        text_config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            head_dim=256,
            global_head_dim=512,
            num_key_value_heads=4,
            attention_k_eq_v=False,  # E2B default
        )
        layout = SimpleNamespace(max_seq=64)
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
        )

        # Call the unbound method with our mock
        caches = StaticGemma4Runtime._allocate_empty_cache(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        self.assertEqual(len(caches), 4)

        for layer_idx, cache in enumerate(caches):
            layer_type = text_config.layer_types[layer_idx]
            is_sliding = layer_type == "sliding_attention"

            if not is_sliding and text_config.global_head_dim:
                expected_head_dim = int(text_config.global_head_dim)
            else:
                expected_head_dim = int(text_config.head_dim)

            expected_num_kv_heads = int(text_config.num_key_value_heads)

            expected_shape = (
                1,
                expected_num_kv_heads,
                layout.max_seq,
                expected_head_dim,
            )
            self.assertEqual(
                tuple(cache.past_k.shape),
                expected_shape,
                f"Layer {layer_idx} (type={layer_type}): "
                f"expected past_k shape {expected_shape}, "
                f"got {tuple(cache.past_k.shape)}",
            )
            self.assertEqual(tuple(cache.past_v.shape), expected_shape)

    def test_all_sliding_layers(self):
        """When all layers are sliding, all caches use head_dim."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        text_config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "sliding_attention"],
            head_dim=256,
            global_head_dim=512,
            num_key_value_heads=4,
            attention_k_eq_v=False,
        )
        layout = SimpleNamespace(max_seq=32)
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
        )

        caches = StaticGemma4Runtime._allocate_empty_cache(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        for cache in caches:
            self.assertEqual(cache.past_k.shape[-1], 256)
            self.assertEqual(cache.past_v.shape[-1], 256)

    def test_all_full_attention_layers(self):
        """When all layers are full_attention, all caches use global_head_dim."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        text_config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            head_dim=256,
            global_head_dim=512,
            num_key_value_heads=4,
            attention_k_eq_v=False,
        )
        layout = SimpleNamespace(max_seq=32)
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
        )

        caches = StaticGemma4Runtime._allocate_empty_cache(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        for cache in caches:
            self.assertEqual(cache.past_k.shape[-1], 512)
            self.assertEqual(cache.past_v.shape[-1], 512)

    def test_no_global_head_dim_falls_back(self):
        """When global_head_dim is None, all layers use head_dim."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        text_config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            head_dim=256,
            global_head_dim=None,
            num_key_value_heads=4,
            attention_k_eq_v=False,
        )
        layout = SimpleNamespace(max_seq=32)
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
        )

        caches = StaticGemma4Runtime._allocate_empty_cache(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        for cache in caches:
            self.assertEqual(cache.past_k.shape[-1], 256)
            self.assertEqual(cache.past_v.shape[-1], 256)


class TestBuildDecodeMasksAndRope(unittest.TestCase):
    """Tests for StaticGemma4Runtime.build_decode_masks_and_rope.

    Verifies that decode masks have correct shapes, values, and sliding
    window boundaries at various decode positions.
    """

    def test_full_attention_mask_shape(self):
        """Full attention mask should have shape (batch_size, 1, max_seq)."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        text_config = SimpleNamespace(
            layer_types=["full_attention"],
            sliding_window=1024,
        )
        layout = SimpleNamespace(max_seq=64)

        def mock_rotary_emb(x, pos_ids, layer_type):
            return torch.ones(1, 1, 256), torch.zeros(1, 1, 256)

        model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(rotary_emb=mock_rotary_emb)
            ),
        )
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
            past_len=10,
            model=model,
        )

        masks, rope = StaticGemma4Runtime.build_decode_masks_and_rope(
            runtime, batch_size=2, dtype=torch.float32  # type: ignore[arg-type]
        )

        self.assertIn("full_attention", masks)
        self.assertEqual(masks["full_attention"].shape, (2, 1, 64))

    def test_full_attention_mask_values(self):
        """Full attention mask should allow positions 0..past_len."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        past_len = 10
        max_seq = 64
        text_config = SimpleNamespace(
            layer_types=["full_attention"],
            sliding_window=1024,
        )
        layout = SimpleNamespace(max_seq=max_seq)

        def mock_rotary_emb(x, pos_ids, layer_type):
            return torch.ones(1, 1, 256), torch.zeros(1, 1, 256)

        model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(rotary_emb=mock_rotary_emb)
            ),
        )
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
            past_len=past_len,
            model=model,
        )

        masks, _ = StaticGemma4Runtime.build_decode_masks_and_rope(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        mask = masks["full_attention"][0, 0, :]  # (max_seq,)
        mask_value = torch.finfo(torch.float32).min

        # Positions 0..past_len should be 0.0 (allowed)
        self.assertTrue(torch.all(mask[: past_len + 1] == 0.0))
        # Positions past_len+1..max_seq-1 should be mask_value (forbidden)
        self.assertTrue(torch.all(mask[past_len + 1 :] == mask_value))

    def test_sliding_window_mask_boundary_early_decode(self):
        """Sliding window should cover [0, past_len] when past_len < sliding_window."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        past_len = 5
        sliding_window = 1024
        max_seq = 64
        text_config = SimpleNamespace(
            layer_types=["sliding_attention"],
            sliding_window=sliding_window,
        )
        layout = SimpleNamespace(max_seq=max_seq)

        def mock_rotary_emb(x, pos_ids, layer_type):
            return torch.ones(1, 1, 256), torch.zeros(1, 1, 256)

        model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(rotary_emb=mock_rotary_emb)
            ),
        )
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
            past_len=past_len,
            model=model,
        )

        masks, _ = StaticGemma4Runtime.build_decode_masks_and_rope(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        mask = masks["sliding_attention"][0, 0, :]  # (max_seq,)
        mask_value = torch.finfo(torch.float32).min

        # When past_len < sliding_window, window_start = 0
        # Allowed: positions 0..past_len
        self.assertTrue(torch.all(mask[: past_len + 1] == 0.0))
        self.assertTrue(torch.all(mask[past_len + 1 :] == mask_value))

    def test_sliding_window_mask_boundary_late_decode(self):
        """Sliding window should cover [past_len-sw+1, past_len] when past_len >= sw."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        past_len = 100
        sliding_window = 32
        max_seq = 128
        text_config = SimpleNamespace(
            layer_types=["sliding_attention"],
            sliding_window=sliding_window,
        )
        layout = SimpleNamespace(max_seq=max_seq)

        def mock_rotary_emb(x, pos_ids, layer_type):
            return torch.ones(1, 1, 256), torch.zeros(1, 1, 256)

        model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(rotary_emb=mock_rotary_emb)
            ),
        )
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
            past_len=past_len,
            model=model,
        )

        masks, _ = StaticGemma4Runtime.build_decode_masks_and_rope(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        mask = masks["sliding_attention"][0, 0, :]  # (max_seq,)
        mask_value = torch.finfo(torch.float32).min

        # window_start = max(0, past_len - sliding_window + 1) = 100 - 32 + 1 = 69
        window_start = max(0, past_len - sliding_window + 1)

        # Allowed: positions window_start..past_len
        self.assertTrue(torch.all(mask[window_start : past_len + 1] == 0.0))
        # Forbidden: positions 0..window_start-1 and past_len+1..max_seq-1
        self.assertTrue(torch.all(mask[:window_start] == mask_value))
        self.assertTrue(torch.all(mask[past_len + 1 :] == mask_value))

    def test_sliding_window_mask_first_step(self):
        """Sliding window at past_len=0 should allow only position 0."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        past_len = 0
        sliding_window = 32
        max_seq = 64
        text_config = SimpleNamespace(
            layer_types=["sliding_attention"],
            sliding_window=sliding_window,
        )
        layout = SimpleNamespace(max_seq=max_seq)

        def mock_rotary_emb(x, pos_ids, layer_type):
            return torch.ones(1, 1, 256), torch.zeros(1, 1, 256)

        model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(rotary_emb=mock_rotary_emb)
            ),
        )
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
            past_len=past_len,
            model=model,
        )

        masks, _ = StaticGemma4Runtime.build_decode_masks_and_rope(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        mask = masks["sliding_attention"][0, 0, :]  # (max_seq,)
        mask_value = torch.finfo(torch.float32).min

        # At past_len=0, only position 0 should be allowed
        self.assertEqual(mask[0].item(), 0.0)
        self.assertTrue(torch.all(mask[1:] == mask_value))

    def test_rope_computed_at_past_len(self):
        """RoPE should be computed at position past_len."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        past_len = 42
        max_seq = 64
        text_config = SimpleNamespace(
            layer_types=["full_attention"],
            sliding_window=1024,
        )
        layout = SimpleNamespace(max_seq=max_seq)

        # Track the position_ids passed to rotary_emb
        captured_pos_ids = []

        def mock_rotary_emb(x, pos_ids, layer_type):
            captured_pos_ids.append(pos_ids.clone())
            return torch.ones(1, 1, 256), torch.zeros(1, 1, 256)

        model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(rotary_emb=mock_rotary_emb)
            ),
        )
        runtime = SimpleNamespace(
            text_config=text_config,
            layout=layout,
            device=torch.device("cpu"),
            past_len=past_len,
            model=model,
        )

        _, _ = StaticGemma4Runtime.build_decode_masks_and_rope(
            runtime, batch_size=1, dtype=torch.float32  # type: ignore[arg-type]
        )

        self.assertEqual(len(captured_pos_ids), 1)
        pos_ids = captured_pos_ids[0]
        self.assertEqual(pos_ids.shape, (1, 1))
        self.assertEqual(pos_ids[0, 0].item(), past_len)


if __name__ == "__main__":
    unittest.main()
