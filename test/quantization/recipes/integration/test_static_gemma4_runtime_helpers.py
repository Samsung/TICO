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

import contextlib
import importlib.util
import io
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
            input_ids,  # type: ignore[arg-type]
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
            input_ids,  # type: ignore[arg-type]
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
            input_ids,  # type: ignore[arg-type]
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
                input_ids,  # type: ignore[arg-type]
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
            input_ids,  # type: ignore[arg-type]
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
        _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")  # type: ignore[arg-type]

    def test_right_padding_no_padding(self):
        """Fully valid sequence (no padding) should pass for right padding."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 2, 3]])
        valid_token_mask = torch.tensor([[True, True, True]])
        _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")  # type: ignore[arg-type]

    def test_right_padding_invalid(self):
        """Non-contiguous valid tokens should raise for right padding."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 0, 3, 0, 0]])
        valid_token_mask = torch.tensor([[True, False, True, False, False]])
        with self.assertRaisesRegex(ValueError, "Right padding expected"):
            _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")  # type: ignore[arg-type]

    def test_unsupported_padding_side_raises(self):
        """Unsupported padding_side should raise ValueError."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        valid_token_mask = torch.tensor([[True, True, True, False, False]])
        with self.assertRaisesRegex(ValueError, "Unsupported padding_side"):
            _validate_padding_layout(input_ids, valid_token_mask, padding_side="left")  # type: ignore[arg-type]

    def test_batched_right_padding_valid(self):
        """Batched right-padded layout should pass."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        valid_token_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")  # type: ignore[arg-type]

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
            _validate_padding_layout(input_ids, valid_token_mask, padding_side="right")  # type: ignore[arg-type]


@unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers is required for static runtime helpers"
)
class TestBuildStaticInputs(unittest.TestCase):
    """Tests for the fixed sequence-length processor contract."""

    def test_uses_layout_max_seq_and_rejects_override(self):
        """The processor batch should always use ``layout.max_seq``."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        class FakeProcessor:
            """Return one small image-text processor batch."""

            def __call__(self, **kwargs):
                del kwargs
                return {
                    "input_ids": torch.tensor([[7, 99, 99, 8]], dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                    "pixel_values": torch.zeros(1, 8, 12),
                    "image_position_ids": torch.zeros(1, 8, 2, dtype=torch.long),
                }

        class FakeVisionProfile:
            """Accept the synthetic processor output."""

            def validate_processor_outputs(self, outputs, *, image_token_id):
                self.outputs = outputs
                self.image_token_id = image_token_id

        runtime = object.__new__(StaticGemma4Runtime)
        runtime.processor = FakeProcessor()  # type: ignore[assignment]
        runtime.text_config = SimpleNamespace(pad_token_id=0)
        runtime.device = torch.device("cpu")
        runtime.layout = SimpleNamespace(max_seq=8)  # type: ignore[assignment]
        runtime.config = SimpleNamespace(image_token_id=99)  # type: ignore[assignment]
        runtime.vision_profile = FakeVisionProfile()  # type: ignore[assignment]

        batch = runtime.build_static_inputs("prompt", image=None)

        self.assertEqual(tuple(batch["llm_input_ids"].shape), (1, 8))
        self.assertEqual(tuple(batch["attention_mask"].shape), (1, 8))
        torch.testing.assert_close(
            batch["llm_input_ids"][0, :4],
            torch.tensor([7, 0, 0, 8], dtype=torch.long),
        )
        self.assertTrue(torch.all(batch["llm_input_ids"][0, 4:] == 0))

        with self.assertRaises(TypeError):
            runtime.build_static_inputs(  # type: ignore[call-arg] # pylint: disable=unexpected-keyword-arg
                "prompt",
                image=None,
                max_seq=4,
            )


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
                layout.max_seq - 1,
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


@unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers is required for static runtime helpers"
)
class TestAttentionMaskFillValue(unittest.TestCase):
    """Tests for the runtime/export additive-mask value contract."""

    def test_prefill_and_decode_use_configured_fill_value(self):
        """Both text paths should use the configured PTQ mask fill value."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        def mock_rotary_emb(x, position_ids, layer_type):
            del position_ids, layer_type
            return torch.ones_like(x), torch.zeros_like(x)

        class FakeModel(torch.nn.Module):
            """Expose a parameter dtype and the Gemma4 rotary hierarchy."""

            def __init__(self):
                super().__init__()
                self.probe = torch.nn.Parameter(torch.zeros(1))
                self.model = SimpleNamespace(
                    language_model=SimpleNamespace(rotary_emb=mock_rotary_emb)
                )

        max_seq = 8
        fill_value = -37.5
        runtime = SimpleNamespace(
            text_config=SimpleNamespace(
                layer_types=["full_attention", "sliding_attention"],
                sliding_window=2,
            ),
            layout=SimpleNamespace(max_seq=max_seq),
            device=torch.device("cpu"),
            past_len=3,
            model=FakeModel(),
            attention_mask_fill_value=fill_value,
        )

        input_ids = torch.ones(1, max_seq, dtype=torch.long)
        valid = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)
        prefill_masks, _ = StaticGemma4Runtime.build_prefill_masks_and_rope(
            runtime,  # type: ignore[arg-type]
            input_ids,
            valid,
        )
        self.assertEqual(
            prefill_masks["full_attention"][0, 0, 0, 3].item(),
            fill_value,
        )
        self.assertEqual(
            prefill_masks["sliding_attention"][0, 0, 2, 0].item(),
            fill_value,
        )

        decode_masks, _ = StaticGemma4Runtime.build_decode_masks_and_rope(
            runtime,  # type: ignore[arg-type]
            batch_size=1,
            dtype=torch.float32,
        )
        self.assertEqual(
            decode_masks["full_attention"][0, 0, 3].item(),
            fill_value,
        )
        self.assertEqual(
            decode_masks["sliding_attention"][0, 0, 0].item(),
            fill_value,
        )
        self.assertEqual(
            decode_masks["full_attention"][0, 0, max_seq - 1].item(),
            0.0,
        )


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
        """Full attention should allow valid past slots and the fixed current slot."""
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
        mask_value = -120.0

        # Valid past-cache slots are allowed.
        self.assertTrue(torch.all(mask[:past_len] == 0.0))
        # Unused persistent-cache slots remain masked.
        self.assertTrue(torch.all(mask[past_len : max_seq - 1] == mask_value))
        # The current token is always appended at the final physical slot.
        self.assertEqual(mask[max_seq - 1].item(), 0.0)

    def test_sliding_window_mask_boundary_early_decode(self):
        """Early sliding decode should allow all valid past slots plus current."""
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
        mask_value = -120.0

        # When past_len < sliding_window, every valid past slot is allowed.
        self.assertTrue(torch.all(mask[:past_len] == 0.0))
        self.assertTrue(torch.all(mask[past_len : max_seq - 1] == mask_value))
        self.assertEqual(mask[max_seq - 1].item(), 0.0)

    def test_sliding_window_mask_boundary_late_decode(self):
        """Late sliding decode should keep W-1 past slots plus current."""
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
        mask_value = -120.0

        # window_start = max(0, past_len - sliding_window + 1) = 100 - 32 + 1 = 69
        window_start = max(0, past_len - sliding_window + 1)

        # Allow the most recent W-1 past slots and the fixed current slot.
        self.assertTrue(torch.all(mask[window_start:past_len] == 0.0))
        self.assertTrue(torch.all(mask[:window_start] == mask_value))
        self.assertTrue(torch.all(mask[past_len : max_seq - 1] == mask_value))
        self.assertEqual(mask[max_seq - 1].item(), 0.0)

    def test_sliding_window_mask_first_step(self):
        """At past_len=0 only the fixed current-token slot should be allowed."""
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
        mask_value = -120.0

        # At past_len=0, only the final current-token slot is allowed.
        self.assertTrue(torch.all(mask[: max_seq - 1] == mask_value))
        self.assertEqual(mask[max_seq - 1].item(), 0.0)

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


@unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers is required for static runtime helpers"
)
class TestVerifyVisionPrefill(unittest.TestCase):
    """Tests for the pixel-values-only vision-prefill verification ABI."""

    def test_uses_pixel_values_only_for_eager_and_export_paths(self):
        """The verification helper should never pass runtime position IDs."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            verify_step_vision_prefill,
        )

        class FakeVisionPrefill(torch.nn.Module):
            """Flatten the fixed batch dimension from synthetic visual tokens."""

            def forward(self, pixel_values):
                return pixel_values.squeeze(0)

        class FakeReferenceModel:
            """Return the same output through the HF-style reference API."""

            dtype = torch.float32

            def get_image_features(
                self,
                *,
                pixel_values,
                image_position_ids,
                return_dict,
            ):
                del image_position_ids, return_dict
                return SimpleNamespace(pooler_output=pixel_values.squeeze(0))

        class FakeWrappedModel:
            """Return the same output through the wrapped eager reference API."""

            def get_image_features(self, *, pixel_values, image_position_ids):
                del image_position_ids
                return pixel_values.squeeze(0)

        vision_prefill = FakeVisionPrefill()
        runtime = SimpleNamespace(
            device=torch.device("cpu"),
            model=FakeReferenceModel(),
            qmodel=SimpleNamespace(
                wrapped=SimpleNamespace(
                    model=SimpleNamespace(wrapped=FakeWrappedModel())
                )
            ),
            _get_or_create_vision_prefill=lambda image_position_ids: vision_prefill,
        )
        batch = {
            "pixel_values": torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
            "image_position_ids": torch.zeros(1, 4, 2, dtype=torch.long),
        }

        with contextlib.redirect_stdout(io.StringIO()):
            visual_embeds = verify_step_vision_prefill(runtime, batch)  # type: ignore[arg-type]

        self.assertEqual(tuple(visual_embeds.shape), (4, 2))
        torch.testing.assert_close(
            visual_embeds,
            batch["pixel_values"].squeeze(0),
        )


class TestDecodeOneFixedCacheContract(unittest.TestCase):
    """Regression tests for the fixed-shape Gemma4 decode cache ABI."""

    def test_decode_keeps_non_shared_and_shared_kv_shapes_fixed(self):
        """Two decode steps should preserve all exported K/V input shapes."""
        from types import SimpleNamespace

        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            LayerCache,
            StaticGemma4Runtime,
        )

        class FakeDecodeLayer:
            """Record decode inputs and optionally produce one-token K/V deltas."""

            def __init__(
                self,
                *,
                is_shared: bool,
                store_full_length_kv: bool,
                delta_values: list[float] | None = None,
            ) -> None:
                self.is_shared = is_shared
                self.delta_values = delta_values or []
                self.call_index = 0
                self.past_shapes: list[tuple[int, ...]] = []
                self.shared_values: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.masks: list[torch.Tensor] = []
                attention = SimpleNamespace(
                    is_kv_shared_layer=is_shared,
                    store_full_length_kv=store_full_length_kv,
                )
                self.wrapped = SimpleNamespace(
                    self_attn=SimpleNamespace(wrapped=attention)
                )

            def __call__(
                self,
                *,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings,
                past_key_value=None,
                per_layer_input=None,
                shared_key_value=None,
            ):
                del position_embeddings, per_layer_input
                self.masks.append(attention_mask.detach().clone())
                if self.is_shared:
                    if past_key_value is not None or shared_key_value is None:
                        raise AssertionError(
                            "Shared layers require only shared_key_value."
                        )
                    self.shared_values.append(
                        tuple(tensor.detach().clone() for tensor in shared_key_value)
                    )
                    return hidden_states

                if past_key_value is None:
                    raise AssertionError("Non-shared layers require past_key_value.")
                self.past_shapes.append(tuple(past_key_value[0].shape))
                value = self.delta_values[self.call_index]
                self.call_index += 1
                new_k = torch.full(
                    (hidden_states.size(0), 1, 1, 2),
                    value,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                new_v = torch.full_like(new_k, -value)
                return hidden_states, new_k, new_v

        max_seq = 8
        owner = FakeDecodeLayer(
            is_shared=False,
            store_full_length_kv=True,
            delta_values=[9.0, 10.0],
        )
        consumer = FakeDecodeLayer(
            is_shared=True,
            store_full_length_kv=False,
        )

        def rotary_emb(x, position_ids, layer_type):
            del position_ids, layer_type
            shape = (x.size(0), 1, 2)
            return (
                torch.ones(shape, device=x.device, dtype=x.dtype),
                torch.zeros(shape, device=x.device, dtype=x.dtype),
            )

        runtime = object.__new__(StaticGemma4Runtime)
        runtime.device = torch.device("cpu")
        runtime.layout = SimpleNamespace(max_seq=max_seq)  # type: ignore[assignment]
        runtime.text_config = SimpleNamespace(
            layer_types=["full_attention", "full_attention"],
            sliding_window=4,
            final_logit_softcapping=None,
        )
        runtime.text_model = SimpleNamespace(hidden_size_per_layer_input=0)
        runtime.model = SimpleNamespace(  # type: ignore[assignment]
            model=SimpleNamespace(language_model=SimpleNamespace(rotary_emb=rotary_emb))
        )
        runtime.token_embedding = lambda ids: torch.zeros(  # type: ignore[assignment]
            ids.size(0), 1, 4, device=ids.device, dtype=torch.float32
        )
        runtime.lm_head = lambda hidden: torch.zeros(  # type: ignore[assignment]
            hidden.size(0), hidden.size(1), 5, device=hidden.device
        )
        runtime.decode_layers = [owner, consumer]  # type: ignore[assignment]
        runtime.past_len = 3

        initial_k = torch.zeros(1, 1, max_seq - 1, 2)
        initial_v = torch.zeros_like(initial_k)
        for index, value in enumerate((1.0, 2.0, 3.0)):
            initial_k[:, :, index, :] = value
            initial_v[:, :, index, :] = -value
        owner_cache = LayerCache(initial_k.clone(), initial_v.clone())
        consumer_cache = LayerCache(
            torch.zeros_like(initial_k), torch.zeros_like(initial_v)
        )
        runtime.layer_caches = [owner_cache, consumer_cache]

        runtime.decode_one(torch.tensor([[1]], dtype=torch.long))
        runtime.decode_one(torch.tensor([[2]], dtype=torch.long))

        self.assertEqual(
            owner.past_shapes,
            [(1, 1, max_seq - 1, 2), (1, 1, max_seq - 1, 2)],
        )
        self.assertEqual(len(consumer.shared_values), 2)
        self.assertEqual(tuple(consumer.shared_values[0][0].shape), (1, 1, max_seq, 2))
        self.assertEqual(tuple(consumer.shared_values[1][0].shape), (1, 1, max_seq, 2))

        expected_first = (
            torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0])
            .view(1, 1, max_seq, 1)
            .expand(-1, -1, -1, 2)
        )
        expected_second = (
            torch.tensor([1.0, 2.0, 3.0, 9.0, 0.0, 0.0, 0.0, 10.0])
            .view(1, 1, max_seq, 1)
            .expand(-1, -1, -1, 2)
        )
        torch.testing.assert_close(consumer.shared_values[0][0], expected_first)
        torch.testing.assert_close(consumer.shared_values[1][0], expected_second)
        torch.testing.assert_close(consumer.shared_values[0][1], -expected_first)
        torch.testing.assert_close(consumer.shared_values[1][1], -expected_second)

        self.assertTrue(torch.all(owner_cache.past_k[:, :, 3, :] == 9.0))
        self.assertTrue(torch.all(owner_cache.past_k[:, :, 4, :] == 10.0))
        self.assertEqual(runtime.past_len, 5)

        first_allowed = owner.masks[0][0, 0].eq(0.0)
        second_allowed = owner.masks[1][0, 0].eq(0.0)
        self.assertTrue(
            torch.equal(
                first_allowed,
                torch.tensor([True, True, True, False, False, False, False, True]),
            )
        )
        self.assertTrue(
            torch.equal(
                second_allowed,
                torch.tensor([True, True, True, True, False, False, False, True]),
            )
        )


if __name__ == "__main__":
    unittest.main()
