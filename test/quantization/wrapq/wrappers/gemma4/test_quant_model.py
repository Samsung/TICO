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

"""Smoke tests for QuantGemma4Model wrapper and _get_placeholder_mask helper."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.quant_model import (
    _get_placeholder_mask,
    QuantGemma4Model,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 support."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model  # noqa: F401
    except Exception:
        return False
    return True


def _make_vision_config():
    """Create a tiny Gemma4 vision config for synthetic smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        patch_size=4,
        position_embedding_size=8,
        pooling_kernel_size=2,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_clipped_linears=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
        standardize=True,
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _make_text_config():
    """Create a tiny Gemma4 text config for synthetic smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0}
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        enable_moe_block=False,
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _make_gemma4_config():
    """Create a tiny Gemma4 top-level config for synthetic smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    return Gemma4Config(
        text_config=_make_text_config(),
        vision_config=_make_vision_config(),
        audio_config=None,
        image_token_id=10,
        video_token_id=11,
        audio_token_id=12,
    )


def _make_gemma4_model():
    """Create a tiny Gemma4Model for testing."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

    config = _make_gemma4_config()
    return Gemma4Model(config).eval()


# ---------------------------------------------------------------------------
# Unit tests for _get_placeholder_mask (no model required)
# ---------------------------------------------------------------------------


class TestGetPlaceholderMask(unittest.TestCase):
    """Test the _get_placeholder_mask helper function."""

    def _make_config(
        self,
        image_token_id: int | None = 10,
        video_token_id: int | None = 11,
        audio_token_id: int | None = 12,
    ):
        """Create a simple config-like object with token IDs."""
        from types import SimpleNamespace

        return SimpleNamespace(
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
        )

    def test_no_placeholders(self):
        """All-text input should produce all-False masks."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        cfg = self._make_config()
        img, vid, aud = _get_placeholder_mask(input_ids, cfg)
        self.assertFalse(img.any())
        self.assertFalse(vid.any())
        self.assertFalse(aud.any())

    def test_image_placeholders(self):
        """Image token IDs should be marked in the image mask only."""
        input_ids = torch.tensor([[1, 10, 10, 4]])
        cfg = self._make_config()
        img, vid, aud = _get_placeholder_mask(input_ids, cfg)
        self.assertTrue(torch.equal(img, torch.tensor([[False, True, True, False]])))
        self.assertFalse(vid.any())
        self.assertFalse(aud.any())

    def test_mixed_placeholders(self):
        """Each placeholder type should appear in its own mask only."""
        input_ids = torch.tensor([[10, 11, 12, 1]])
        cfg = self._make_config()
        img, vid, aud = _get_placeholder_mask(input_ids, cfg)
        self.assertTrue(torch.equal(img, torch.tensor([[True, False, False, False]])))
        self.assertTrue(torch.equal(vid, torch.tensor([[False, True, False, False]])))
        self.assertTrue(torch.equal(aud, torch.tensor([[False, False, True, False]])))

    def test_missing_token_ids(self):
        """When a token ID is None, the corresponding mask should be all-False."""
        input_ids = torch.tensor([[1, 2, 3]])
        cfg = self._make_config(
            image_token_id=None, video_token_id=None, audio_token_id=None
        )
        img, vid, aud = _get_placeholder_mask(input_ids, cfg)
        self.assertFalse(img.any())
        self.assertFalse(vid.any())
        self.assertFalse(aud.any())


# ---------------------------------------------------------------------------
# Unit tests for QuantGemma4Model input validation
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4ModelValidation(unittest.TestCase):
    """Test input validation in QuantGemma4Model.forward."""

    def setUp(self):
        """Create a tiny Gemma4Model and wrap it."""
        torch.manual_seed(2026)
        self.fp_model = _make_gemma4_model()
        self.qcfg = PTQConfig(
            model_args={
                "vision": {
                    "visual_start_idx": 0,
                    "num_visual_tokens": 4,
                }
            }
        )
        self.qmodel = prepare(self.fp_model, self.qcfg).eval()

    def test_reject_both_input_ids_and_inputs_embeds(self):
        """Providing both input_ids and inputs_embeds should raise ValueError."""
        input_ids = torch.tensor([[1, 2, 3]])
        inputs_embeds = torch.randn(1, 3, 64)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.qmodel(input_ids=input_ids, inputs_embeds=inputs_embeds)

    def test_reject_neither_input_ids_nor_inputs_embeds(self):
        """Providing neither input_ids nor inputs_embeds should raise ValueError."""
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.qmodel()

    def test_reject_input_ids_with_per_layer_inputs(self):
        """Providing both input_ids and per_layer_inputs should raise ValueError."""
        input_ids = torch.tensor([[1, 2, 3]])
        per_layer_inputs = torch.randn(1, 3, 1, 256)
        with self.assertRaisesRegex(ValueError, "per_layer_inputs"):
            self.qmodel(input_ids=input_ids, per_layer_inputs=per_layer_inputs)


# ---------------------------------------------------------------------------
# Smoke tests for QuantGemma4Model forward
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4ModelSmoke(unittest.TestCase):
    """Exercise QuantGemma4Model wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4Model modules."""
        torch.manual_seed(2026)
        self.fp_model = _make_gemma4_model()
        self.fp_ref = copy.deepcopy(self.fp_model).eval()
        self.config = self.fp_model.config
        self.text_config = self.config.get_text_config()
        self.seq_len = 8
        self.batch_size = 1
        self.qcfg = PTQConfig(
            model_args={
                "vision": {
                    "visual_start_idx": 0,
                    "num_visual_tokens": 4,
                }
            }
        )

    def _text_only_sample(self):
        """Create a text-only sample (no image tokens)."""
        return {
            "input_ids": torch.randint(
                0, self.text_config.vocab_size, (self.batch_size, self.seq_len)
            ),
        }

    def _text_with_image_sample(self):
        """Create a sample with image placeholder tokens at the start."""
        input_ids = torch.randint(
            0, self.text_config.vocab_size, (self.batch_size, self.seq_len)
        )
        # Place image tokens at positions 0..3
        input_ids[0, :4] = self.config.image_token_id
        return {
            "input_ids": input_ids,
        }

    def test_text_only_forward_is_finite(self):
        """Text-only forward through the wrapper should produce finite output."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample)

        # The output is a Gemma4TextModelOutputWithPast; check last_hidden_state
        hidden = out.last_hidden_state
        self.assertTrue(torch.isfinite(hidden).all())

    def test_placeholder_replacement_produces_pad_embedding(self):
        """Image placeholder tokens should be replaced with pad_token_id before embedding."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_with_image_sample()
        input_ids = sample["input_ids"]

        # Calibrate and convert
        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)
        quantized = convert(qmodel)

        # After quantization, the forward should still work with image tokens
        with torch.no_grad():
            out = quantized(**sample)

        hidden = out.last_hidden_state
        self.assertTrue(torch.isfinite(hidden).all())

    def test_prepare_convert_flow(self):
        """Full prepare → calibrate → convert flow should succeed."""
        qmodel = prepare(self.fp_model, self.qcfg)
        self.assertIsInstance(qmodel, PTQWrapper)
        self.assertIsInstance(qmodel.wrapped, QuantGemma4Model)

        sample = self._text_only_sample()
        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            out = quantized(**sample)

        hidden = out.last_hidden_state
        self.assertEqual(
            hidden.shape,
            (self.batch_size, self.seq_len, self.text_config.hidden_size),
        )
        self.assertTrue(torch.isfinite(hidden).all())

    def test_inputs_embeds_without_per_layer_inputs_raises(self):
        """When PLE is enabled and inputs_embeds is given without per_layer_inputs, it should raise."""
        # This test only applies when hidden_size_per_layer_input > 0
        if not self.text_config.hidden_size_per_layer_input:
            self.skipTest("PLE not enabled for this config")

        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        # Calibrate first
        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)
        quantized = convert(qmodel)

        # Now call with inputs_embeds but no per_layer_inputs
        inputs_embeds = torch.randn(
            self.batch_size, self.seq_len, self.text_config.hidden_size
        )
        with self.assertRaises((ValueError, RuntimeError)):
            with torch.no_grad():
                quantized(inputs_embeds=inputs_embeds)

    def test_inputs_embeds_with_per_layer_inputs_works(self):
        """When PLE is enabled and both inputs_embeds and per_layer_inputs are given, it should work."""
        if not self.text_config.hidden_size_per_layer_input:
            self.skipTest("PLE not enabled for this config")

        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        # Calibrate
        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)
        quantized = convert(qmodel)

        # Call with both inputs_embeds and per_layer_inputs
        inputs_embeds = torch.randn(
            self.batch_size, self.seq_len, self.text_config.hidden_size
        )
        per_layer_inputs = torch.randn(
            self.batch_size,
            self.seq_len,
            self.text_config.num_hidden_layers,
            self.text_config.hidden_size_per_layer_input,
        )
        with torch.no_grad():
            out = quantized(
                inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs
            )

        hidden = out.last_hidden_state
        self.assertTrue(torch.isfinite(hidden).all())


if __name__ == "__main__":
    unittest.main()
