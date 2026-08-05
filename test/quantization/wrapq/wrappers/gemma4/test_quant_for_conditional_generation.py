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

"""Smoke tests for QuantGemma4ForConditionalGeneration wrapper."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.quant_for_conditional_generation import (
    QuantGemma4ForConditionalGeneration,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 support."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4ForConditionalGeneration,
        )
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
        # Enable logit softcapping to exercise that code path.
        final_logit_softcapping=30.0,
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


def _make_gemma4_for_conditional_generation():
    """Create a tiny Gemma4ForConditionalGeneration for testing."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration,
    )

    config = _make_gemma4_config()
    return Gemma4ForConditionalGeneration(config).eval()


# ---------------------------------------------------------------------------
# Smoke tests for QuantGemma4ForConditionalGeneration forward
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4ForConditionalGenerationSmoke(unittest.TestCase):
    """Exercise QuantGemma4ForConditionalGeneration wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4ForConditionalGeneration modules."""
        torch.manual_seed(2026)
        self.fp_model = _make_gemma4_for_conditional_generation()
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

    def test_prepare_returns_correct_wrapper(self):
        """prepare() should return a PTQWrapper wrapping QuantGemma4ForConditionalGeneration."""
        qmodel = prepare(self.fp_model, self.qcfg)
        self.assertIsInstance(qmodel, PTQWrapper)
        self.assertIsInstance(qmodel.wrapped, QuantGemma4ForConditionalGeneration)

    def test_text_only_forward_is_finite(self):
        """Text-only forward through the wrapper should produce finite logits."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample)

        # The wrapper returns logits directly (not a Gemma4CausalLMOutputWithPast).
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_output_shape(self):
        """Forward output should have shape (batch, seq_len, vocab_size)."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample)

        self.assertEqual(
            out.shape,
            (self.batch_size, self.seq_len, self.text_config.vocab_size),
        )

    def test_logits_to_keep(self):
        """logits_to_keep=1 should produce logits only for the last position."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample, logits_to_keep=1)

        self.assertEqual(
            out.shape,
            (self.batch_size, 1, self.text_config.vocab_size),
        )

    def test_logit_softcapping_applied(self):
        """When final_logit_softcapping is set, logits should be bounded."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample)

        softcap = self.text_config.final_logit_softcapping
        if softcap is not None:
            self.assertTrue(out.abs().max().item() <= softcap + 1e-5)

    def test_prepare_convert_flow(self):
        """Full prepare → calibrate → convert flow should succeed."""
        qmodel = prepare(self.fp_model, self.qcfg)
        self.assertIsInstance(qmodel, PTQWrapper)
        self.assertIsInstance(qmodel.wrapped, QuantGemma4ForConditionalGeneration)

        sample = self._text_only_sample()
        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            out = quantized(**sample)

        self.assertEqual(
            out.shape,
            (self.batch_size, self.seq_len, self.text_config.vocab_size),
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_observers_calibrated(self):
        """After convert, the softcapping and logits observers should have qparams."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        wrapped = getattr(quantized, "wrapped", quantized)
        for obs in wrapped._all_observers():
            self.assertTrue(
                obs.has_qparams,
                f"Observer {obs.name} was not calibrated",
            )


# ---------------------------------------------------------------------------
# Tests for Generation support in QuantGemma4ForConditionalGeneration
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4ForConditionalGenerationGeneration(unittest.TestCase):
    """Exercise GenerationMixin support added to QuantGemma4ForConditionalGeneration."""

    def setUp(self):
        """Create deterministic tiny Gemma4ForConditionalGeneration modules."""
        torch.manual_seed(2026)
        self.fp_model = _make_gemma4_for_conditional_generation()
        self.fp_ref = copy.deepcopy(self.fp_model).eval()
        self.config = self.fp_model.config
        self.text_config = self.config.get_text_config()
        self.seq_len = 4
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
        """Create a text-only sample (no image/video/audio tokens).

        Token IDs 10, 11, 12 are reserved for image, video, and audio
        placeholders.  We exclude them so the multimodal validation in
        ``QuantGemma4Model`` does not reject the input.
        """
        special_ids = {
            self.config.image_token_id,
            self.config.video_token_id,
            self.config.audio_token_id,
        }
        valid_ids = [
            i for i in range(self.text_config.vocab_size) if i not in special_ids
        ]
        idx = torch.randint(0, len(valid_ids), (self.batch_size, self.seq_len))
        input_ids = torch.tensor(valid_ids)[idx]
        return {"input_ids": input_ids}

    def _make_quantized(self):
        """prepare → calibrate → convert and return the quantized wrapper."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()
        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)
        return convert(qmodel)

    # --- End-to-end generation ---------------------------------------------

    def test_generate_returns_tensor(self):
        """generate() should return a tensor of token ids."""
        quantized = self._make_quantized()
        sample = self._text_only_sample()

        with torch.no_grad():
            out = quantized.generate(
                **sample,
                max_new_tokens=1,
                do_sample=False,
            )

        self.assertIsInstance(out, torch.Tensor)
        # Output should be at least as long as the input.
        self.assertGreaterEqual(out.shape[1], sample["input_ids"].shape[1])

    def test_generate_extends_sequence(self):
        """generate(max_new_tokens=N) should add exactly N new tokens."""
        quantized = self._make_quantized()
        sample = self._text_only_sample()
        max_new = 1

        with torch.no_grad():
            out = quantized.generate(
                **sample,
                max_new_tokens=max_new,
                do_sample=False,
            )

        self.assertEqual(out.shape[1], self.seq_len + max_new)


if __name__ == "__main__":
    unittest.main()
