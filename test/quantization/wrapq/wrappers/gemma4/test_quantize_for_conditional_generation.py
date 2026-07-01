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

"""Smoke tests for Gemma4ForConditionalGeneration prepare-calibrate-convert flow."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 support."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4Config,
            Gemma4TextConfig,
            Gemma4VisionConfig,
        )
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
    """Create a tiny Gemma4 top-level config for smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    return Gemma4Config(
        text_config=_make_text_config(),
        vision_config=_make_vision_config(),
        audio_config=None,
        image_token_id=10,
        video_token_id=11,
        audio_token_id=12,
    )


def _pixel_position_ids(batch_size: int, num_patches: int) -> torch.Tensor:
    """Create deterministic 2D pixel position ids for a patch grid."""
    side = int(num_patches**0.5)
    coords = torch.arange(num_patches)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4ForConditionalGenerationSmoke(unittest.TestCase):
    """Exercise Gemma4ForConditionalGeneration wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4ForConditionalGeneration modules."""
        torch.manual_seed(2026)
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )

        self.cfg = _make_gemma4_config()
        self.text_cfg = self.cfg.get_text_config()
        self.vision_cfg = self.cfg.vision_config
        self.fp_model = Gemma4ForConditionalGeneration(self.cfg).eval()
        self.fp_ref = copy.deepcopy(self.fp_model).eval()
        self.batch_size = 1
        self.seq_len = 16
        self.num_patches = 16
        self.patch_size = self.vision_cfg.patch_size
        self.num_visual_tokens = 4

    def _ptq_config(self):
        """Build the PTQ config used by smoke tests."""
        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_cfg.num_hidden_layers),
            num_vision_layers=int(self.vision_cfg.num_hidden_layers),
            model_args={
                "vision": {
                    "visual_start_idx": 0,
                    "num_visual_tokens": self.num_visual_tokens,
                }
            },
        )

    def _text_sample(self):
        """Create one synthetic text-only input (no image placeholders).

        Token IDs are kept in [0, 9] to avoid colliding with the image
        placeholder token ID (10).
        """
        return {
            "input_ids": torch.randint(0, 10, (self.batch_size, self.seq_len)),
        }

    def _image_text_sample(self):
        """Create one synthetic image-text input with image placeholders."""
        input_ids = torch.randint(0, 10, (self.batch_size, self.seq_len))
        input_ids[0, : self.num_visual_tokens] = self.cfg.image_token_id
        patch_dim = 3 * self.patch_size**2
        return {
            "input_ids": input_ids,
            "pixel_values": torch.randn(self.batch_size, self.num_patches, patch_dim),
            "image_position_ids": _pixel_position_ids(
                self.batch_size, self.num_patches
            ),
        }

    def test_no_quant_model_matches_reference(self):
        """The wrapper should match the floating-point module before quantization."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_for_conditional_generation import (
            QuantGemma4ForConditionalGeneration,
        )

        wrapped = QuantGemma4ForConditionalGeneration(
            self.fp_model, qcfg=self._ptq_config()
        ).eval()
        sample = self._text_sample()

        with torch.no_grad():
            quant_out = wrapped(**sample)
            fp_out = self.fp_ref(**sample).logits

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_prepare_convert_flow(self):
        """Quantize Gemma4ForConditionalGeneration and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_for_conditional_generation import (
            QuantGemma4ForConditionalGeneration,
        )

        prepared = prepare(self.fp_model, self._ptq_config())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4ForConditionalGeneration)

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._text_sample())

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        sample = self._text_sample()
        with torch.no_grad():
            quant_out = quantized(**sample)
            fp_out = self.fp_ref(**sample).logits

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())

    def test_prepare_convert_flow_with_image(self):
        """Quantize Gemma4ForConditionalGeneration with image-text calibration data."""
        prepared = prepare(self.fp_model, self._ptq_config())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._image_text_sample())

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        sample = self._image_text_sample()
        with torch.no_grad():
            quant_out = quantized(**sample)
            fp_out = self.fp_ref(**sample).logits

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())

    def test_logits_to_keep(self):
        """logits_to_keep=1 should produce logits only for the last position."""
        prepared = prepare(self.fp_model, self._ptq_config())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._text_sample())

        quantized = convert(prepared)

        sample = self._text_sample()
        with torch.no_grad():
            quant_out = quantized(**sample, logits_to_keep=1)
            fp_out = self.fp_ref(**sample, logits_to_keep=1).logits

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertEqual(quant_out.shape[1], 1)
        self.assertTrue(torch.isfinite(quant_out).all())

    def test_logit_softcapping_bounded(self):
        """When final_logit_softcapping is set, logits should be bounded."""
        prepared = prepare(self.fp_model, self._ptq_config())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._text_sample())

        quantized = convert(prepared)

        sample = self._text_sample()
        with torch.no_grad():
            quant_out = quantized(**sample)

        softcap = self.text_cfg.final_logit_softcapping
        if softcap is not None:
            self.assertTrue(quant_out.abs().max().item() <= softcap + 1e-5)


if __name__ == "__main__":
    unittest.main()
