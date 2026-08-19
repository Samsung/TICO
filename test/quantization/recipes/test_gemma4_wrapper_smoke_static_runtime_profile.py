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

"""Tests for the Gemma4 E2B static-runtime wrapper-smoke profile."""

import unittest
from types import SimpleNamespace
from typing import cast

import torch

from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    _has_gemma4,
    _padding_positions_from_ids,
    _static_pixel_position_ids,
    Gemma4ForCausalLMCase,
    Gemma4MultimodalEmbedderCase,
    Gemma4StaticRuntimeShape,
    Gemma4TextAttentionKEqVPrefillCase,
    Gemma4TextDecoderLayerDecodeCase,
    Gemma4TextDecoderLayerPrefillCase,
    Gemma4TextMLPCase,
    Gemma4VisionAttentionCase,
    Gemma4VisionModelCase,
    Gemma4VisionPoolerCase,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
    QuantGemma4VisionModel,
)


def _static_cfg(**overrides: int) -> dict:
    """Build a minimal E2B static-runtime wrapper-smoke config."""
    static_runtime = {
        "max_seq": 2_048,
        "profile": "e2b_66x36_264",
    }
    static_runtime.update(overrides)
    return {
        "debug": {
            "wrapper_smoke": {
                "gemma4": {
                    "size_profile": "e2b_static_runtime",
                    "static_runtime": static_runtime,
                }
            }
        }
    }


class TestGemma4StaticRuntimeShape(unittest.TestCase):
    """Validate the static text and vision shape contract without model weights."""

    def test_default_shape_matches_processor_contract(self):
        """The default profile should reproduce the fixed E2B image layout."""
        shape = Gemma4StaticRuntimeShape()

        self.assertEqual(shape.max_seq, 2_048)
        self.assertEqual(shape.visual_grid_height, 12)
        self.assertEqual(shape.visual_grid_width, 22)
        self.assertEqual(shape.patch_grid_height, 36)
        self.assertEqual(shape.patch_grid_width, 66)
        self.assertEqual(shape.num_valid_patches, 2_376)
        self.assertEqual(shape.num_patches, 2_520)
        self.assertEqual(shape.num_padding_patches, 144)
        self.assertEqual(shape.num_padding_soft_tokens, 16)

    def test_static_position_ids_encode_valid_grid_then_padding(self):
        """The synthetic layout should match Gemma4 processor padding semantics."""
        shape = Gemma4StaticRuntimeShape()
        position_ids = _static_pixel_position_ids(shape)
        padding = _padding_positions_from_ids(position_ids)

        self.assertEqual(tuple(position_ids.shape), (1, 2_520, 2))
        self.assertEqual(int(padding.sum().item()), 144)
        self.assertTrue(torch.equal(position_ids[0, 0], torch.tensor([0, 0])))
        last_valid = position_ids[0, shape.num_valid_patches - 1]
        self.assertTrue(torch.equal(last_valid, torch.tensor([65, 35])))
        self.assertTrue((position_ids[0, shape.num_valid_patches :] == -1).all())

    def test_invalid_static_layouts_are_rejected(self):
        """Malformed runtime profiles should fail before any module allocation."""
        with self.assertRaisesRegex(ValueError, "at least 2"):
            Gemma4StaticRuntimeShape(max_seq=1)
        with self.assertRaisesRegex(ValueError, "visual-token count"):
            Gemma4StaticRuntimeShape(num_visual_tokens=265)
        with self.assertRaisesRegex(ValueError, "patch_grid_width"):
            Gemma4StaticRuntimeShape(patch_grid_width=56)
        with self.assertRaisesRegex(ValueError, "processor-supported"):
            Gemma4StaticRuntimeShape(max_soft_tokens=300)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            Gemma4StaticRuntimeShape(num_visual_tokens=400, max_soft_tokens=280)

    def test_case_shape_helpers_select_static_dimensions(self):
        """Supported cases should map to the correct static tensor lengths."""
        cfg = _static_cfg()

        decode = Gemma4TextDecoderLayerDecodeCase()
        decode.validate_config(cfg)
        self.assertEqual(decode._decode_max_seq(default=8), 2_048)

        vision = Gemma4VisionModelCase()
        vision.validate_config(cfg)
        self.assertEqual(vision._vision_patch_seq_len(default=36), 2_520)

        pooler = Gemma4VisionPoolerCase()
        pooler.validate_config(cfg)
        self.assertEqual(pooler._vision_pool_output_length(default=4), 280)

        embedder = Gemma4MultimodalEmbedderCase()
        embedder.validate_config(cfg)
        self.assertEqual(embedder._visual_token_seq_len(default=16), 264)

    def test_profile_rejects_synthetic_or_unbounded_cases(self):
        """Static runtime should cover real bounded E2B branches only."""
        cfg = _static_cfg()
        with self.assertRaisesRegex(ValueError, "does not support"):
            Gemma4TextAttentionKEqVPrefillCase().validate_config(cfg)
        with self.assertRaisesRegex(ValueError, "does not support"):
            Gemma4ForCausalLMCase().validate_config(cfg)

    def test_static_circle_filename_is_distinct(self):
        """Static-runtime artifacts should not overwrite other profile outputs."""
        self.assertEqual(
            Gemma4TextMLPCase().export_filename(_static_cfg()),
            "gemma4_text_mlp.e2b_static_runtime.q.circle",
        )


_GEMMA4_AVAILABILITY = _has_gemma4()


@unittest.skipUnless(
    _GEMMA4_AVAILABILITY.available,
    _GEMMA4_AVAILABILITY.reason or "Gemma4 is unavailable",
)
class TestGemma4StaticRuntimeConfigs(unittest.TestCase):
    """Validate static-runtime config dimensions without constructing modules."""

    def test_text_config_enables_real_e2b_ple_width(self):
        """Static decoder cases should expose the external 256-wide PLE input."""
        case = Gemma4TextDecoderLayerPrefillCase()
        text_cfg = case._make_text_config(
            _static_cfg(),
            layer_types=("sliding_attention", "full_attention"),
            hidden_size_per_layer_input=256,
        )

        self.assertEqual(text_cfg.hidden_size, 1_536)
        self.assertEqual(text_cfg.intermediate_size, 6_144)
        self.assertEqual(text_cfg.hidden_size_per_layer_input, 256)
        self.assertEqual(text_cfg.sliding_window, 512)
        self.assertEqual(text_cfg.global_head_dim, 512)

    def test_vision_config_uses_e2b_width_without_standardization(self):
        """The real E2B vision config should retain standardize=False."""
        vision_cfg = Gemma4VisionAttentionCase()._make_vision_config(_static_cfg())

        self.assertEqual(vision_cfg.hidden_size, 768)
        self.assertEqual(vision_cfg.intermediate_size, 3_072)
        self.assertEqual(vision_cfg.num_hidden_layers, 1)
        self.assertEqual(vision_cfg.pooling_kernel_size, 3)
        self.assertEqual(vision_cfg.patch_size, 16)
        self.assertFalse(vision_cfg.standardize)


class TestQuantGemma4VisionModelObserverSelection(unittest.TestCase):
    """Ensure static export only requires observers on reachable branches."""

    def test_non_standardized_model_skips_standardization_observers(self):
        """E2B standardize=False should not require uncalibrated dead observers."""
        last_hidden = object()
        strip_padding = object()
        fake = cast(
            QuantGemma4VisionModel,
            SimpleNamespace(
                config=SimpleNamespace(standardize=False),
                obs_last_hidden_state=last_hidden,
                obs_strip_padding=strip_padding,
                obs_minus_bias=object(),
                obs_std_bias=None,
                obs_std_scale=None,
            ),
        )

        self.assertEqual(
            tuple(QuantGemma4VisionModel._all_observers(fake)),
            (last_hidden, strip_padding),
        )

    @unittest.skipUnless(
        _GEMMA4_AVAILABILITY.available,
        _GEMMA4_AVAILABILITY.reason or "Gemma4 is unavailable",
    )
    def test_non_standardized_forward_export_skips_standardization(self):
        """The real E2B export path should run without std buffers or observers."""
        hidden_size = 8
        seq_len = 4

        fake = cast(
            QuantGemma4VisionModel,
            SimpleNamespace(
                config=SimpleNamespace(standardize=False, hidden_size=hidden_size),
                patch_embedder_export=lambda pixels: pixels,
                encoder_export=lambda inputs_embeds: inputs_embeds,
                pooler_export=lambda **kwargs: kwargs["hidden_states"],
                output_length=seq_len,
                num_valid_pool_outputs=seq_len,
                obs_strip_padding=object(),
                obs_last_hidden_state=object(),
                obs_minus_bias=object(),
                obs_std_bias=None,
                obs_std_scale=None,
                _fq=lambda tensor, observer: tensor,
            ),
        )
        pixel_values = torch.randn(1, seq_len, hidden_size)

        output = QuantGemma4VisionModel.forward_export(fake, pixel_values)

        self.assertEqual(tuple(output.last_hidden_state.shape), (seq_len, hidden_size))

    def test_standardized_model_keeps_standardization_observers(self):
        """The existing standardized tiny path should retain all observers."""
        last_hidden = object()
        strip_padding = object()
        minus_bias = object()
        std_bias = object()
        std_scale = object()
        fake = cast(
            QuantGemma4VisionModel,
            SimpleNamespace(
                config=SimpleNamespace(standardize=True),
                obs_last_hidden_state=last_hidden,
                obs_strip_padding=strip_padding,
                obs_minus_bias=minus_bias,
                obs_std_bias=std_bias,
                obs_std_scale=std_scale,
            ),
        )

        self.assertEqual(
            tuple(QuantGemma4VisionModel._all_observers(fake)),
            (last_hidden, strip_padding, minus_bias, std_bias, std_scale),
        )


if __name__ == "__main__":
    unittest.main()
