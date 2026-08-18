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

"""Regression tests for Gemma4 standalone vision export input contracts."""

import unittest

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import ForwardInput
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    Gemma4VisionAttentionCase,
    Gemma4VisionEncoderLayerCase,
    Gemma4VisionModelCase,
    Gemma4VisionPatchEmbedderCase,
    Gemma4VisionPoolerCase,
)


class _EncoderLayerExportOwner(torch.nn.Module):
    """Minimal wrapper owner used to verify static adapter selection."""

    def __init__(self) -> None:
        super().__init__()
        self.requested_mode: str | None = None

    def as_export_module(self, mode: str) -> torch.nn.Module:
        """Record the requested mode and return a simple exportable module."""
        self.requested_mode = mode
        return torch.nn.Identity()


class TestGemma4VisionExportInputContracts(unittest.TestCase):
    """Keep construction-only position tensors out of standalone export ABIs."""

    def setUp(self) -> None:
        """Create small tensors shared by all contract tests."""
        self.hidden = torch.randn(1, 4, 8)
        self.attention_mask = torch.zeros(1, 1, 4, 4)
        self.cos = torch.randn(1, 4, 2)
        self.sin = torch.randn(1, 4, 2)
        self.position_ids = torch.zeros(1, 4, 2, dtype=torch.long)
        self.padding_positions = torch.zeros(1, 4, dtype=torch.bool)

    def test_attention_export_input_omits_position_ids(self) -> None:
        """Standalone vision attention should export only tensors it consumes."""
        sample = ForwardInput(
            (),
            {
                "hidden_states": self.hidden,
                "position_embeddings": (self.cos, self.sin),
                "attention_mask": self.attention_mask,
                "position_ids": self.position_ids,
            },
        )

        export_input = Gemma4VisionAttentionCase().export_input(sample, {})

        self.assertEqual(dict(export_input.kwargs), {})
        self.assertEqual(len(export_input.args), 3)
        torch.testing.assert_close(export_input.args[0], self.hidden)
        torch.testing.assert_close(export_input.args[1][0], self.cos)
        torch.testing.assert_close(export_input.args[1][1], self.sin)
        torch.testing.assert_close(export_input.args[2], self.attention_mask)

    def test_encoder_layer_export_uses_static_adapter(self) -> None:
        """The encoder-layer smoke case should not export the eager wrapper."""
        owner = _EncoderLayerExportOwner()
        quantized = torch.nn.Module()
        quantized.wrapped = owner

        export_module = Gemma4VisionEncoderLayerCase().export_module(quantized, {})

        self.assertIsInstance(export_module, torch.nn.Identity)
        self.assertEqual(owner.requested_mode, "prefill")

    def test_encoder_layer_export_input_omits_position_ids(self) -> None:
        """Static encoder-layer export should omit compatibility-only ids."""
        sample = ForwardInput(
            (),
            {
                "hidden_states": self.hidden,
                "position_embeddings": (self.cos, self.sin),
                "attention_mask": self.attention_mask,
                "position_ids": self.position_ids,
            },
        )

        export_input = Gemma4VisionEncoderLayerCase().export_input(sample, {})

        self.assertEqual(dict(export_input.kwargs), {})
        self.assertEqual(len(export_input.args), 3)
        torch.testing.assert_close(export_input.args[0], self.hidden)
        torch.testing.assert_close(export_input.args[1], self.attention_mask)
        torch.testing.assert_close(export_input.args[2][0], self.cos)
        torch.testing.assert_close(export_input.args[2][1], self.sin)

    def test_patch_embedder_export_input_omits_baked_profile_tensors(self) -> None:
        """Static patch embedding should export only processor pixel values."""
        pixel_values = torch.randn(1, 4, 12)
        sample = ForwardInput(
            (pixel_values, self.position_ids, self.padding_positions),
        )

        export_input = Gemma4VisionPatchEmbedderCase().export_input(sample, {})

        self.assertEqual(dict(export_input.kwargs), {})
        self.assertEqual(len(export_input.args), 1)
        torch.testing.assert_close(export_input.args[0], pixel_values)

    def test_vision_model_export_input_omits_baked_position_ids(self) -> None:
        """The full static vision model should expose only pixel values."""
        pixel_values = torch.randn(1, 4, 12)
        sample = ForwardInput(
            (),
            {
                "pixel_values": pixel_values,
                "pixel_position_ids": self.position_ids,
                "return_dict": True,
            },
        )

        export_input = Gemma4VisionModelCase().export_input(sample, {})

        self.assertEqual(dict(export_input.kwargs), {})
        self.assertEqual(len(export_input.args), 1)
        torch.testing.assert_close(export_input.args[0], pixel_values)

    def test_pooler_export_input_omits_precomputed_position_ids(self) -> None:
        """Static pooler export should keep only hidden states and padding."""
        sample = ForwardInput(
            (),
            {
                "hidden_states": self.hidden,
                "pixel_position_ids": self.position_ids,
                "padding_positions": self.padding_positions,
                "output_length": 1,
            },
        )

        export_input = Gemma4VisionPoolerCase().export_input(sample, {})

        self.assertEqual(dict(export_input.kwargs), {})
        self.assertEqual(len(export_input.args), 2)
        torch.testing.assert_close(export_input.args[0], self.hidden)
        torch.testing.assert_close(export_input.args[1], self.padding_positions)


if __name__ == "__main__":
    unittest.main()
