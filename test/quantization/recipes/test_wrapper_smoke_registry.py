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

"""Registry tests for wrapper smoke cases."""

import unittest

from tico.quantization.recipes.debug.wrapper_smoke import case_names, get_case


class TestWrapperSmokeRegistry(unittest.TestCase):
    """Validate that legacy quantize_* examples have registered smoke cases."""

    def test_legacy_module_cases_are_registered(self):
        """Every migrated module-level quantize_* example should have a case."""
        expected = {
            "nn_linear",
            "nn_conv3d",
            "nn_conv3d_special_case",
            "nn_layernorm",
            "nn_tied_embedding",
            "llama_attention_prefill",
            "llama_attention_decode",
            "llama_mlp",
            "llama_decoder_layer_prefill",
            "llama_decoder_layer_decode",
            "qwen3_vl_text_attention_prefill",
            "qwen3_vl_text_attention_decode",
            "qwen3_vl_text_mlp",
            "qwen3_vl_text_decoder_layer",
            "qwen3_vl_text_model",
            "qwen3_vl_vision_attention",
            "qwen3_vl_vision_mlp",
            "qwen3_vl_vision_block",
            "qwen3_vl_vision_patch_embed",
            "qwen3_vl_vision_patch_merger",
            "qwen3_vl_vision_model",
            "qwen3_vl_model",
            "qwen3_vl_for_conditional_generation",
        }
        self.assertTrue(expected.issubset(set(case_names())))

    def test_get_case_returns_named_case(self):
        """Registry lookup should return the requested case object."""
        self.assertEqual(get_case("nn_linear").name, "nn_linear")


if __name__ == "__main__":
    unittest.main()
