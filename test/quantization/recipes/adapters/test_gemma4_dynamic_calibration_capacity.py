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

"""Real CPU calibration budgets independent of the Gemma static export profile."""

import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
    QuantGemma4TextModel,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


class TestGemma4DynamicCalibrationCapacity(unittest.TestCase):
    def _context(self):
        torch.manual_seed(1073)
        config = Gemma4Config.from_dict(
            {
                "text_config": {
                    "vocab_size": 32,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "num_global_key_value_heads": 2,
                    "head_dim": 4,
                    "global_head_dim": 4,
                    "max_position_embeddings": 32,
                    "layer_types": ["full_attention", "full_attention"],
                    "rope_parameters": {
                        "full_attention": {
                            "rope_type": "default",
                            "rope_theta": 10000.0,
                        }
                    },
                    "hidden_size_per_layer_input": 4,
                    "vocab_size_per_layer_input": 32,
                    "num_kv_shared_layers": 1,
                    "enable_moe_block": False,
                    "attention_k_eq_v": False,
                    "attention_dropout": 0.0,
                    "use_cache": False,
                },
                "vision_config": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "patch_size": 4,
                    "position_embedding_size": 8,
                    "pooling_kernel_size": 2,
                    "max_position_embeddings": 128,
                    "use_clipped_linears": False,
                    "standardize": True,
                    "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
                },
                "audio_config": None,
                "image_token_id": 32,
                "video_token_id": 33,
                "audio_token_id": 34,
            }
        )
        config._attn_implementation = "eager"
        model = Gemma4ForConditionalGeneration(config).eval()
        model_args = {"text": {"max_seq": 8}, "vision": {"num_visual_tokens": 4}}
        ids = torch.arange(1, 15).reshape(1, 14)
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 16, "decode_steps": 2},
                "pipeline": [{"name": "ptq"}],
                "model_args": model_args,
                "runtime": {"show_progress": False},
            },
            adapter=Gemma4Adapter(),
            model=model,
            calibration_inputs=[
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            ],
        )
        return ctx, prepare(model, PTQConfig(model_args=model_args))

    def test_raw_model_calibration_budget_ignores_static_profile(self):
        """Data preparation reserves fourteen native eager tokens plus two decodes."""
        ctx, _prepared = self._context()
        self.assertEqual(ctx.adapter._decode_calibration_capacity(ctx), 16)

    def test_prepared_calibration_exceeds_static_profile_with_real_cache(self):
        """A real PLE/shared-KV wrapper calibrates fourteen plus two tokens at S=8."""
        ctx, prepared = self._context()
        lengths: list[int] = []
        handle = prepared.register_forward_hook(
            lambda _module, _args, output: lengths.append(
                int(output.past_key_values.get_seq_length())
            )
        )
        try:
            ctx.adapter.calibrate_prepared_model(ctx, prepared, {"name": "ptq"})
        finally:
            handle.remove()
        self.assertEqual(lengths, [14, 15, 16])
        convert(prepared)

    def test_native_and_recipe_limits_still_reject_overflow_before_forward(self):
        """Separating the static profile does not relax either real token budget."""
        ctx, prepared = self._context()
        for budget, length in ((16, 15), (64, 31)):
            with self.subTest(budget=budget, prompt=length):
                ctx.cfg["calibration"]["seq_len"] = budget
                ctx.calibration_inputs = [
                    {"input_ids": torch.ones(1, length, dtype=torch.long)}
                ]
                with self.assertRaisesRegex(ValueError, "calibration context budget"):
                    ctx.adapter.calibrate_prepared_model(ctx, prepared, {"name": "ptq"})

    def test_explicit_static_execution_retains_its_capacity(self):
        """An explicitly static text wrapper still uses the eight-token profile."""
        ctx, prepared = self._context()
        for module in prepared.modules():
            if isinstance(module, QuantGemma4TextModel):
                module.force_export = True
        self.assertEqual(ctx.adapter._decode_calibration_capacity(ctx, prepared), 8)


if __name__ == "__main__":
    unittest.main()
