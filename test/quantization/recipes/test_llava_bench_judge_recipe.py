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

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import unittest
from unittest.mock import patch

import tico.quantization.recipes.evaluation.llava_bench_judge as judge_recipe


class TestLlavaBenchJudgeRecipe(unittest.TestCase):
    def test_build_recipe_config_uses_nested_values(self):
        """Nested recipe mappings should populate the typed judge config."""
        llava_cfg = {
            "n_samples": "3",
            "start_index": 2,
            "max_seq_len": 2048,
            "max_new_tokens": 512,
            "temperature": 0.25,
            "image_min_pixels": 3136,
            "image_max_pixels": 602112,
            "resized_height": 704,
            "resized_width": 832,
            "visual_token_margin": 384,
            "candidate_label": "quant",
            "baseline_label": "fp",
            "candidate_answers": "candidate.jsonl",
            "baseline_answers": "baseline.jsonl",
            "output": {
                "dir": "./out/custom_llava",
                "answers": "answers.jsonl",
                "reviews": "reviews.jsonl",
                "summary": "summary.json",
            },
            "judge": {
                "enabled": "true",
                "model_id": "local-judge",
                "device": "cpu",
                "dtype": "bfloat16",
                "max_new_tokens": 128,
                "temperature": 0.1,
                "swap_order": "false",
            },
        }
        model_cfg = {
            "name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
            "trust_remote_code": False,
            "hf_token": "model-token",
        }
        runtime_cfg = {"show_progress": False}

        config = judge_recipe.build_llava_bench_judge_config(
            llava_cfg=llava_cfg,
            model_cfg=model_cfg,
            runtime_cfg=runtime_cfg,
            default_n_samples=50,
            default_max_seq_len=2048,
            default_device="cuda",
        )

        self.assertEqual(config.n_samples, 3)
        self.assertEqual(config.start_index, 2)
        self.assertEqual(config.max_seq_len, 2048)
        self.assertEqual(config.max_new_tokens, 512)
        self.assertEqual(config.temperature, 0.25)
        self.assertEqual(config.image_min_pixels, 3136)
        self.assertEqual(config.image_max_pixels, 602112)
        self.assertEqual(config.resized_height, 704)
        self.assertEqual(config.resized_width, 832)
        self.assertEqual(config.visual_token_margin, 384)
        self.assertEqual(config.candidate_label, "quant")
        self.assertEqual(config.baseline_label, "fp")
        self.assertEqual(config.candidate_answers_path, "candidate.jsonl")
        self.assertEqual(config.baseline_answers_path, "baseline.jsonl")
        self.assertEqual(config.output_dir, "./out/custom_llava")
        self.assertEqual(config.answers_out, "answers.jsonl")
        self.assertEqual(config.reviews_out, "reviews.jsonl")
        self.assertEqual(config.summary_out, "summary.json")
        self.assertTrue(config.judge_enabled)
        self.assertEqual(config.judge_model_id, "local-judge")
        self.assertEqual(config.judge_device, "cpu")
        self.assertEqual(config.judge_dtype, "bfloat16")
        self.assertEqual(config.judge_max_new_tokens, 128)
        self.assertEqual(config.judge_temperature, 0.1)
        self.assertFalse(config.judge_swap_order)
        self.assertFalse(config.trust_remote_code)
        self.assertEqual(config.hf_token, "model-token")
        self.assertTrue(config.quiet)

    def test_build_recipe_config_rejects_non_mapping_judge_config(self):
        """The judge config should fail clearly when it is not a mapping."""
        with self.assertRaises(TypeError):
            judge_recipe.build_llava_bench_judge_config(
                llava_cfg={"judge": True},
                model_cfg={},
                runtime_cfg={},
                default_n_samples=1,
                default_max_seq_len=None,
                default_device="cpu",
            )

    def test_evaluate_wrapper_invokes_core_and_summary_printer(self):
        """The recipe wrapper should build config, run evaluation, and print summary."""
        captured = {}
        summary = {"count": 1, "judge_model_id": "judge"}

        def fake_evaluate_llava_bench_with_judge(**kwargs):
            captured.update(kwargs)
            return summary

        with patch.object(
            judge_recipe,
            "evaluate_llava_bench_with_judge",
            fake_evaluate_llava_bench_with_judge,
        ), patch.object(judge_recipe, "print_llava_bench_judge_summary") as printer:
            result = judge_recipe.evaluate_and_print_llava_bench_judge(
                model="target-model",
                processor="target-processor",
                device="cpu",
                llava_cfg={
                    "enabled": True,
                    "mode": "judge",
                    "n_samples": 2,
                    "judge": {"enabled": False},
                },
                model_cfg={"name_or_path": "target-id"},
                runtime_cfg={"show_progress": True},
                default_n_samples=5,
                default_max_seq_len=128,
            )

        self.assertIs(result, summary)
        self.assertEqual(captured["model"], "target-model")
        self.assertEqual(captured["processor"], "target-processor")
        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(captured["config"].n_samples, 2)
        self.assertFalse(captured["config"].judge_enabled)
        printer.assert_called_once_with(summary)


if __name__ == "__main__":
    unittest.main()
