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
from types import SimpleNamespace
from unittest.mock import DEFAULT, patch

import tico.quantization.recipes.adapters.gemma4 as gemma4_mod
import tico.quantization.recipes.adapters.llama as llama_mod
import tico.quantization.recipes.adapters.qwen3_vl as qwen3_vl_mod

import torch
from tico.quantization.recipes.context import RecipeContext


class TestEvaluationTargetAdapters(unittest.TestCase):
    """Tests for adapter-level exclusive evaluation target selection."""

    def _make_llama_context(
        self,
        evaluation: dict,
    ) -> RecipeContext:
        """Build a lightweight LLaMA context without loading a checkpoint."""
        adapter = llama_mod.LlamaAdapter()
        ctx = RecipeContext(
            cfg={
                "model": {"family": "llama"},
                "calibration": {"seq_len": 2048},
                "evaluation": evaluation,
            },
            adapter=adapter,
        )
        ctx.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=2048)
        )
        ctx.tokenizer = object()
        ctx.device = torch.device("cpu")
        return ctx

    def test_llama_selected_lm_eval_skips_enabled_ppl(self):
        """Selecting lm_eval should skip an otherwise configured perplexity run."""
        ctx = self._make_llama_context(
            {
                "enabled": True,
                "selected_tasks": ["lm_eval"],
                "perplexity": {"dataset": "Salesforce/wikitext"},
                "lm_eval_tasks": "mmlu,hellaswag",
                "max_seq_len": 2048,
            }
        )

        with patch.object(
            llama_mod,
            "evaluate_perplexity",
        ) as evaluate_perplexity, patch.object(
            llama_mod,
            "evaluate_lm_tasks",
        ) as evaluate_lm_tasks:
            ctx.adapter.evaluate(ctx)

        evaluate_perplexity.assert_not_called()
        evaluate_lm_tasks.assert_called_once()
        self.assertEqual(
            evaluate_lm_tasks.call_args.kwargs["tasks"],
            "mmlu,hellaswag",
        )

    def test_llama_selected_ppl_uses_defaults_when_config_is_null(self):
        """Selecting ppl should enable the evaluator while preserving default details."""
        ctx = self._make_llama_context(
            {
                "enabled": True,
                "selected_tasks": ["ppl"],
                "perplexity": None,
                "lm_eval_tasks": "mmlu",
                "max_seq_len": 2048,
            }
        )

        with patch.object(
            llama_mod,
            "evaluate_perplexity",
            return_value=1.25,
        ) as evaluate_perplexity, patch.object(
            llama_mod,
            "evaluate_lm_tasks",
        ) as evaluate_lm_tasks:
            ctx.adapter.evaluate(ctx)

        evaluate_lm_tasks.assert_not_called()
        evaluate_perplexity.assert_called_once()
        self.assertEqual(
            evaluate_perplexity.call_args.kwargs["dataset_name"],
            "Salesforce/wikitext",
        )
        self.assertEqual(
            evaluate_perplexity.call_args.kwargs["dataset_config"],
            "wikitext-2-raw-v1",
        )

    def _make_vlm_context(
        self,
        adapter,
    ) -> RecipeContext:
        """Build a lightweight VLM context with every non-selected target enabled."""
        evaluation = {
            "enabled": True,
            "selected_tasks": ["mmmu", "ppl"],
            "vlm_tasks": ["vqav2"],
            "coco": True,
            "llava_bench": {"enabled": True, "mode": "judge"},
            "videomme": {"enabled": True},
            "mmlu": {"enabled": True},
            "hellaswag": {"enabled": True},
            "mmmu": {
                "enabled": False,
                "dataset": "MMMU/MMMU_Pro",
                "subjects": ["vision"],
            },
            "ppl": {
                "enabled": False,
                "dataset": "wikitext2",
                "split": "test",
            },
            "n_samples": 3,
            "max_seq_len": 2048,
        }
        ctx = RecipeContext(
            cfg={
                "model": {"family": adapter.family},
                "runtime": {"show_progress": False},
                "calibration": {"seq_len": 2048},
                "evaluation": evaluation,
            },
            adapter=adapter,
        )
        ctx.processor = SimpleNamespace(tokenizer=object())
        ctx.model = SimpleNamespace(
            config=SimpleNamespace(
                use_cache=False,
                text_config=SimpleNamespace(use_cache=False),
            )
        )
        ctx.device = torch.device("cpu")
        return ctx

    def test_gemma4_selected_ppl_preserves_chat_prefix_mode(self):
        """Gemma4 should preserve chat-prefix PPL when ppl is selected."""
        adapter = gemma4_mod.Gemma4Adapter()
        ctx = self._make_vlm_context(adapter)
        ctx.cfg["evaluation"]["selected_tasks"] = ["ppl"]
        ctx.cfg["evaluation"]["ppl"]["mode"] = "chat-prefix"

        with patch.object(
            gemma4_mod,
            "evaluate_vlm_text_ppl",
        ) as evaluate_raw_ppl, patch.object(
            gemma4_mod,
            "evaluate_vlm_text_ppl_chat_prefix",
            return_value=2.5,
        ) as evaluate_chat_prefix_ppl:
            adapter.evaluate(ctx)

        evaluate_raw_ppl.assert_not_called()
        evaluate_chat_prefix_ppl.assert_called_once()
        self.assertEqual(
            evaluate_chat_prefix_ppl.call_args.kwargs["dataset_name"],
            "wikitext2",
        )
        self.assertEqual(
            evaluate_chat_prefix_ppl.call_args.kwargs["stride"],
            512,
        )

    def test_vlm_adapters_run_only_selected_targets(self):
        """Qwen3-VL and Gemma4 should run only the selected top-level targets."""
        cases = (
            (qwen3_vl_mod, qwen3_vl_mod.Qwen3VLAdapter()),
            (gemma4_mod, gemma4_mod.Gemma4Adapter()),
        )

        for module, adapter in cases:
            with self.subTest(family=adapter.family):
                ctx = self._make_vlm_context(adapter)
                with patch.multiple(
                    module,
                    evaluate_vqa_tasks=DEFAULT,
                    evaluate_coco=DEFAULT,
                    evaluate_and_print_llava_bench_judge=DEFAULT,
                    evaluate_llava_bench=DEFAULT,
                    evaluate_and_print_video_mme=DEFAULT,
                    evaluate_and_print_mmlu=DEFAULT,
                    evaluate_and_print_hellaswag=DEFAULT,
                    evaluate_and_print_mmmu=DEFAULT,
                    evaluate_vlm_text_ppl=DEFAULT,
                ) as mocks:
                    mocks["evaluate_vlm_text_ppl"].return_value = 2.0
                    adapter.evaluate(ctx)

                mocks["evaluate_and_print_mmmu"].assert_called_once()
                mocks["evaluate_vlm_text_ppl"].assert_called_once()
                for skipped_name in (
                    "evaluate_vqa_tasks",
                    "evaluate_coco",
                    "evaluate_and_print_llava_bench_judge",
                    "evaluate_llava_bench",
                    "evaluate_and_print_video_mme",
                    "evaluate_and_print_mmlu",
                    "evaluate_and_print_hellaswag",
                ):
                    mocks[skipped_name].assert_not_called()


if __name__ == "__main__":
    unittest.main()
