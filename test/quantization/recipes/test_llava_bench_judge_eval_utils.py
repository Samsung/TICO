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

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import tico.quantization.evaluation.llava_bench_judge_eval_utils as judge_utils

import torch
from tico.quantization.evaluation.llava_bench_judge_eval_utils import (
    LlavaBenchJudgeConfig,
)


class FakeImage:
    """Minimal PIL-like image used to validate pre-processor resize."""

    def __init__(self, size: tuple[int, int]):
        self.size = size
        self.resize_calls: list[tuple[tuple[int, int], Any]] = []

    def resize(self, size: tuple[int, int], resample: Any = None):
        """Return a new fake image with the requested size."""
        self.resize_calls.append((size, resample))
        return FakeImage(size)


class FakeTokenizer:
    """Minimal tokenizer for answer decoding."""

    def __init__(self):
        self.decoded_ids: list[int] = []
        self.pad_token_id = 0
        self.eos_token_id = 1

    def decode(self, ids, skip_special_tokens: bool = True):
        """Record generated IDs and return deterministic text."""
        self.decoded_ids = [int(value) for value in ids]
        return "decoded answer"


class FakePromptProcessor:
    """Minimal processor that records chat-template calls."""

    def __init__(self, input_len: int = 64):
        self.messages: list[dict[str, Any]] = []
        self.tokenize: bool | None = None
        self.add_generation_prompt: bool | None = None
        self.return_dict: bool | None = None
        self.processor_kwargs: dict[str, Any] = {}
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(patch_size=16, merge_size=2)
        self.input_len = input_len

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        """Record template arguments and return text or tokenized inputs."""
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.return_dict = kwargs.get("return_dict")
        self.processor_kwargs = kwargs
        if tokenize:
            return {
                "input_ids": torch.ones((1, self.input_len), dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 44, 52]]),
            }
        return "rendered prompt"


class FakeModel:
    """Minimal generation model used by unit tests."""

    def __init__(self):
        self.generate_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs):
        """Record generation kwargs and append two generated IDs."""
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        extra = torch.tensor([[201, 202]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)


class TestLlavaBenchJudgeEvalUtils(unittest.TestCase):
    def test_build_vlm_prompt_uses_original_question_without_short_answer_constraint(
        self,
    ):
        """LLaVA-Bench generation should not add short-answer instructions."""
        processor = FakePromptProcessor()
        question = "Describe the image and explain why the scene is unusual."

        prompt = judge_utils.build_vlm_prompt(processor, question)

        self.assertEqual(prompt, "rendered prompt")
        self.assertFalse(processor.tokenize)
        self.assertTrue(processor.add_generation_prompt)
        messages = cast(list[dict[str, Any]], processor.messages)
        content = cast(list[dict[str, str]], messages[0]["content"])
        text_part = content[1]["text"]
        self.assertEqual(text_part, question)
        self.assertNotIn("Return ONLY", text_part)
        self.assertNotIn("final answer", text_part)

    def test_image_is_resized_before_apply_chat_template(self):
        """Image pixel caps should resize PIL-like images before tokenization."""
        processor = FakePromptProcessor(input_len=600)
        image = FakeImage((2560, 2180))
        config = LlavaBenchJudgeConfig(
            max_seq_len=2048,
            max_new_tokens=512,
            image_max_pixels=602112,
        )

        inputs = judge_utils.build_vlm_processor_inputs(
            processor=processor,
            image=image,
            question="What is shown?",
            config=config,
            max_length=judge_utils._input_max_length(config),
        )

        resized_image = processor.messages[0]["content"][0]["image"]
        self.assertEqual(resized_image.size, (832, 704))
        self.assertEqual(image.resize_calls[0][0], (832, 704))
        self.assertEqual(int(inputs["input_ids"].shape[1]), 600)

    def test_generate_one_answer_uses_resized_image_and_reserved_budget(self):
        """Generation should decode only newly generated tokens after resizing."""
        processor = FakePromptProcessor(input_len=600)
        model = FakeModel()
        image = FakeImage((2560, 2180))
        config = LlavaBenchJudgeConfig(
            max_seq_len=2048,
            max_new_tokens=512,
            image_max_pixels=602112,
        )

        answer = judge_utils.generate_one_answer(
            model=model,
            processor=processor,
            image=image,
            question="What is shown?",
            device="cpu",
            config=config,
        )

        self.assertEqual(answer, "decoded answer")
        self.assertEqual(processor.tokenizer.decoded_ids, [201, 202])
        generate_kwargs = cast(dict[str, Any], model.generate_kwargs)
        self.assertEqual(generate_kwargs["max_new_tokens"], 512)
        self.assertFalse(generate_kwargs["do_sample"])

    def test_generate_one_answer_raises_when_processed_input_exceeds_budget(self):
        """Too-long samples should raise instead of being skipped."""
        processor = FakePromptProcessor(input_len=5462)
        model = FakeModel()
        image = FakeImage((2560, 2180))
        config = LlavaBenchJudgeConfig(
            max_seq_len=2048,
            max_new_tokens=512,
            image_max_pixels=602112,
        )

        with self.assertRaisesRegex(ValueError, "does not fit"):
            judge_utils.generate_one_answer(
                model=model,
                processor=processor,
                image=image,
                question="What is shown?",
                device="cpu",
                config=config,
            )

    def test_parse_judge_output_accepts_json_and_clamps_scores(self):
        """Judge output parsing should tolerate text around a JSON object."""
        raw = (
            "Here is my review:\n"
            '{"score_a": 12, "score_b": 8.5, "winner": "A", "reason": "A is better."}'
        )

        score_a, score_b, winner, reason = judge_utils.parse_judge_output(raw)

        self.assertEqual(score_a, 10.0)
        self.assertEqual(score_b, 8.5)
        self.assertEqual(winner, "A")
        self.assertEqual(reason, "A is better.")

    def test_judge_pair_maps_swapped_scores_to_canonical_labels(self):
        """Swapped prompts should be converted back to baseline/candidate scores."""
        config = LlavaBenchJudgeConfig(judge_swap_order=False)
        baseline = {
            "question_id": "q1",
            "prompt": "What is happening?",
            "text": "A reference-quality answer.",
            "reference_answer": "A reference-quality answer.",
            "context": "A person is holding an umbrella.",
            "model_id": "fp",
        }
        candidate = {
            "question_id": "q1",
            "prompt": "What is happening?",
            "text": "A candidate answer.",
            "reference_answer": "A reference-quality answer.",
            "context": "A person is holding an umbrella.",
            "model_id": "quant",
        }

        with patch.object(
            judge_utils,
            "run_judge_once",
            return_value=json.dumps(
                {
                    "score_a": 7,
                    "score_b": 3,
                    "winner": "A",
                    "reason": "A is more complete.",
                }
            ),
        ):
            result = judge_utils.judge_pair(
                tokenizer=object(),
                model=object(),
                config=config,
                baseline=baseline,
                candidate=candidate,
                swap=True,
            )

        self.assertEqual(result.score_baseline, 3.0)
        self.assertEqual(result.score_candidate, 7.0)
        self.assertEqual(result.winner, "candidate")
        self.assertEqual(result.baseline_label, "fp")
        self.assertEqual(result.candidate_label, "quant")
        self.assertTrue(result.metadata["swapped"])

    def test_summarize_reviews_computes_relative_score_and_writes_file(self):
        """Review summaries should include means, relative score, and wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            config = LlavaBenchJudgeConfig(
                summary_out=str(summary_path), judge_model_id="judge-model"
            )
            reviews = [
                {
                    "score_baseline": 8.0,
                    "score_candidate": 4.0,
                    "winner": "baseline",
                    "baseline_label": "fp",
                    "candidate_label": "quant",
                },
                {
                    "score_baseline": 6.0,
                    "score_candidate": 8.0,
                    "winner": "candidate",
                    "baseline_label": "fp",
                    "candidate_label": "quant",
                },
            ]

            summary = judge_utils.summarize_reviews(reviews=reviews, config=config)

            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["mean_baseline_score"], 7.0)
            self.assertEqual(summary["mean_candidate_score"], 6.0)
            self.assertAlmostEqual(summary["candidate_relative_score"], 85.7142857)
            self.assertEqual(summary["wins"], {"baseline": 1, "candidate": 1, "tie": 0})
            self.assertTrue(summary_path.exists())
            self.assertEqual(
                json.loads(summary_path.read_text())["judge_model_id"], "judge-model"
            )

    def test_input_max_length_reserves_generation_budget(self):
        """The input budget should reserve tokens for answer generation."""
        config = LlavaBenchJudgeConfig(max_seq_len=128, max_new_tokens=32)

        self.assertEqual(judge_utils._input_max_length(config), 96)

        bad_config = LlavaBenchJudgeConfig(max_seq_len=32, max_new_tokens=32)
        with self.assertRaises(ValueError):
            judge_utils._input_max_length(bad_config)


if __name__ == "__main__":
    unittest.main()
