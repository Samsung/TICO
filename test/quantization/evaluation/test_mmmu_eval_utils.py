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

import sys
import unittest
from unittest.mock import Mock, patch

from tico.quantization.evaluation.mmmu_eval_utils import (
    DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
    evaluate_subject,
    extract_answer,
    get_mmmu_pro_vision_prompt,
    load_data,
    MMMU_PRO_VISION_PROMPTS,
    resolve_mmmu_max_new_tokens,
)


class TestExtractAnswer(unittest.TestCase):
    def test_extracts_supported_answer_formats(self):
        cases = {
            "C": "C",
            "C.": "C",
            "(D)": "D",
            "Answer: C": "C",
            "The answer is C.": "C",
            "Option (J)": "J",
            "I think the answer is C.": "C",
            "I would choose B.": "B",
            "A. This is the first option.": "A",
            "Reasoning on earlier lines.\nC": "C",
            "**Answer: C**": "C",
        }

        for generated, expected in cases.items():
            with self.subTest(generated=generated):
                self.assertEqual(extract_answer(generated), expected)

    def test_uses_last_explicit_answer(self):
        generated = "Option A might fit, but after checking, the answer is C."
        self.assertEqual(extract_answer(generated), "C")

    def test_does_not_treat_leading_word_as_answer(self):
        self.assertIsNone(extract_answer("I think this is ambiguous."))

    def test_returns_none_for_empty_or_unparseable_output(self):
        self.assertIsNone(extract_answer(""))
        self.assertIsNone(extract_answer("The image contains a diagram."))


class TestMmmuDataLoading(unittest.TestCase):
    def test_missing_datasets_dependency_is_reported_at_runtime(self):
        with patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaisesRegex(
                RuntimeError,
                "optional 'datasets' package is required",
            ):
                load_data(
                    dataset="MMMU/MMMU_Pro",
                    subject="vision",
                    split="test",
                    n_samples=1,
                )


class TestMmmuProVisionPrompt(unittest.TestCase):
    def test_official_direct_is_default_prompt(self):
        self.assertEqual(
            DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
            "official_direct",
        )
        self.assertEqual(
            get_mmmu_pro_vision_prompt(DEFAULT_MMMU_PRO_VISION_PROMPT_MODE),
            (
                "Answer with the option letter from the given choices directly. "
                "The last line of your response should be of the following format: "
                "'Answer: $LETTER' (without quotes) where LETTER is one of options."
            ),
        )

    def test_rejects_unknown_prompt_mode(self):
        with self.assertRaisesRegex(ValueError, "Invalid MMMU-Pro vision prompt_mode"):
            get_mmmu_pro_vision_prompt("unknown")

    def test_resolves_official_mode_default_generation_budget(self):
        for prompt_mode, expected in (
            ("official_direct", 50),
            ("official_cot", 50),
        ):
            with self.subTest(prompt_mode=prompt_mode):
                self.assertEqual(
                    resolve_mmmu_max_new_tokens(
                        None,
                        dataset="MMMU/MMMU_Pro",
                        subject="vision",
                        prompt_mode=prompt_mode,
                    ),
                    expected,
                )

    def test_preserves_explicit_generation_budget(self):
        self.assertEqual(
            resolve_mmmu_max_new_tokens(
                32,
                dataset="MMMU/MMMU_Pro",
                subject="vision",
                prompt_mode="official_direct",
            ),
            32,
        )

    def test_uses_legacy_default_for_non_vision_subject(self):
        self.assertEqual(
            resolve_mmmu_max_new_tokens(
                None,
                dataset="MMMU/MMMU",
                subject="Accounting",
            ),
            16,
        )

    def test_rejects_non_positive_generation_budget(self):
        with self.assertRaisesRegex(ValueError, "must be a positive integer"):
            resolve_mmmu_max_new_tokens(
                0,
                dataset="MMMU/MMMU_Pro",
                subject="vision",
            )

    @patch("tico.quantization.evaluation.mmmu_eval_utils.load_data")
    @patch("tico.quantization.evaluation.mmmu_eval_utils.generate_image_only_answer")
    def test_evaluate_subject_uses_official_prompt_and_budget(
        self,
        mock_generate_image_only_answer,
        mock_load_data,
    ):
        image = object()
        mock_load_data.return_value = [
            {
                "id": "sample-1",
                "image": image,
                "options": ["one", "two", "three", "four"],
                "answer": "C",
            }
        ]
        mock_generate_image_only_answer.return_value = "Answer: C"

        result = evaluate_subject(
            model=Mock(),
            processor=Mock(),
            dataset="MMMU/MMMU_Pro",
            eval_split="test",
            few_shot_split="test",
            subject="vision",
            device="cpu",
            max_new_tokens=None,
            n_shots=0,
            n_samples=1,
            prompt_mode="official_direct",
            verbose=False,
        )

        self.assertEqual(result, (1, 1, 0))
        call_kwargs = mock_generate_image_only_answer.call_args.kwargs
        self.assertIs(call_kwargs["image"], image)
        self.assertEqual(
            call_kwargs["question"],
            MMMU_PRO_VISION_PROMPTS["official_direct"],
        )
        self.assertEqual(call_kwargs["max_new_tokens"], 50)


if __name__ == "__main__":
    unittest.main()
