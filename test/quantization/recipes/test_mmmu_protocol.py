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

import ast
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from tico.quantization.evaluation import mmmu_eval_utils as mmmu
from tico.quantization.recipes.evaluation import mmmu as recipe

# The exact prompt used by TICO before prompt modes existed (ten options).
LEGACY_STRICT_PROMPT = (
    "You are a strict single-letter answering machine. Answer the multiple-choice "
    "question shown in the image. Return only one letter from A to J."
)
VISION = dict(dataset="MMMU/MMMU_Pro", subject="vision")


class TestMmmuBudgetDefaults(unittest.TestCase):
    def test_strict_single_letter_is_the_default_with_16_tokens(self):
        self.assertEqual(
            mmmu.DEFAULT_MMMU_PRO_VISION_PROMPT_MODE, "strict_single_letter"
        )
        self.assertEqual(mmmu.resolve_mmmu_max_new_tokens(None, **VISION), 16)
        self.assertEqual(
            mmmu.resolve_mmmu_max_new_tokens(
                None, prompt_mode="strict_single_letter", **VISION
            ),
            16,
        )

    def test_direct_default_is_256_and_cot_requires_explicit_budget(self):
        self.assertEqual(
            mmmu.resolve_mmmu_max_new_tokens(
                None, prompt_mode="official_direct", **VISION
            ),
            256,
        )
        with self.assertRaisesRegex(ValueError, "explicit max_new_tokens"):
            mmmu.resolve_mmmu_max_new_tokens(None, prompt_mode="official_cot", **VISION)
        self.assertEqual(
            mmmu.resolve_mmmu_max_new_tokens(512, prompt_mode="official_cot", **VISION),
            512,
        )

    def test_mode_defaults_are_defined_once(self):
        self.assertEqual(
            mmmu.MMMU_PRO_VISION_DEFAULT_MAX_NEW_TOKENS,
            {"strict_single_letter": 16, "official_direct": 256, "official_cot": None},
        )
        self.assertEqual(
            set(mmmu.MMMU_PRO_VISION_DEFAULT_MAX_NEW_TOKENS),
            set(mmmu.MMMU_PRO_VISION_PROMPT_MODES),
        )

    def test_non_vision_default_stays_16_for_every_mode(self):
        for mode in mmmu.MMMU_PRO_VISION_PROMPT_MODES:
            with self.subTest(mode=mode):
                self.assertEqual(
                    mmmu.resolve_mmmu_max_new_tokens(
                        None,
                        dataset="MMMU/MMMU",
                        subject="Accounting",
                        prompt_mode=mode,
                    ),
                    16,
                )

    def test_explicit_budget_wins_in_every_mode(self):
        for mode, value in (
            ("strict_single_letter", 32),
            ("official_direct", 16),
            ("official_cot", 1024),
        ):
            with self.subTest(mode=mode):
                self.assertEqual(
                    mmmu.resolve_mmmu_max_new_tokens(value, prompt_mode=mode, **VISION),
                    value,
                )

    def test_invalid_explicit_budgets_are_rejected_not_converted(self):
        for mode in mmmu.MMMU_PRO_VISION_PROMPT_MODES:
            for value in (0, -1, True, False, 2.5, "256", 16.0):
                with self.subTest(mode=mode, value=value), self.assertRaises(
                    ValueError
                ):
                    mmmu.resolve_mmmu_max_new_tokens(
                        value, prompt_mode=mode, **VISION  # type: ignore[arg-type]
                    )

    def test_invalid_prompt_mode_is_rejected_even_with_explicit_budget(self):
        invalid_modes: tuple[object, ...] = ("strict", "direct", "", None, [])
        for mode in invalid_modes:
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "prompt_mode"
            ):
                mmmu.resolve_mmmu_max_new_tokens(
                    16, prompt_mode=mode, **VISION  # type: ignore[arg-type]
                )


class TestMmmuPrompts(unittest.TestCase):
    def test_ten_choice_strict_prompt_matches_the_legacy_text_exactly(self):
        self.assertEqual(
            mmmu.get_mmmu_pro_vision_prompt("strict_single_letter", 10),
            LEGACY_STRICT_PROMPT,
        )

    def test_strict_prompt_last_label_follows_the_choice_count(self):
        expected = {4: "A to D", 9: "A to I", 10: "A to J", 12: "A to L", 26: "A to Z"}
        for count, span in expected.items():
            with self.subTest(count=count):
                prompt = mmmu.get_mmmu_pro_vision_prompt("strict_single_letter", count)
                self.assertTrue(prompt.endswith(f"Return only one letter from {span}."))
                self.assertTrue(
                    prompt.startswith(
                        "You are a strict single-letter answering machine. "
                        "Answer the multiple-choice question shown in the image. "
                    )
                )
        single = mmmu.get_mmmu_pro_vision_prompt("strict_single_letter", 1)
        self.assertTrue(single.endswith("Return only the letter A."))
        self.assertNotIn("from A to", single)

    def test_strict_prompt_requires_the_actual_choice_count(self):
        with self.assertRaisesRegex(ValueError, "num_choices"):
            mmmu.get_mmmu_pro_vision_prompt("strict_single_letter")
        for count in (0, 27, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                mmmu.get_mmmu_pro_vision_prompt(
                    "strict_single_letter", count  # type: ignore[arg-type]
                )

    def test_official_prompts_are_unchanged_and_ignore_the_choice_count(self):
        direct = (
            "Answer with the option letter from the given choices directly. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options."
        )
        cot = (
            "Write out the multiple-choice question in the image and then solve it. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        )
        self.assertEqual(mmmu.get_mmmu_pro_vision_prompt("official_direct"), direct)
        self.assertEqual(mmmu.get_mmmu_pro_vision_prompt("official_direct", 12), direct)
        self.assertEqual(mmmu.get_mmmu_pro_vision_prompt("official_cot", 4), cot)
        self.assertEqual(mmmu.MMMU_PRO_VISION_PROMPTS["official_cot"], cot)

    def test_invalid_mode(self):
        invalid_modes: tuple[Any, ...] = (
            "invalid",
            "",
            None,
            [],
            "STRICT_SINGLE_LETTER",
        )
        for value in invalid_modes:
            with self.subTest(value=value), self.assertRaises(ValueError):
                mmmu.get_mmmu_pro_vision_prompt(value, 10)


class TestChoiceLabels(unittest.TestCase):
    def test_labels_follow_the_choice_count(self):
        self.assertEqual(mmmu.mmmu_choice_labels(1), "A")
        self.assertEqual(mmmu.mmmu_choice_labels(4), "ABCD")
        self.assertEqual(mmmu.mmmu_choice_labels(9), "ABCDEFGHI")
        self.assertEqual(mmmu.mmmu_choice_labels(10), "ABCDEFGHIJ")
        self.assertEqual(mmmu.mmmu_choice_labels(12), "ABCDEFGHIJKL")
        self.assertEqual(mmmu.mmmu_choice_labels(26), "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_unsupported_counts_raise_instead_of_clamping(self):
        for count in (0, -1, 27, 100, True, False, 2.5, "10", None):
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError, "num_choices"
            ):
                mmmu.mmmu_choice_labels(count)  # type: ignore[arg-type]


class TestMmmuParser(unittest.TestCase):
    def test_parser_version(self):
        self.assertEqual(mmmu.MMMU_ANSWER_PARSER_VERSION, "tico-mmmu-v4")
        self.assertEqual(
            mmmu.MMMU_LEGACY_OPEN_PARSER_VERSION, "tico-mmmu-legacy-open-v1"
        )

    def test_answer_formats(self):
        cases = {
            "C": "C",
            "c": "C",
            "C.": "C",
            "(D)": "D",
            "Answer: C": "C",
            "**Answer:** C": "C",
            "**Answer: C**": "C",
            "The answer is C.": "C",
            "I think the answer is C.": "C",
            "Option (J)": "J",
            "I would choose B.": "B",
            "A. This is the first option.": "A",
            "Reasoning.\nC": "C",
            "```\nC\n```": "C",
            "Answer: c": "C",
            "Answer: (C)": "C",
            "Answer: C)": "C",
        }
        for text, answer in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mmmu.extract_answer(text), answer)

    def test_parenthesised_label_followed_by_text(self):
        cases = {
            "Answer: (H)HENRI DE TOULOUSE-LAUTREC": ("H", "explicit"),
            "(B)Yes.": ("B", "leading"),
            "(F)Treponema pallidum": ("F", "leading"),
            "Answer: (C)No": ("C", "explicit"),
            "Answer: (B)Q,P": ("B", "explicit"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mmmu._parse_answer(text, 10), expected)

    def test_bare_letter_needs_a_boundary(self):
        for text in (
            "Answer: HENRI",
            "Answer: Yes",
            "The answer is No.",
            "Answer: XYZ",
        ):
            with self.subTest(text=text):
                self.assertEqual(mmmu._parse_answer(text, 26), (None, "unparsed"))

    def test_extended_label_range(self):
        self.assertEqual(mmmu._parse_answer("Answer: L", 12), ("L", "explicit"))
        self.assertEqual(mmmu._parse_answer("(K)Some text", 12), ("K", "leading"))
        self.assertEqual(mmmu._parse_answer("Answer: Z", 26), ("Z", "explicit"))
        self.assertEqual(mmmu._parse_answer("z", 26), ("Z", "bare"))
        for count, inside, outside in (
            (4, "D", "E"),
            (9, "I", "J"),
            (10, "J", "K"),
            (12, "L", "M"),
        ):
            with self.subTest(count=count):
                self.assertEqual(
                    mmmu.extract_answer(f"Answer: {inside}", count), inside
                )
                self.assertEqual(
                    mmmu._parse_answer(f"Answer: {outside}", count),
                    (None, "out_of_range"),
                )
                self.assertEqual(
                    mmmu._parse_answer(outside, count), (None, "out_of_range")
                )

    def test_omitted_num_choices_keeps_the_ten_label_default(self):
        self.assertEqual(mmmu.extract_answer("Answer: J"), "J")
        self.assertIsNone(mmmu.extract_answer("Answer: K"))
        self.assertEqual(mmmu.extract_answer("Answer: K", 12), "K")
        self.assertEqual(mmmu._parse_answer("Answer: K"), (None, "out_of_range"))

    def test_explicit_final_answer_beats_option_mentions(self):
        for text in (
            "Answer: C\nOption A is incorrect.",
            "Option A might fit, but the answer is C.",
            "Answer: A. After checking, final answer: C.",
            "A. First guess.\n**Answer:** C",
            "(A) looked plausible at first.\nAnswer: C",
        ):
            with self.subTest(text=text):
                self.assertEqual(mmmu.extract_answer(text), "C")

    def test_does_not_guess_from_prose(self):
        for text in (
            "",
            "I think this is ambiguous.",
            "To answer a question, inspect the image.",
            "To answer a question like this, inspect it.",
            "The answer is a circle.",
            "The answer is an apple.",
            "Option A is incorrect.",
            "The image contains a diagram.",
            "Because (A) is wrong, I cannot decide.",
            "The options (A) and (B) both fail.",
            "No.",
            "Yes",
            "e.g. the diagram",
            "Not enough information.",
            "Let me think.\nIt is unclear.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(mmmu.extract_answer(text, 26))

    def test_ambiguous_answers_return_none(self):
        cases = {
            "Answer: C or D": "ambiguous_declaration",
            "Answer: C/D": "ambiguous_declaration",
            "Answer: (C)/(D)": "ambiguous_declaration",
            "Answer: C and D": "ambiguous_declaration",
            "(C) or (D)": "ambiguous_leading",
            "(C)/(D)": "ambiguous_leading",
        }
        for text, method in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mmmu._parse_answer(text, 10), (None, method))

    def test_last_declaration_does_not_fall_back_to_an_earlier_one(self):
        self.assertEqual(
            mmmu._parse_answer("Answer: A\nAnswer: Z", 12), (None, "out_of_range")
        )
        self.assertEqual(
            mmmu._parse_answer("Answer: A\nFinal answer: C or D", 12),
            (None, "ambiguous_declaration"),
        )
        self.assertEqual(
            mmmu._parse_answer("Answer: A.\nAnswer: J", 4), (None, "out_of_range")
        )
        self.assertEqual(mmmu._parse_answer("J", 9), (None, "out_of_range"))

    def test_last_hyphenated_declaration_is_ambiguous_not_skipped(self):
        # v3 did not accept ``-`` as a bare-label boundary, so ``Answer: C-D``
        # was never a candidate and the earlier ``Answer: B`` was returned.
        ambiguous = (
            "Answer: B\nAnswer: C-D",
            "Answer: B\nAnswer: (C)-(D)",
            "Answer: B\nAnswer: C/D",
            "Answer: B\nAnswer: C or D",
            "Answer: B\nAnswer: C - D",
            "Answer: B\nAnswer: C-",
        )
        for text in ambiguous:
            with self.subTest(text=text):
                self.assertEqual(
                    mmmu._parse_answer(text, 12), (None, "ambiguous_declaration")
                )
                self.assertIsNone(mmmu.extract_answer(text, 12))
        self.assertEqual(
            mmmu._parse_answer("Answer: B\nAnswer: Z", 12), (None, "out_of_range")
        )
        self.assertEqual(
            mmmu._parse_answer("Answer: B\nAnswer: C.", 12), ("C", "explicit")
        )
        # A hyphenated word after the declaration is not a label either.
        self.assertIsNone(mmmu.extract_answer("Answer: X-ray", 26))
        # Leading and bare contexts stay consistent with the declaration rule.
        self.assertEqual(mmmu._parse_answer("(C)-(D)", 10), (None, "ambiguous_leading"))
        self.assertEqual(mmmu._parse_answer("C-D", 10), (None, "unparsed"))

    def test_v3_regression_cases_are_unchanged(self):
        cases = {
            "Answer: (H)HENRI DE TOULOUSE-LAUTREC": ("H", "explicit"),
            "(B)Yes.": ("B", "leading"),
            "**Answer:** C": ("C", "explicit"),
            "Answer: HENRI": (None, "unparsed"),
            "To answer a question, inspect the image.": (None, "unparsed"),
            "Option A is incorrect.": (None, "unparsed"),
            "Answer: C\nOption A is incorrect.": ("C", "explicit"),
            "Answer: (C)/(D)": (None, "ambiguous_declaration"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mmmu._parse_answer(text, 26), expected)


class TestLegacyOpenAnswer(unittest.TestCase):
    def test_letter_extraction_and_raw_text_fallback(self):
        cases = {
            "42": ("42", "legacy_text"),
            " 42 \n": ("42", "legacy_text"),
            "Answer: 42": ("Answer: 42", "legacy_text"),
            "x = 3": ("x = 3", "legacy_text"),
            "k": ("k", "legacy_text"),
            "a": ("A", "legacy_letter"),
            "(b)": ("B", "legacy_letter"),
            "A nucleus": ("A", "legacy_letter"),
            "B. 42": ("B", "legacy_letter"),
            "Option B": ("B", "legacy_letter"),
            "The answer is (j)": ("J", "legacy_letter"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mmmu.legacy_open_answer(text), expected)

    def test_invalid_choice_counts_raise(self):
        for count in (0, 27, True, -3, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError, "num_choices"
            ):
                mmmu.extract_answer("C", num_choices=count)  # type: ignore[arg-type]


class TestMmmuDatasetHelpers(unittest.TestCase):
    def test_lazy_optional_dataset_dependency(self):
        with patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaisesRegex(RuntimeError, "optional 'datasets'"):
                mmmu.load_data("MMMU/MMMU_Pro", "vision", "test", n_samples=1)

    def test_zero_samples_and_offset_are_respected(self):
        fake = ModuleType("datasets")
        fake.load_dataset = lambda **kwargs: iter([{"id": 0}, {"id": 1}, {"id": 2}])  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"datasets": fake}):
            self.assertEqual(
                list(mmmu.load_data("MMMU/MMMU_Pro", "vision", "test", n_samples=0)), []
            )
            self.assertEqual(
                list(mmmu.load_data("MMMU/MMMU_Pro", "vision", "test", start=1)),
                [{"id": 1}, {"id": 2}],
            )


class TestSampleValidation(unittest.TestCase):
    def test_valid_structures_return_labels(self):
        self.assertEqual(mmmu.validate_sample_choices("s", ["a"], "A"), "A")
        self.assertEqual(
            mmmu.validate_sample_choices("s", ["a"] * 12, "l"), "ABCDEFGHIJKL"
        )
        self.assertEqual(mmmu.validate_sample_choices("s", ["a"] * 4, " c "), "ABCD")
        self.assertEqual(len(mmmu.validate_sample_choices("s", ["a"] * 26, "Z")), 26)

    def test_unsupported_structures_name_the_sample(self):
        cases = {
            "empty": ([], "A", "unsupported number of choices"),
            "too_many": (["a"] * 27, "A", "unsupported number of choices"),
            "not_a_list": ("A. one B. two", "A", "options must be a list"),
            "gold_outside": (["a", "b"], "C", "gold answer"),
            "gold_not_a_letter": (["a", "b"], "42", "gold answer"),
        }
        for name, (choices, gold, fragment) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(
                ValueError, f"(?s)Sample 'test_X_1'.*{fragment}"
            ):
                mmmu.validate_sample_choices("test_X_1", choices, gold)

    def test_gold_must_be_exactly_one_valid_label(self):
        # ``in "ABCD"`` is a substring test; these all passed it before.
        invalid: tuple[Any, ...] = ("", " ", "AB", "A,B", "ab", None, 1, True, 0.5)
        for gold in invalid:
            with self.subTest(gold=gold), self.assertRaisesRegex(
                ValueError, "Sample 'test_X_1'.*gold answer"
            ):
                mmmu.validate_sample_choices("test_X_1", ["a"] * 4, gold)
            with self.subTest(gold=gold, helper="normalize"), self.assertRaisesRegex(
                ValueError, "Sample 'test_X_1'.*gold answer"
            ):
                mmmu.normalize_mmmu_gold_label("test_X_1", gold, "ABCD")

    def test_gold_is_normalized_consistently(self):
        for gold in ("C", "c", " C ", "\tc\n"):
            with self.subTest(gold=gold):
                self.assertEqual(mmmu.normalize_mmmu_gold_label("s", gold, "ABCD"), "C")
        with self.assertRaisesRegex(ValueError, "gold answer 'E'"):
            mmmu.normalize_mmmu_gold_label("s", "E", "ABCD")
        self.assertEqual(mmmu.normalize_mmmu_gold_label("s", "l", "ABCDEFGHIJKL"), "L")


class TestScoringPolicy(unittest.TestCase):
    def test_missing_or_multiple_choice_type_selects_multiple_choice(self):
        for question_type in (None, "multiple-choice"):
            for vision_only in (True, False):
                with self.subTest(question_type=question_type, vision=vision_only):
                    self.assertEqual(
                        mmmu.resolve_mmmu_scoring_policy(
                            "s", question_type, ["a", "b"], vision_only=vision_only
                        ),
                        "multiple_choice",
                    )
        # Empty options are not reinterpreted as open-ended without the type.
        self.assertEqual(
            mmmu.resolve_mmmu_scoring_policy("s", None, [], vision_only=False),
            "multiple_choice",
        )
        self.assertEqual(
            mmmu.resolve_mmmu_scoring_policy(
                "s", "multiple-choice", [], vision_only=False
            ),
            "multiple_choice",
        )

    def test_open_requires_a_non_vision_subject_and_no_options(self):
        self.assertEqual(
            mmmu.resolve_mmmu_scoring_policy("s", "open", [], vision_only=False),
            "legacy_open",
        )
        cases = {
            "vision": ("open", [], True, "vision"),
            "with_options": ("open", ["a", "b"], False, "contradictory"),
            "non_list": ("open", "[]", False, "contradictory"),
            "unknown": ("fill-in", [], False, "unsupported question_type"),
            "empty_type": ("", [], False, "unsupported question_type"),
            "bool_type": (True, [], False, "unsupported question_type"),
        }
        for name, (question_type, choices, vision_only, fragment) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(
                ValueError, f"(?s)Sample 'test_X_1'.*{fragment}"
            ):
                mmmu.resolve_mmmu_scoring_policy(
                    "test_X_1", question_type, choices, vision_only=vision_only
                )

    def test_get_item_forwards_question_type(self):
        row = {"id": "x", "image": None, "options": "[]", "answer": "42"}
        self.assertIsNone(mmmu.get_item_mmmu(row)["question_type"])
        self.assertEqual(mmmu.get_item_mmmu(row)["choices"], [])
        row["question_type"] = "open"
        self.assertEqual(mmmu.get_item_mmmu(row)["question_type"], "open")


class TestMmmuEvaluationLogging(unittest.TestCase):
    def kwargs(self, **overrides):
        kwargs = dict(
            model=object(),
            processor=object(),
            dataset="MMMU/MMMU_Pro",
            subjects=["vision"],
            device="cpu",
            n_shots=0,
            n_samples=2,
            max_seq_len=2048,
            verbose=False,
        )
        kwargs.update(overrides)
        return kwargs

    def sample(self, sample_id="one", num_choices=4, answer="C", **overrides):
        sample = {
            "id": sample_id,
            "image": object(),
            "options": [f"option {i}" for i in range(num_choices)],
            "answer": answer,
        }
        sample.update(overrides)
        return sample

    def read(self, path):
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    def test_default_prompt_budget_and_legacy_result_shape(self):
        with patch.object(
            mmmu, "load_data", return_value=[self.sample()]
        ), patch.object(
            mmmu, "generate_image_only_answer", return_value="C"
        ) as generate:
            result = mmmu.evaluate_mmmu(**self.kwargs())
        self.assertEqual(result, {"vision": (1, 1, 0)})
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 16)
        self.assertEqual(
            generate.call_args.kwargs["question"],
            mmmu.get_mmmu_pro_vision_prompt("strict_single_letter", 4),
        )
        self.assertTrue(generate.call_args.kwargs["question"].endswith("from A to D."))
        self.assertNotIn("diagnostics", generate.call_args.kwargs)
        self.assertNotIn("input_max_seq_len", generate.call_args.kwargs)

    def test_ten_choice_default_prompt_is_the_legacy_prompt(self):
        with patch.object(
            mmmu, "load_data", return_value=[self.sample(num_choices=10, answer="J")]
        ), patch.object(
            mmmu, "generate_image_only_answer", return_value="J"
        ) as generate:
            result = mmmu.evaluate_mmmu(**self.kwargs())
        self.assertEqual(result, {"vision": (1, 1, 0)})
        self.assertEqual(generate.call_args.kwargs["question"], LEGACY_STRICT_PROMPT)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 16)

    def test_official_direct_is_an_explicit_option_with_256_default(self):
        with patch.object(
            mmmu, "load_data", return_value=[self.sample(num_choices=12, answer="L")]
        ), patch.object(
            mmmu, "generate_image_only_answer", return_value="Answer: L"
        ) as generate:
            result = mmmu.evaluate_mmmu(**self.kwargs(prompt_mode="official_direct"))
        self.assertEqual(result, {"vision": (1, 1, 0)})
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 256)
        self.assertEqual(
            generate.call_args.kwargs["question"],
            mmmu.MMMU_PRO_VISION_PROMPTS["official_direct"],
        )

    def test_explicit_budget_overrides_mode_default(self):
        for mode, budget in (("strict_single_letter", 64), ("official_direct", 16)):
            with self.subTest(mode=mode), patch.object(
                mmmu, "load_data", return_value=[self.sample()]
            ), patch.object(
                mmmu, "generate_image_only_answer", return_value="C"
            ) as generate:
                mmmu.evaluate_mmmu(
                    **self.kwargs(prompt_mode=mode, max_new_tokens=budget)
                )
            self.assertEqual(generate.call_args.kwargs["max_new_tokens"], budget)

    def test_cot_without_budget_fails_before_dataset_loading(self):
        with patch.object(mmmu, "load_data") as load, self.assertRaisesRegex(
            ValueError, "explicit max_new_tokens"
        ):
            mmmu.evaluate_mmmu(**self.kwargs(prompt_mode="official_cot"))
        load.assert_not_called()

    def test_twelve_choice_sample_is_evaluated_normally(self):
        samples = [
            self.sample("ten", num_choices=10, answer="B"),
            self.sample("twelve", num_choices=12, answer="L"),
            self.sample("four", num_choices=4, answer="D"),
        ]
        prompts: dict[str, str] = {}

        def generate(**kwargs):
            image = kwargs["image"]
            sample = next(s for s in samples if s["image"] is image)
            prompts[sample["id"]] = kwargs["question"]
            return {"ten": "B", "twelve": "L", "four": "I"}[sample["id"]]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data", return_value=samples), patch.object(
                mmmu, "generate_image_only_answer", side_effect=generate
            ):
                result = mmmu.evaluate_mmmu(
                    **self.kwargs(n_samples=3, output_jsonl=path)
                )
            records = self.read(path)
        # 'four' answered I, which is outside A-D: wrong, not skipped.
        self.assertEqual(result, {"vision": (2, 3, 0)})
        self.assertEqual(prompts["ten"], LEGACY_STRICT_PROMPT)
        self.assertTrue(prompts["twelve"].endswith("from A to L."))
        self.assertTrue(prompts["four"].endswith("from A to D."))
        self.assertEqual(records[-1]["record_type"], "run_summary")
        self.assertEqual(records[-1]["results"], {"vision": [2, 3, 0]})
        by_id = {r["id"]: r for r in records if r["record_type"] == "sample"}
        self.assertEqual(by_id["twelve"]["num_choices"], 12)
        self.assertEqual(by_id["twelve"]["valid_labels"], "ABCDEFGHIJKL")
        self.assertEqual(by_id["twelve"]["predicted"], "L")
        self.assertTrue(by_id["twelve"]["correct"])
        self.assertEqual(by_id["twelve"]["prompt_mode"], "strict_single_letter")
        self.assertEqual(by_id["twelve"]["prompt"], prompts["twelve"])
        self.assertEqual(by_id["four"]["parse_method"], "out_of_range")
        self.assertTrue(by_id["four"]["parse_failed"])
        self.assertIsNone(by_id["four"]["predicted"])
        self.assertFalse(by_id["four"]["correct"])
        self.assertEqual(by_id["twelve"]["scoring_policy"], "multiple_choice")
        self.assertEqual(by_id["twelve"]["parser_version"], "tico-mmmu-v4")
        self.assertIsNone(by_id["twelve"]["question_type"])
        start = next(r for r in records if r["record_type"] == "subject_start")
        self.assertEqual(start["max_new_tokens"], 16)
        self.assertEqual(start["prompt_mode"], "strict_single_letter")
        self.assertEqual(start["parser_version"], "tico-mmmu-v4")
        self.assertEqual(records[0]["parser_version"], "tico-mmmu-v4")
        self.assertEqual(records[0]["open_parser_version"], "tico-mmmu-legacy-open-v1")
        summary = next(r for r in records if r["record_type"] == "subject_summary")
        self.assertEqual(
            (summary["correct"], summary["total"], summary["skipped"]), (2, 3, 0)
        )
        self.assertEqual(summary["parse_failures"], 1)

    def test_unsupported_choice_structures_fail_before_generation(self):
        cases = {
            "zero": self.sample("zero", num_choices=0),
            "too_many": self.sample("too_many", num_choices=27, answer="A"),
            # A string literal that is not a list after ast.literal_eval.
            "not_a_list": self.sample("not_a_list", options="'A. x B. y'"),
            "gold_out_of_range": self.sample(
                "gold_out_of_range", num_choices=4, answer="E"
            ),
        }
        for name, sample in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run.jsonl"
                with patch.object(
                    mmmu, "load_data", return_value=[self.sample("ok"), sample]
                ), patch.object(
                    mmmu, "generate_image_only_answer", return_value="C"
                ) as generate:
                    with self.assertRaisesRegex(ValueError, f"Sample '{name}'"):
                        mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
                # Only the valid sample before it was generated; nothing was skipped.
                self.assertEqual(generate.call_count, 1)
                records = self.read(path)
                self.assertEqual(records[-1]["record_type"], "run_error")
                self.assertIn(f"Sample '{name}'", records[-1]["error"])
                self.assertFalse(
                    any(r["record_type"] == "run_summary" for r in records)
                )
                self.assertFalse(any(r.get("status") == "skipped" for r in records))

    def test_multiple_choice_gold_is_normalized_once_for_scoring_and_logging(self):
        samples = [
            self.sample("lower", answer="c"),
            self.sample("padded", answer=" C "),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data", return_value=samples), patch.object(
                mmmu, "generate_image_only_answer", return_value="C"
            ):
                result = mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
            records = self.read(path)
        self.assertEqual(result, {"vision": (2, 2, 0)})
        by_id = {r["id"]: r for r in records if r["record_type"] == "sample"}
        self.assertEqual(by_id["lower"]["gold"], "C")
        self.assertEqual(by_id["lower"]["gold_raw"], "c")
        self.assertEqual(by_id["padded"]["gold"], "C")
        self.assertEqual(by_id["padded"]["gold_raw"], " C ")
        self.assertTrue(by_id["padded"]["correct"])

    def test_invalid_multiple_choice_golds_fail_before_generation(self):
        invalid: tuple[Any, ...] = ("", " ", "AB", "A,B", None, 1, True)
        for gold in invalid:
            with self.subTest(gold=gold), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run.jsonl"
                samples = [self.sample("ok"), self.sample("bad_gold", answer=gold)]
                with patch.object(
                    mmmu, "load_data", return_value=samples
                ), patch.object(
                    mmmu, "generate_image_only_answer", return_value="C"
                ) as generate:
                    with self.assertRaisesRegex(
                        ValueError, "Sample 'bad_gold'.*gold answer"
                    ):
                        mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
                self.assertEqual(generate.call_count, 1)
                records = self.read(path)
                self.assertEqual(records[-1]["record_type"], "run_error")
                self.assertFalse(any(r.get("status") == "skipped" for r in records))

    def test_jsonl_contains_raw_outputs_parse_failures_and_completion(self):
        samples = [self.sample(), self.sample("two")]

        def generate(**kwargs):
            kwargs["diagnostics"].update(
                input_tokens=12,
                generated_tokens=2,
                eos_observed=True,
                length_limited=False,
                stop_reason="eos",
            )
            return (
                "Answer: C"
                if kwargs["image"] is samples[0]["image"]
                else "I am unsure."
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "fp.jsonl"
            with patch.object(mmmu, "load_data", return_value=samples), patch.object(
                mmmu, "generate_image_only_answer", side_effect=generate
            ):
                result = mmmu.evaluate_mmmu(
                    **self.kwargs(output_jsonl=path, input_max_seq_len=1536)
                )
            records = self.read(path)
        # The parse failure stays in the denominator as a wrong answer.
        self.assertEqual(result, {"vision": (1, 2, 0)})
        self.assertEqual(records[0]["parser_version"], mmmu.MMMU_ANSWER_PARSER_VERSION)
        self.assertEqual(records[0]["prompt_mode"], "strict_single_letter")
        self.assertEqual(records[-1]["record_type"], "run_summary")
        examples = [r for r in records if r["record_type"] == "sample"]
        self.assertEqual([r["id"] for r in examples], ["one", "two"])
        self.assertTrue(examples[1]["parse_failed"])
        self.assertEqual(examples[1]["parse_method"], "unparsed")
        self.assertEqual(examples[1]["generated"], "I am unsure.")
        self.assertEqual(examples[1]["valid_labels"], "ABCD")
        summary = next(r for r in records if r["record_type"] == "subject_summary")
        self.assertEqual(summary["parse_failure_rate"], 0.5)
        self.assertEqual(summary["eos_count"], 2)
        self.assertEqual(summary["total"], 2)

    def test_logs_are_not_overwritten_or_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.jsonl"
            path.write_text("preserve me")
            with patch.object(mmmu, "load_data") as load, self.assertRaises(
                FileExistsError
            ):
                mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
            load.assert_not_called()
            self.assertEqual(path.read_text(), "preserve me")

    def test_invalid_budgets_fail_before_dataset_and_log_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data") as load, self.assertRaisesRegex(
                ValueError, "must not exceed"
            ):
                mmmu.evaluate_mmmu(
                    **self.kwargs(
                        output_jsonl=path, input_max_seq_len=1800, max_new_tokens=512
                    )
                )
            load.assert_not_called()
            self.assertFalse(path.exists())

    def test_default_budget_participates_in_total_length_validation(self):
        # strict default 16 fits, official_direct default 256 does not.
        with patch.object(mmmu, "load_data", return_value=[]):
            self.assertEqual(
                mmmu.evaluate_mmmu(**self.kwargs(input_max_seq_len=2032)),
                {"vision": (0, 0, 0)},
            )
        with patch.object(mmmu, "load_data") as load, self.assertRaisesRegex(
            ValueError, "must not exceed"
        ):
            mmmu.evaluate_mmmu(
                **self.kwargs(input_max_seq_len=2032, prompt_mode="official_direct")
            )
        load.assert_not_called()

    def test_all_skips_are_recorded_and_zero_total_can_be_printed(self):
        samples = [self.sample(image_2=object()), self.sample("two")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data", return_value=samples), patch.object(
                mmmu,
                "generate_image_only_answer",
                side_effect=RuntimeError("fake runtime failure"),
            ):
                result = mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
            records = self.read(path)
        self.assertEqual(result, {"vision": (0, 0, 2)})
        examples = [r for r in records if r["record_type"] == "sample"]
        self.assertEqual(
            [r["skip_reason"] for r in examples], ["multiple_images", "RuntimeError"]
        )
        self.assertEqual(records[-1]["record_type"], "run_summary")
        with contextlib.redirect_stdout(io.StringIO()):
            mmmu.print_mmmu_results(result)

    def test_unexpected_value_error_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(
                mmmu, "load_data", return_value=[self.sample()]
            ), patch.object(
                mmmu,
                "generate_image_only_answer",
                side_effect=ValueError("unexpected shape"),
            ):
                with self.assertRaisesRegex(ValueError, "unexpected shape"):
                    mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
            records = self.read(path)
        self.assertEqual(records[-1]["record_type"], "run_error")
        self.assertFalse(any(r["record_type"] == "run_summary" for r in records))

    def test_invalid_subjects_are_prevalidated(self):
        with patch.object(mmmu, "load_data") as load, self.assertRaises(ValueError):
            mmmu.evaluate_mmmu(**self.kwargs(subjects=["vision", "bad-subject"]))
        load.assert_not_called()

    def test_non_vision_keeps_existing_prompt_path_and_budget(self):
        sample = self.sample(question="Which is correct?")
        with patch.object(mmmu, "load_data", return_value=[sample]), patch.object(
            mmmu, "generate_answer", return_value="C"
        ) as generate, patch.object(mmmu, "generate_image_only_answer") as image_only:
            result = mmmu.evaluate_mmmu(
                **self.kwargs(dataset="MMMU/MMMU", subjects=["Accounting"])
            )
        self.assertEqual(result, {"Accounting": (1, 1, 0)})
        image_only.assert_not_called()
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 16)
        question = generate.call_args.kwargs["question"]
        self.assertIn("Which is correct?", question)
        self.assertIn("A. option 0", question)
        self.assertNotIn("strict single-letter", question)

    # --- non-Pro MMMU open-ended compatibility -------------------------------

    def open_sample(self, sample_id="open", answer="42", question="How many?"):
        # Shaped like an MMMU/MMMU row: ``options`` is the string "[]" and the
        # image lives in ``image_1``.
        return {
            "id": sample_id,
            "image_1": object(),
            "image_2": None,
            "question": question,
            "options": "[]",
            "answer": answer,
            "question_type": "open",
        }

    def mc_sample(self, sample_id="mc", answer="C", question="Which?"):
        return self.sample(
            sample_id,
            answer=answer,
            question=question,
            question_type="multiple-choice",
        )

    def non_vision(self, **overrides):
        return self.kwargs(dataset="MMMU/MMMU", subjects=["Accounting"], **overrides)

    def test_open_question_is_evaluated_with_legacy_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(
                mmmu, "load_data", return_value=[self.open_sample()]
            ), patch.object(
                mmmu, "generate_answer", return_value="42"
            ) as generate, patch.object(
                mmmu, "generate_image_only_answer"
            ) as image_only:
                result = mmmu.evaluate_mmmu(**self.non_vision(output_jsonl=path))
            records = self.read(path)
        self.assertEqual(result, {"Accounting": (1, 1, 0)})
        image_only.assert_not_called()
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 16)
        prompt = generate.call_args.kwargs["question"]
        self.assertIn("How many?", prompt)
        self.assertTrue(prompt.endswith("\nAnswer:"))
        self.assertNotIn("42", prompt)
        record = next(r for r in records if r["record_type"] == "sample")
        self.assertEqual(record["status"], "evaluated")
        self.assertEqual(record["question_type"], "open")
        self.assertEqual(record["scoring_policy"], "legacy_open")
        self.assertEqual(record["parser_version"], "tico-mmmu-legacy-open-v1")
        self.assertIsNone(record["num_choices"])
        self.assertIsNone(record["valid_labels"])
        self.assertEqual(record["gold"], "42")
        self.assertEqual(record["predicted"], "42")
        self.assertEqual(record["parse_method"], "legacy_text")
        self.assertFalse(record["parse_failed"])
        self.assertTrue(record["correct"])
        start = next(r for r in records if r["record_type"] == "subject_start")
        self.assertEqual(start["parser_version"], "tico-mmmu-v4")
        self.assertEqual(start["open_parser_version"], "tico-mmmu-legacy-open-v1")

    def test_wrong_open_answer_is_counted_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(
                mmmu, "load_data", return_value=[self.open_sample()]
            ), patch.object(mmmu, "generate_answer", return_value="41"):
                result = mmmu.evaluate_mmmu(**self.non_vision(output_jsonl=path))
            records = self.read(path)
        self.assertEqual(result, {"Accounting": (0, 1, 0)})
        record = next(r for r in records if r["record_type"] == "sample")
        self.assertEqual(record["status"], "evaluated")
        self.assertEqual(record["predicted"], "41")
        self.assertFalse(record["correct"])
        summary = next(r for r in records if r["record_type"] == "subject_summary")
        self.assertEqual(summary["parse_failures"], 0)
        self.assertEqual(records[-1]["record_type"], "run_summary")

    def test_open_scoring_reproduces_the_pre_pr_behaviour(self):
        # Expected values are derived by hand from the pre-PR evaluator: the
        # legacy ``extract_answer`` (leading ``A``-``J`` label or
        # ``answer/option/choice X`` mention -> letter, otherwise the stripped
        # text) compared with ``answer.upper()``. Several entries are legacy
        # quirks (``x`` vs ``X``, ``42.`` vs ``42``) that are preserved, not
        # endorsed.
        cases = (
            # (generated, gold, predicted, correct)
            ("42", "42", "42", True),
            (" 42 \n", "42", "42", True),
            ("41", "42", "41", False),
            ("Answer: 42", "42", "Answer: 42", False),
            ("42.", "42", "42.", False),
            ("a", "a", "A", True),
            ("(b)", "b", "B", True),
            ("Option B", "B", "B", True),
            ("A nucleus", "A nucleus", "A", False),
            ("x", "x", "x", False),
            ("x = 3", "x = 3", "x = 3", False),
            ("['24/7', '3.429']", "['24/7', '3.429']", "['24/7', '3.429']", True),
        )
        samples = [
            self.open_sample(f"open_{i}", answer=gold)
            for i, (_, gold, _, _) in enumerate(cases)
        ]
        by_image = {s["image_1"]: cases[i][0] for i, s in enumerate(samples)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data", return_value=samples), patch.object(
                mmmu,
                "generate_answer",
                side_effect=lambda **kwargs: by_image[kwargs["image"]],
            ):
                result = mmmu.evaluate_mmmu(
                    **self.non_vision(n_samples=len(cases), output_jsonl=path)
                )
            records = [r for r in self.read(path) if r["record_type"] == "sample"]
        expected_correct = sum(int(c) for _, _, _, c in cases)
        self.assertEqual(result, {"Accounting": (expected_correct, len(cases), 0)})
        for record, (generated, gold, predicted, correct) in zip(records, cases):
            with self.subTest(generated=generated, gold=gold):
                self.assertEqual(record["gold"], gold.upper())
                self.assertEqual(record["predicted"], predicted)
                self.assertEqual(record["correct"], correct)
                self.assertEqual(record["scoring_policy"], "legacy_open")

    def test_open_gold_keeps_the_legacy_upper_only_comparison(self):
        # Multiple-choice gold normalisation (strip, single label) does not
        # apply: the legacy value is ``answer.upper()`` and nothing else.
        with patch.object(
            mmmu, "load_data", return_value=[self.open_sample(answer=" 42 ")]
        ), patch.object(mmmu, "generate_answer", return_value="42"):
            self.assertEqual(
                mmmu.evaluate_mmmu(**self.non_vision()), {"Accounting": (0, 1, 0)}
            )
        with patch.object(
            mmmu, "load_data", return_value=[self.open_sample(answer=42)]
        ), patch.object(mmmu, "generate_answer") as generate:
            with self.assertRaisesRegex(ValueError, "Sample 'open'.*must be a string"):
                mmmu.evaluate_mmmu(**self.non_vision())
        generate.assert_not_called()

    def test_mixed_question_types_share_one_denominator(self):
        samples = [
            self.mc_sample("mc_right", answer="C"),
            self.open_sample("open_wrong", answer="42"),
            self.mc_sample("mc_unparsed", answer="B"),
            self.open_sample("open_right", answer="7"),
        ]
        outputs = {
            "mc_right": "C",
            "open_wrong": "41",
            "mc_unparsed": "I am unsure.",
            "open_right": "7",
        }
        by_image = {s.get("image_1", s.get("image")): outputs[s["id"]] for s in samples}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data", return_value=samples), patch.object(
                mmmu,
                "generate_answer",
                side_effect=lambda **kwargs: by_image[kwargs["image"]],
            ):
                result = mmmu.evaluate_mmmu(
                    **self.non_vision(n_samples=4, output_jsonl=path)
                )
            records = self.read(path)
        self.assertEqual(result, {"Accounting": (2, 4, 0)})
        by_id = {r["id"]: r for r in records if r["record_type"] == "sample"}
        self.assertEqual(
            {k: v["scoring_policy"] for k, v in by_id.items()},
            {
                "mc_right": "multiple_choice",
                "open_wrong": "legacy_open",
                "mc_unparsed": "multiple_choice",
                "open_right": "legacy_open",
            },
        )
        self.assertEqual(by_id["mc_right"]["valid_labels"], "ABCD")
        self.assertEqual(by_id["mc_right"]["parser_version"], "tico-mmmu-v4")
        self.assertEqual(by_id["mc_unparsed"]["parse_method"], "unparsed")
        self.assertTrue(by_id["mc_unparsed"]["parse_failed"])
        self.assertFalse(by_id["open_wrong"]["parse_failed"])
        summary = next(r for r in records if r["record_type"] == "subject_summary")
        self.assertEqual((summary["correct"], summary["total"]), (2, 4))
        # Only the multiple-choice parser has a failure notion.
        self.assertEqual(summary["parse_failures"], 1)
        self.assertEqual(summary["parse_failure_rate"], 0.25)
        self.assertEqual(records[-1]["results"], {"Accounting": [2, 4, 0]})

    def test_few_shot_examples_may_be_open_without_leaking_the_target_gold(self):
        shots = [
            self.open_sample("shot_open", answer="17", question="Shot one?"),
            self.mc_sample("shot_mc", answer="B", question="Shot two?"),
        ]
        target = self.open_sample("target", answer="7391", question="Target?")

        def load_data(**kwargs):
            return (
                list(shots[: kwargs["n_samples"]])
                if kwargs["split"] == "test"
                else [target]
            )

        for n_shots in (0, 2):
            with self.subTest(n_shots=n_shots), patch.object(
                mmmu, "load_data", side_effect=load_data
            ), patch.object(mmmu, "generate_answer", return_value="7391") as generate:
                result = mmmu.evaluate_mmmu(**self.non_vision(n_shots=n_shots))
            self.assertEqual(result, {"Accounting": (1, 1, 0)})
            prompt = generate.call_args.kwargs["question"]
            self.assertIn("Target?", prompt)
            self.assertTrue(prompt.endswith("Target?\nAnswer:"))
            self.assertNotIn("7391", prompt)
            if n_shots:
                self.assertIn("Shot one?\nAnswer: 17", prompt)
                self.assertIn("Shot two?\nA. option 0", prompt)
                self.assertIn("Answer: B", prompt)
            else:
                self.assertNotIn("Shot", prompt)
                self.assertNotIn("17", prompt)

    def test_explicit_multiple_choice_with_empty_options_is_not_treated_as_open(self):
        sample = self.mc_sample("mc_empty", answer="A")
        sample["options"] = "[]"
        with patch.object(mmmu, "load_data", return_value=[sample]), patch.object(
            mmmu, "generate_answer"
        ) as generate:
            with self.assertRaisesRegex(
                ValueError, "Sample 'mc_empty'.*unsupported number of choices"
            ):
                mmmu.evaluate_mmmu(**self.non_vision())
        generate.assert_not_called()
        # Without a question_type, an empty option list is still an error.
        untyped = self.sample("untyped", num_choices=0, answer="A")
        with patch.object(mmmu, "load_data", return_value=[untyped]), patch.object(
            mmmu, "generate_answer"
        ) as generate:
            with self.assertRaisesRegex(ValueError, "Sample 'untyped'"):
                mmmu.evaluate_mmmu(**self.non_vision())
        generate.assert_not_called()

    def test_contradictory_metadata_fails_before_generation_with_the_id(self):
        open_with_options = self.open_sample("open_opts")
        open_with_options["options"] = "['x', 'y']"
        unknown = self.open_sample("unknown_type")
        unknown["question_type"] = "fill-in"
        vision_open = self.sample("vision_open", question_type="open")
        cases = {
            "open_opts": (self.non_vision(), open_with_options, "contradictory"),
            "unknown_type": (self.non_vision(), unknown, "unsupported question_type"),
            "vision_open": (self.kwargs(), vision_open, "vision"),
        }
        for sample_id, (kwargs, sample, fragment) in cases.items():
            with self.subTest(sample=sample_id), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run.jsonl"
                with patch.object(
                    mmmu, "load_data", return_value=[sample]
                ), patch.object(mmmu, "generate_answer") as generate, patch.object(
                    mmmu, "generate_image_only_answer"
                ) as image_only:
                    with self.assertRaisesRegex(
                        ValueError, f"(?s)Sample '{sample_id}'.*{fragment}"
                    ):
                        mmmu.evaluate_mmmu(**dict(kwargs, output_jsonl=path))
                generate.assert_not_called()
                image_only.assert_not_called()
                records = self.read(path)
                self.assertEqual(records[-1]["record_type"], "run_error")
                self.assertFalse(any(r.get("status") == "skipped" for r in records))

    def test_recipe_forwards_optional_settings(self):
        for prompt_mode, expected in (
            (None, "strict_single_letter"),
            ("official_direct", "official_direct"),
        ):
            with self.subTest(prompt_mode=prompt_mode), patch.object(
                recipe, "evaluate_mmmu", return_value={}
            ) as evaluate, patch.object(recipe, "print_mmmu_results"):
                recipe.evaluate_and_print_mmmu(
                    model=object(),
                    processor=object(),
                    dataset="MMMU/MMMU_Pro",
                    subjects=["vision"],
                    device="cpu",
                    n_shots=0,
                    n_samples=2,
                    max_new_tokens=None,
                    max_seq_len=2048,
                    temperature=0.0,
                    verbose=False,
                    prompt_mode=prompt_mode,
                    input_max_seq_len=1536,
                    output_jsonl="fp.jsonl",
                )
            self.assertEqual(evaluate.call_args.kwargs["prompt_mode"], expected)
            self.assertIsNone(evaluate.call_args.kwargs["max_new_tokens"])
            self.assertEqual(evaluate.call_args.kwargs["input_max_seq_len"], 1536)
            self.assertEqual(evaluate.call_args.kwargs["output_jsonl"], "fp.jsonl")


class TestMmmuAdapterWiring(unittest.TestCase):
    def test_both_adapter_calls_forward_config_without_fixed_values(self):
        # Exercise the actual call expressions without constructing large model
        # adapters or importing their optional model-family dependencies.
        root = Path(__file__).resolve().parents[3]
        for family in ("qwen3_vl", "gemma4"):
            path = root / "tico/quantization/recipes/adapters" / f"{family}.py"
            tree = ast.parse(path.read_text())
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "evaluate_and_print_mmmu"
            ]
            self.assertEqual(len(calls), 1)
            configs: tuple[dict[str, Any], ...] = (
                {},
                {
                    "max_new_tokens": 512,
                    "input_max_seq_len": 1536,
                    "prompt_mode": "official_direct",
                    "output_jsonl": "fp.jsonl",
                },
                {"prompt_mode": "strict_single_letter", "max_new_tokens": 16},
            )
            for config in configs:
                capture = Mock()
                namespace = dict(
                    evaluate_and_print_mmmu=capture,
                    ctx=SimpleNamespace(
                        model=object(), processor=object(), device="cpu"
                    ),
                    mmmu=config,
                    subjects=["vision"],
                    max_seq_len=2048,
                    verbose=False,
                )
                eval(compile(ast.Expression(calls[0]), str(path), "eval"), namespace)
                kwargs = capture.call_args.kwargs
                # Mode defaults are resolved by the evaluator, never by adapters.
                self.assertEqual(kwargs["max_new_tokens"], config.get("max_new_tokens"))
                self.assertEqual(kwargs["prompt_mode"], config.get("prompt_mode"))
                self.assertEqual(
                    kwargs["input_max_seq_len"], config.get("input_max_seq_len")
                )
                self.assertEqual(kwargs["output_jsonl"], config.get("output_jsonl"))


class TestMmmuExampleConfigs(unittest.TestCase):
    def test_all_mmmu_pro_vision_presets_use_the_strict_defaults(self):
        import yaml

        root = Path(__file__).resolve().parents[3]
        presets = (
            "qwen3_vl_eval_suite",
            "gemma4_eval_suite",
            "qwen3_vl_eval_suite_mx_override_polices",
        )
        for preset in presets:
            with self.subTest(preset=preset):
                path = root / "tico/quantization/examples/configs" / (preset + ".yaml")
                config = yaml.safe_load(path.read_text())["evaluation"]
                self.assertEqual(config["max_seq_len"], 2048)
                config = config["mmmu"]
                self.assertEqual(config["dataset"], "MMMU/MMMU_Pro")
                self.assertEqual(config["subjects"], ["vision"])
                self.assertEqual(config["prompt_mode"], "strict_single_letter")
                self.assertEqual(config["max_new_tokens"], 16)
                self.assertEqual(config["n_shots"], 0)
                self.assertIsNone(config["input_max_seq_len"])
                self.assertIsNone(config["output_jsonl"])
                self.assertEqual(config["temperature"], 0.0)
                # The preset's explicit values match the evaluator's defaults.
                self.assertEqual(
                    mmmu.resolve_mmmu_max_new_tokens(
                        config["max_new_tokens"],
                        prompt_mode=config["prompt_mode"],
                        **VISION,
                    ),
                    mmmu.resolve_mmmu_max_new_tokens(None, **VISION),
                )


if __name__ == "__main__":
    unittest.main()
