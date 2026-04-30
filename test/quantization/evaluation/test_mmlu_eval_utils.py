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

"""
Unit tests for MMLU evaluation utilities.

Tests cover:
- Subject name normalization
- Answer extraction strategies
- Prompt construction
- Result aggregation
- Subject parsing
"""

import unittest
from typing import Any
from unittest.mock import patch

from tico.quantization.evaluation.mmlu_eval_utils import (
    aggregate_results,
    ALL_SUBJECTS,
    build_mmlu_prompt,
    extract_answer_letter,
    format_mmlu_question,
    get_item_mmlu,
    MMLU_SUBJECTS,
    normalize_subject_name,
    parse_mmlu_subjects,
    print_mmlu_results,
)


class TestNormalizeSubjectName(unittest.TestCase):
    """Test subject name normalization."""

    def test_normalize_lowercase(self):
        """Test that uppercase is converted to lowercase."""
        result = normalize_subject_name("College_Physics")
        self.assertEqual(result, "college_physics")

    def test_normalize_spaces(self):
        """Test that spaces are converted to underscores."""
        result = normalize_subject_name("college physics")
        self.assertEqual(result, "college_physics")

    def test_normalize_hyphens(self):
        """Test that hyphens are converted to underscores."""
        result = normalize_subject_name("college-physics")
        self.assertEqual(result, "college_physics")

    def test_normalize_mixed(self):
        """Test normalization with mixed case and separators."""
        result = normalize_subject_name("College Physics-101")
        self.assertEqual(result, "college_physics_101")

    def test_normalize_already_normalized(self):
        """Test that already normalized names are unchanged."""
        result = normalize_subject_name("college_physics")
        self.assertEqual(result, "college_physics")


class TestGetItemMmlu(unittest.TestCase):
    """Test MMLU item conversion."""

    def test_get_item_basic(self):
        """Test basic item conversion."""
        raw_item = {
            "subject": "college_physics",
            "question": "What is the speed of light?",
            "choices": ["3e8 m/s", "3e5 m/s", "3e6 m/s", "3e9 m/s"],
            "answer": "A",
        }
        result = get_item_mmlu(raw_item)

        self.assertEqual(result["subject"], "college_physics")
        self.assertEqual(result["question"], "What is the speed of light?")
        self.assertEqual(len(result["choices"]), 4)
        self.assertEqual(result["answer"], "A")

    def test_get_item_truncates_choices(self):
        """Test that choices are truncated to 4 items."""
        raw_item = {
            "subject": "test",
            "question": "Question?",
            "choices": ["A", "B", "C", "D", "E", "F"],  # More than 4
            "answer": "A",
        }
        result = get_item_mmlu(raw_item)

        self.assertEqual(len(result["choices"]), 4)
        self.assertEqual(result["choices"], ["A", "B", "C", "D"])

    def test_get_item_missing_fields(self):
        """Test handling of missing fields."""
        raw_item: dict[str, Any] = {}
        result = get_item_mmlu(raw_item)

        self.assertEqual(result["subject"], "")
        self.assertEqual(result["question"], "")
        self.assertEqual(result["choices"], [])
        self.assertEqual(result["answer"], "")

    def test_get_item_integer_answer(self):
        """Test that integer answers are converted to letters."""
        # Some datasets use integer (0,1,2,3) instead of letters
        raw_item = {
            "question": "Test?",
            "choices": ["A", "B", "C", "D"],
            "answer": 0,  # Integer 0 should become "A"
        }
        result = get_item_mmlu(raw_item)
        self.assertEqual(result["answer"], "A")

        raw_item["answer"] = 1  # Integer 1 should become "B"
        result = get_item_mmlu(raw_item)
        self.assertEqual(result["answer"], "B")

        raw_item["answer"] = 2  # Integer 2 should become "C"
        result = get_item_mmlu(raw_item)
        self.assertEqual(result["answer"], "C")

        raw_item["answer"] = 3  # Integer 3 should become "D"
        result = get_item_mmlu(raw_item)
        self.assertEqual(result["answer"], "D")


class TestFormatMmluQuestion(unittest.TestCase):
    """Test MMLU question formatting."""

    def test_format_with_answer(self):
        """Test formatting a question with answer."""
        question = "What is 2+2?"
        choices = ["3", "4", "5", "6"]
        answer = "B"

        result = format_mmlu_question(question, choices, answer)

        self.assertIn("What is 2+2?", result)
        self.assertIn("A. 3", result)
        self.assertIn("B. 4", result)
        self.assertIn("C. 5", result)
        self.assertIn("D. 6", result)
        self.assertIn("Answer: B", result)

    def test_format_without_answer(self):
        """Test formatting a question without answer (target question)."""
        question = "What is 3+3?"
        choices = ["5", "6", "7", "8"]

        result = format_mmlu_question(question, choices, answer=None)

        self.assertIn("What is 3+3?", result)
        self.assertIn("A. 5", result)
        self.assertIn("B. 6", result)
        self.assertNotIn("Answer:", result)

    def test_format_uses_letters(self):
        """Test that choices are labeled A, B, C, D."""
        question = "Test?"
        choices = ["first", "second", "third", "fourth"]

        result = format_mmlu_question(question, choices, answer=None)

        self.assertIn("A. first", result)
        self.assertIn("B. second", result)
        self.assertIn("C. third", result)
        self.assertIn("D. fourth", result)


class TestBuildMmluPrompt(unittest.TestCase):
    """Test MMLU prompt construction."""

    def test_build_prompt_includes_subject(self):
        """Test that prompt includes subject name."""
        result = build_mmlu_prompt(
            question="What is gravity?",
            choices=["A force", "A particle", "A wave", "A field"],
            subject="college_physics",
            few_shot_examples=[],
        )

        self.assertIn("College Physics", result)  # Subject is formatted for display

    def test_build_prompt_includes_few_shot(self):
        """Test that prompt includes few-shot examples."""
        few_shot_examples = [
            {
                "question": "What is 1+1?",
                "choices": ["1", "2", "3", "4"],
                "answer": "B",
            },
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B",
            },
        ]

        result = build_mmlu_prompt(
            question="What is 3+3?",
            choices=["5", "6", "7", "8"],
            subject="math",
            few_shot_examples=few_shot_examples,
        )

        # Check that few-shot examples are included with answers
        self.assertIn("What is 1+1?", result)
        self.assertIn("Answer: B", result)
        self.assertIn("What is 2+2?", result)

    def test_build_prompt_ends_with_answer_prompt(self):
        """Test that prompt ends with 'Answer:' for model completion."""
        result = build_mmlu_prompt(
            question="What is the test?",
            choices=["A", "B", "C", "D"],
            subject="test",
            few_shot_examples=[],
        )

        self.assertTrue(result.endswith("Answer:"))

    def test_build_prompt_target_has_no_answer(self):
        """Test that target question does not include answer."""
        result = build_mmlu_prompt(
            question="Target question?",
            choices=["A", "B", "C", "D"],
            subject="test",
            few_shot_examples=[
                {
                    "question": "Example?",
                    "choices": ["1", "2", "3", "4"],
                    "answer": "A",
                }
            ],
        )

        # Count occurrences of "Answer:"
        # Should appear once for the few-shot example, and once at the end
        # But target question should NOT have an answer
        lines = result.split("\n")
        answer_lines = [l for l in lines if l.startswith("Answer:")]
        self.assertEqual(len(answer_lines), 2)  # One for few-shot, one at end


class TestExtractAnswerLetter(unittest.TestCase):
    """Test answer extraction from model output."""

    def test_extract_single_letter(self):
        """Test extraction of single letter."""
        result = extract_answer_letter("A")
        self.assertEqual(result, "A")

    def test_extract_letter_with_period(self):
        """Test extraction of letter followed by period."""
        result = extract_answer_letter("B.")
        self.assertEqual(result, "B")

    def test_extract_letter_after_answer_prefix(self):
        """Test extraction after 'Answer:' prefix."""
        result = extract_answer_letter("Answer: C")
        self.assertEqual(result, "C")

    def test_extract_letter_with_whitespace(self):
        """Test extraction with surrounding whitespace."""
        result = extract_answer_letter("  D  ")
        self.assertEqual(result, "D")

    def test_extract_first_letter_in_text(self):
        """Test extraction of first valid letter in text."""
        # Note: Function finds first A/B/C/D character, so need text without those
        # "This picks B for sure" - "picks" has no ABCD, "for" has no ABCD
        result = extract_answer_letter("Result: B")
        self.assertEqual(result, "B")

    def test_extract_lowercase_letter(self):
        """Test extraction of lowercase letter (should be uppercased)."""
        result = extract_answer_letter("c")
        self.assertEqual(result, "C")

    def test_extract_no_valid_letter(self):
        """Test extraction when no valid letter is found."""
        # Text without A, B, C, or D letters
        result = extract_answer_letter("This result is unknown.")
        self.assertIsNone(result)

    def test_extract_ignores_non_abcd(self):
        """Test that letters other than A/B/C/D are ignored."""
        # Text with only E, F, G, H letters (no A, B, C, D)
        result = extract_answer_letter("E is one option")
        # Note: "option" doesn't contain A/B/C/D, "one" doesn't either
        # "E" is not in ABCD, but we need to check if function ignores it
        # The function finds first occurrence of A/B/C/D, so E should be ignored
        self.assertIsNone(result)

    def test_extract_prioritizes_answer_pattern(self):
        """Test that 'Answer: X' pattern is prioritized."""
        result = extract_answer_letter("A. First option\nAnswer: B")
        self.assertEqual(result, "B")


class TestAggregateResults(unittest.TestCase):
    """Test result aggregation."""

    def test_aggregate_single_subject(self):
        """Test aggregation for a single subject."""
        results = {"college_physics": (8, 10)}

        aggregated = aggregate_results(results)

        self.assertEqual(aggregated["subjects"]["college_physics"], 0.8)
        self.assertEqual(aggregated["total_correct"], 8)
        self.assertEqual(aggregated["total_count"], 10)
        self.assertEqual(aggregated["overall"], 0.8)

    def test_aggregate_multiple_subjects(self):
        """Test aggregation across multiple subjects."""
        results = {
            "college_physics": (8, 10),
            "abstract_algebra": (6, 10),
        }

        aggregated = aggregate_results(results)

        self.assertEqual(aggregated["total_correct"], 14)
        self.assertEqual(aggregated["total_count"], 20)
        self.assertEqual(aggregated["overall"], 0.7)

    def test_aggregate_by_domain(self):
        """Test aggregation by domain."""
        # Create results for STEM subjects
        results = {
            "college_physics": (8, 10),  # STEM
            "abstract_algebra": (6, 10),  # STEM
        }

        aggregated = aggregate_results(results)

        self.assertIn("STEM", aggregated["domains"])
        # STEM domain should have 14/20 = 0.7
        self.assertEqual(aggregated["domains"]["STEM"], 0.7)

    def test_aggregate_empty_results(self):
        """Test aggregation with empty results."""
        results: dict[str, tuple[int, int]] = {}

        aggregated = aggregate_results(results)

        self.assertEqual(aggregated["total_correct"], 0)
        self.assertEqual(aggregated["total_count"], 0)
        self.assertEqual(aggregated["overall"], 0.0)

    def test_aggregate_zero_total(self):
        """Test aggregation handles zero total gracefully."""
        results = {"subject": (0, 0)}

        aggregated = aggregate_results(results)

        self.assertEqual(aggregated["subjects"]["subject"], 0.0)


class TestParseMmluSubjects(unittest.TestCase):
    """Test subject parsing."""

    def test_parse_none_returns_none(self):
        """Test that None returns None (all subjects)."""
        result = parse_mmlu_subjects(None)
        self.assertIsNone(result)

    def test_parse_all_returns_none(self):
        """Test that 'all' returns None (all subjects)."""
        result = parse_mmlu_subjects("all")
        self.assertIsNone(result)

    def test_parse_stem_domain(self):
        """Test parsing STEM domain."""
        result = parse_mmlu_subjects("stem")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        # Check that STEM subjects are included
        assert result is not None  # Type guard for Pylance
        self.assertIn("college_physics", result)
        self.assertIn("abstract_algebra", result)

    def test_parse_humanities_domain(self):
        """Test parsing Humanities domain."""
        result = parse_mmlu_subjects("humanities")

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        self.assertIn("philosophy", result)
        self.assertIn("world_religions", result)

    def test_parse_comma_separated(self):
        """Test parsing comma-separated subject names."""
        result = parse_mmlu_subjects("college_physics,abstract_algebra")

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        self.assertIn("college_physics", result)
        self.assertIn("abstract_algebra", result)

    def test_parse_removes_duplicates(self):
        """Test that duplicate subjects are removed."""
        result = parse_mmlu_subjects("college_physics,college_physics,college_physics")

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        self.assertEqual(result.count("college_physics"), 1)

    def test_parse_partial_match(self):
        """Test partial name matching."""
        result = parse_mmlu_subjects("physics")

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        # Should match college_physics, high_school_physics, conceptual_physics
        self.assertIn("college_physics", result)
        self.assertIn("high_school_physics", result)

    def test_parse_invalid_subject_raises(self):
        """Test that invalid subject raises ValueError."""
        with self.assertRaises(ValueError) as context:
            parse_mmlu_subjects("invalid_subject_xyz")

        self.assertIn("Unknown subject", str(context.exception))


class TestPrintMmluResults(unittest.TestCase):
    """Test result printing."""

    @patch("builtins.print")
    def test_print_basic(self, mock_print):
        """Test basic result printing."""
        results = {
            "subjects": {"college_physics": 0.8, "abstract_algebra": 0.6},
            "domains": {"STEM": 0.7},
            "overall": 0.7,
            "total_correct": 14,
            "total_count": 20,
        }

        print_mmlu_results(results)

        # Check that print was called multiple times
        self.assertTrue(mock_print.call_count > 0)

        # Check that overall accuracy is printed
        printed_text = "\n".join(
            str(call.args[0]) for call in mock_print.call_args_list
        )
        self.assertIn("Overall Accuracy", printed_text)


class TestMmluSubjectsData(unittest.TestCase):
    """Test MMLU subject data integrity."""

    def test_all_subjects_not_empty(self):
        """Test that ALL_SUBJECTS is populated."""
        self.assertTrue(len(ALL_SUBJECTS) > 0)

    def test_mmlu_subjects_has_domains(self):
        """Test that MMLU_SUBJECTS has expected domains."""
        expected_domains = {"STEM", "Humanities", "Social Sciences", "Other"}
        self.assertEqual(set(MMLU_SUBJECTS.keys()), expected_domains)

    def test_all_subjects_flattened_correctly(self):
        """Test that ALL_SUBJECTS contains all subjects from domains."""
        all_from_domains = []
        for subjects in MMLU_SUBJECTS.values():
            all_from_domains.extend(subjects)

        self.assertEqual(set(ALL_SUBJECTS), set(all_from_domains))

    def test_no_duplicate_subjects(self):
        """Test that there are no duplicate subjects across domains."""
        all_subjects = []
        for subjects in MMLU_SUBJECTS.values():
            all_subjects.extend(subjects)

        self.assertEqual(len(all_subjects), len(set(all_subjects)))


if __name__ == "__main__":
    unittest.main()
