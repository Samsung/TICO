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
from unittest.mock import Mock, patch

from tico.quantization.evaluation import mmmu_eval_utils as mmmu
from tico.quantization.recipes.evaluation import mmmu as recipe


class TestMmmuProtocol(unittest.TestCase):
    def test_direct_default(self):
        self.assertEqual(mmmu.resolve_mmmu_max_new_tokens(
            None, dataset="MMMU/MMMU_Pro", subject="vision"), 256)

    def test_cot_requires_explicit_budget(self):
        with self.assertRaisesRegex(ValueError, "explicit max_new_tokens"):
            mmmu.resolve_mmmu_max_new_tokens(
                None, dataset="MMMU/MMMU_Pro", subject="vision", prompt_mode="official_cot")
        self.assertEqual(mmmu.resolve_mmmu_max_new_tokens(
            512, dataset="MMMU/MMMU_Pro", subject="vision", prompt_mode="official_cot"), 512)

    def test_non_vision_default_stays_16(self):
        self.assertEqual(mmmu.resolve_mmmu_max_new_tokens(
            None, dataset="MMMU/MMMU", subject="Accounting"), 16)

    def test_explicit_budget_is_preserved_and_validated(self):
        self.assertEqual(mmmu.resolve_mmmu_max_new_tokens(
            32, dataset="MMMU/MMMU_Pro", subject="vision"), 32)
        for value in (0, -1, True, 2.5, "256"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mmmu.resolve_mmmu_max_new_tokens(value, dataset="MMMU/MMMU_Pro", subject="vision")

    def test_official_prompt_and_invalid_mode(self):
        direct = mmmu.get_mmmu_pro_vision_prompt("official_direct")
        self.assertEqual(direct,
            "Answer with the option letter from the given choices directly. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options.")
        self.assertIn("Think step by step", mmmu.get_mmmu_pro_vision_prompt("official_cot"))
        for value in ("invalid", "", None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mmmu.get_mmmu_pro_vision_prompt(value)

    def test_answer_formats(self):
        cases = {
            "C": "C", "c": "C", "C.": "C", "(D)": "D",
            "Answer: C": "C", "**Answer:** C": "C", "**Answer: C**": "C",
            "The answer is C.": "C", "I think the answer is C.": "C",
            "Option (J)": "J", "I would choose B.": "B",
            "A. This is the first option.": "A", "Reasoning.\nC": "C",
            "```\nC\n```": "C", "Answer: c": "C",
        }
        for text, answer in cases.items():
            with self.subTest(text=text):
                self.assertEqual(mmmu.extract_answer(text), answer)

    def test_explicit_final_answer_beats_option_mentions(self):
        for text in (
            "Answer: C\nOption A is incorrect.",
            "Option A might fit, but the answer is C.",
            "Answer: A. After checking, final answer: C.",
            "A. First guess.\n**Answer:** C",
        ):
            with self.subTest(text=text):
                self.assertEqual(mmmu.extract_answer(text), "C")

    def test_does_not_guess_from_prose(self):
        for text in (
            "", "I think this is ambiguous.", "To answer a question like this, inspect it.",
            "The answer is a circle.", "Option A is incorrect.",
            "The image contains a diagram.", "Answer: C or D", "Answer: C/D",
        ):
            with self.subTest(text=text):
                self.assertIsNone(mmmu.extract_answer(text))

    def test_choice_range_and_invalid_final_answer(self):
        self.assertIsNone(mmmu.extract_answer("Answer: A.\nAnswer: J", num_choices=4))
        for count in (0, 11, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                mmmu.extract_answer("C", num_choices=count)

    def test_lazy_optional_dataset_dependency(self):
        with patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaisesRegex(RuntimeError, "optional 'datasets'"):
                mmmu.load_data("MMMU/MMMU_Pro", "vision", "test", n_samples=1)

    def test_zero_samples_and_offset_are_respected(self):
        fake = ModuleType("datasets")
        fake.load_dataset = lambda **kwargs: iter([{"id": 0}, {"id": 1}, {"id": 2}])
        with patch.dict(sys.modules, {"datasets": fake}):
            self.assertEqual(list(mmmu.load_data("MMMU/MMMU_Pro", "vision", "test", n_samples=0)), [])
            self.assertEqual(list(mmmu.load_data("MMMU/MMMU_Pro", "vision", "test", start=1)), [{"id": 1}, {"id": 2}])


class TestMmmuEvaluationLogging(unittest.TestCase):
    def kwargs(self, **overrides):
        kwargs = dict(model=object(), processor=object(), dataset="MMMU/MMMU_Pro",
                      subjects=["vision"], device="cpu", n_shots=0, n_samples=2,
                      max_seq_len=2048, verbose=False)
        kwargs.update(overrides)
        return kwargs

    def sample(self, sample_id="one", **overrides):
        sample = {"id": sample_id, "image": object(), "options": ["a", "b", "c", "d"], "answer": "C"}
        sample.update(overrides)
        return sample

    def test_default_prompt_budget_and_legacy_result_shape(self):
        with patch.object(mmmu, "load_data", return_value=[self.sample()]), patch.object(
            mmmu, "generate_image_only_answer", return_value="Answer: C"
        ) as generate:
            result = mmmu.evaluate_mmmu(**self.kwargs())
        self.assertEqual(result, {"vision": (1, 1, 0)})
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 256)
        self.assertEqual(generate.call_args.kwargs["question"], mmmu.MMMU_PRO_VISION_PROMPTS["official_direct"])
        self.assertNotIn("diagnostics", generate.call_args.kwargs)

    def test_jsonl_contains_raw_outputs_parse_failures_and_completion(self):
        samples = [self.sample(), self.sample("two")]
        def generate(**kwargs):
            kwargs["diagnostics"].update(input_tokens=12, generated_tokens=2,
                                         eos_observed=True, length_limited=False, stop_reason="eos")
            return "Answer: C" if kwargs["image"] is samples[0]["image"] else "I am unsure."
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "fp.jsonl"
            with patch.object(mmmu, "load_data", return_value=samples), patch.object(
                mmmu, "generate_image_only_answer", side_effect=generate):
                result = mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path, input_max_seq_len=1536))
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(result, {"vision": (1, 2, 0)})
        self.assertEqual(records[0]["parser_version"], mmmu.MMMU_ANSWER_PARSER_VERSION)
        self.assertEqual(records[-1]["record_type"], "run_summary")
        examples = [r for r in records if r["record_type"] == "sample"]
        self.assertEqual([r["id"] for r in examples], ["one", "two"])
        self.assertTrue(examples[1]["parse_failed"])
        self.assertEqual(examples[1]["generated"], "I am unsure.")
        summary = next(r for r in records if r["record_type"] == "subject_summary")
        self.assertEqual(summary["parse_failure_rate"], 0.5)
        self.assertEqual(summary["eos_count"], 2)

    def test_logs_are_not_overwritten_or_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.jsonl"
            path.write_text("preserve me")
            with patch.object(mmmu, "load_data") as load, self.assertRaises(FileExistsError):
                mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
            load.assert_not_called()
            self.assertEqual(path.read_text(), "preserve me")

    def test_invalid_budgets_fail_before_dataset_and_log_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data") as load, self.assertRaisesRegex(ValueError, "must not exceed"):
                mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path, input_max_seq_len=1800, max_new_tokens=512))
            load.assert_not_called()
            self.assertFalse(path.exists())

    def test_all_skips_are_recorded_and_zero_total_can_be_printed(self):
        samples = [self.sample(image_2=object()), self.sample("two")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data", return_value=samples), patch.object(
                mmmu, "generate_image_only_answer", side_effect=RuntimeError("fake runtime failure")):
                result = mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(result, {"vision": (0, 0, 2)})
        examples = [r for r in records if r["record_type"] == "sample"]
        self.assertEqual([r["skip_reason"] for r in examples], ["multiple_images", "RuntimeError"])
        with contextlib.redirect_stdout(io.StringIO()):
            mmmu.print_mmmu_results(result)

    def test_unexpected_value_error_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            with patch.object(mmmu, "load_data", return_value=[self.sample()]), patch.object(
                mmmu, "generate_image_only_answer", side_effect=ValueError("unexpected shape")):
                with self.assertRaisesRegex(ValueError, "unexpected shape"):
                    mmmu.evaluate_mmmu(**self.kwargs(output_jsonl=path))
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(records[-1]["record_type"], "run_error")
        self.assertFalse(any(r["record_type"] == "run_summary" for r in records))

    def test_invalid_subjects_are_prevalidated(self):
        with patch.object(mmmu, "load_data") as load, self.assertRaises(ValueError):
            mmmu.evaluate_mmmu(**self.kwargs(subjects=["vision", "bad-subject"]))
        load.assert_not_called()

    def test_recipe_forwards_optional_settings(self):
        with patch.object(recipe, "evaluate_mmmu", return_value={}) as evaluate, patch.object(recipe, "print_mmmu_results"):
            recipe.evaluate_and_print_mmmu(model=object(), processor=object(), dataset="MMMU/MMMU_Pro",
                subjects=["vision"], device="cpu", n_shots=0, n_samples=2, max_new_tokens=None,
                max_seq_len=2048, temperature=0.0, verbose=False, input_max_seq_len=1536, output_jsonl="fp.jsonl")
        self.assertEqual(evaluate.call_args.kwargs["prompt_mode"], "official_direct")
        self.assertIsNone(evaluate.call_args.kwargs["max_new_tokens"])
        self.assertEqual(evaluate.call_args.kwargs["input_max_seq_len"], 1536)
        self.assertEqual(evaluate.call_args.kwargs["output_jsonl"], "fp.jsonl")


class TestMmmuAdapterWiring(unittest.TestCase):
    def test_both_adapter_calls_forward_config_without_forcing_16_tokens(self):
        # Exercise the actual call expressions without constructing large model
        # adapters or importing their optional model-family dependencies.
        root = Path(__file__).resolve().parents[3]
        for family in ("qwen3_vl", "gemma4"):
            path = root / "tico/quantization/recipes/adapters" / f"{family}.py"
            tree = ast.parse(path.read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "evaluate_and_print_mmmu"]
            self.assertEqual(len(calls), 1)
            for config in ({}, {"max_new_tokens": 512, "input_max_seq_len": 1536, "prompt_mode": "official_direct", "output_jsonl": "fp.jsonl"}):
                capture = Mock()
                namespace = dict(evaluate_and_print_mmmu=capture, ctx=SimpleNamespace(model=object(), processor=object(), device="cpu"),
                                 mmmu=config, subjects=["vision"], max_seq_len=2048, verbose=False)
                eval(compile(ast.Expression(calls[0]), str(path), "eval"), namespace)
                self.assertEqual(capture.call_args.kwargs["max_new_tokens"], config.get("max_new_tokens"))
                self.assertEqual(capture.call_args.kwargs["input_max_seq_len"], config.get("input_max_seq_len"))
                self.assertEqual(capture.call_args.kwargs["output_jsonl"], config.get("output_jsonl"))


class TestMmmuExampleConfigs(unittest.TestCase):
    def test_all_mmmu_pro_vision_presets_use_the_new_defaults(self):
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
                config = yaml.safe_load(path.read_text())["evaluation"]["mmmu"]
                self.assertEqual(config["prompt_mode"], "official_direct")
                self.assertEqual(config["max_new_tokens"], 256)
                self.assertEqual(config["n_shots"], 0)
                self.assertIsNone(config["input_max_seq_len"])
                self.assertIsNone(config["output_jsonl"])


if __name__ == "__main__":
    unittest.main()
