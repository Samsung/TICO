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

import contextlib
import copy
import io
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import tico.quantization.recipes.adapters.gemma4 as gemma4_mod
import tico.quantization.recipes.adapters.qwen3_vl as qwen3_vl_mod
import tico.quantization.recipes.evaluation.mmmu as mmmu_mod

import torch
from tico.quantization.recipes.context import RecipeContext


class TestAdapterMmmuRouting(unittest.TestCase):
    """Exercise real adapter selection, normalization, and MMMU forwarding.

    Only the evaluator boundary that loads data and runs inference is replaced;
    recipe preflight, the MMMU wrapper, and result printing remain real.
    """

    ADAPTERS = (
        (qwen3_vl_mod, qwen3_vl_mod.Qwen3VLAdapter),
        (gemma4_mod, gemma4_mod.Gemma4Adapter),
    )

    @staticmethod
    def _context(adapter: Any, evaluation: dict[str, Any]) -> RecipeContext:
        ctx = RecipeContext(
            cfg={
                "model": {"family": adapter.family},
                "runtime": {"show_progress": False},
                "calibration": {"seq_len": 256},
                "evaluation": evaluation,
            },
            adapter=adapter,
        )
        ctx.model = SimpleNamespace(
            config=SimpleNamespace(
                use_cache=False,
                text_config=SimpleNamespace(use_cache=False),
            )
        )
        ctx.processor = SimpleNamespace(tokenizer=object())
        ctx.device = torch.device("cpu")
        return ctx

    @staticmethod
    def _evaluation(**mmmu: Any) -> dict[str, Any]:
        return {
            "enabled": True,
            "mmmu": {"enabled": True, "dataset": "MMMU/MMMU_Pro", **mmmu},
        }

    def _run(self, adapter: Any, evaluation: dict[str, Any]):
        ctx = self._context(adapter, evaluation)
        with patch.object(
            mmmu_mod, "evaluate_mmmu", return_value={"vision": (1, 2, 0)}
        ) as evaluate, contextlib.redirect_stdout(io.StringIO()) as stdout:
            adapter.evaluate(ctx)
        return evaluate, stdout.getvalue(), ctx

    def test_default_arguments_and_real_result_printing(self):
        for _module, adapter_cls in self.ADAPTERS:
            with self.subTest(adapter=adapter_cls.family):
                evaluation = self._evaluation()
                evaluation["n_samples"] = 99
                evaluate, output, ctx = self._run(adapter_cls(), evaluation)
                evaluate.assert_called_once_with(
                    model=ctx.model,
                    processor=ctx.processor,
                    dataset="MMMU/MMMU_Pro",
                    subjects=None,
                    device="cpu",
                    n_shots=5,
                    n_samples=-1,
                    max_new_tokens=None,
                    max_seq_len=None,
                    temperature=0.0,
                    verbose=False,
                    prompt_mode="strict_single_letter",
                    input_max_seq_len=None,
                    output_jsonl=None,
                )
                self.assertIn("vision", output)
                self.assertIn("0.5000", output)

    def test_subject_sentinels_and_explicit_subjects(self):
        cases: tuple[tuple[Any, Any], ...] = (
            ("mmmu", None),
            (["mmmu"], None),
            (None, None),
            ([], []),
            (["vision", "standard (4 options)"], ["vision", "standard (4 options)"]),
            (["MMMU"], ["MMMU"]),
            (("mmmu",), ("mmmu",)),
        )
        for _module, adapter_cls in self.ADAPTERS:
            for subjects, expected in cases:
                with self.subTest(adapter=adapter_cls.family, subjects=subjects):
                    evaluate, _output, _ctx = self._run(
                        adapter_cls(), self._evaluation(subjects=subjects)
                    )
                    actual = evaluate.call_args.kwargs["subjects"]
                    self.assertEqual(actual, expected)
                    if expected is not None:
                        self.assertIs(actual, subjects)

    def test_explicit_settings_keep_coercion_and_protocol_passthrough(self):
        for _module, adapter_cls in self.ADAPTERS:
            with self.subTest(adapter=adapter_cls.family):
                evaluation = self._evaluation(
                    subjects=["vision"],
                    n_shots="0",
                    n_samples="7",
                    max_new_tokens=512,
                    temperature="0.25",
                    verbose=0,
                    prompt_mode="official_direct",
                    input_max_seq_len=1536,
                    output_jsonl="mmmu.jsonl",
                )
                evaluation.update(max_seq_len=2048, verbose=True)
                evaluate, _output, ctx = self._run(adapter_cls(), evaluation)
                evaluate.assert_called_once_with(
                    model=ctx.model,
                    processor=ctx.processor,
                    dataset="MMMU/MMMU_Pro",
                    subjects=["vision"],
                    device="cpu",
                    n_shots=0,
                    n_samples=7,
                    max_new_tokens=512,
                    max_seq_len=2048,
                    temperature=0.25,
                    verbose=False,
                    prompt_mode="official_direct",
                    input_max_seq_len=1536,
                    output_jsonl="mmmu.jsonl",
                )

    def test_inherits_top_level_verbose_without_changing_config(self):
        for _module, adapter_cls in self.ADAPTERS:
            with self.subTest(adapter=adapter_cls.family):
                evaluation = self._evaluation(subjects=["mmmu"])
                evaluation["verbose"] = True
                before = copy.deepcopy(evaluation)
                evaluate, _output, _ctx = self._run(adapter_cls(), evaluation)
                self.assertIs(evaluate.call_args.kwargs["verbose"], True)
                self.assertEqual(evaluation, before)

    def test_enabled_and_selected_target_policies(self):
        cases: tuple[tuple[dict[str, Any], bool], ...] = (
            ({}, True),
            ({"selected_tasks": None}, True),
            ({"selected_tasks": ["mmmu"]}, True),
            ({"selected_tasks": []}, False),
            ({"enabled": False, "selected_tasks": ["mmmu"]}, False),
        )
        for _module, adapter_cls in self.ADAPTERS:
            for override, expected in cases:
                with self.subTest(adapter=adapter_cls.family, override=override):
                    evaluation = self._evaluation()
                    evaluation.update(override)
                    evaluate, _output, _ctx = self._run(adapter_cls(), evaluation)
                    self.assertEqual(evaluate.call_count, int(expected))
            with self.subTest(adapter=adapter_cls.family, nested_enabled=False):
                evaluation = self._evaluation(enabled=False)
                evaluate, _output, _ctx = self._run(adapter_cls(), evaluation)
                evaluate.assert_not_called()
                evaluation["selected_tasks"] = ["mmmu"]
                evaluate, _output, _ctx = self._run(adapter_cls(), evaluation)
                evaluate.assert_called_once()

    def test_absent_null_and_false_mapping_stay_off_without_selection(self):
        cases: tuple[dict[str, Any], ...] = (
            {"enabled": True},
            {"enabled": True, "mmmu": None},
            {"enabled": True, "mmmu": False},
            {"enabled": True, "mmmu": {}},
            {"enabled": True, "selected_tasks": [], "mmmu": "invalid"},
        )
        for _module, adapter_cls in self.ADAPTERS:
            for evaluation in cases:
                with self.subTest(adapter=adapter_cls.family, evaluation=evaluation):
                    evaluate, _output, _ctx = self._run(adapter_cls(), evaluation)
                    evaluate.assert_not_called()

    def test_selected_missing_dataset_keeps_preflight_failure(self):
        for _module, adapter_cls in self.ADAPTERS:
            for raw in (None, False, {}, {"dataset": ""}):
                with self.subTest(adapter=adapter_cls.family, mmmu=raw):
                    ctx = self._context(
                        adapter_cls(),
                        {"enabled": True, "selected_tasks": ["mmmu"], "mmmu": raw},
                    )
                    with patch.object(mmmu_mod, "evaluate_mmmu") as evaluate:
                        with self.assertRaisesRegex(ValueError, "MMMU/MMMU mixes"):
                            ctx.adapter.evaluate(ctx)
                    evaluate.assert_not_called()
                    self.assertFalse(ctx.model.config.use_cache)

    def test_invalid_mapping_and_argument_coercion_propagate(self):
        for _module, adapter_cls in self.ADAPTERS:
            invalid_values: tuple[object, ...] = (True, "enabled", [])
            for raw in invalid_values:
                with self.subTest(adapter=adapter_cls.family, mmmu=raw):
                    ctx = self._context(adapter_cls(), {"enabled": True, "mmmu": raw})
                    with patch.object(mmmu_mod, "evaluate_mmmu") as evaluate:
                        with self.assertRaisesRegex(
                            TypeError, "mmmu must be a mapping"
                        ):
                            ctx.adapter.evaluate(ctx)
                    evaluate.assert_not_called()
            with self.subTest(adapter=adapter_cls.family, invalid_n_shots=True):
                ctx = self._context(adapter_cls(), self._evaluation(n_shots="invalid"))
                with patch.object(mmmu_mod, "evaluate_mmmu") as evaluate:
                    with self.assertRaisesRegex(ValueError, "invalid literal"):
                        ctx.adapter.evaluate(ctx)
                evaluate.assert_not_called()

    def test_evaluator_failure_propagates_without_fallback(self):
        for _module, adapter_cls in self.ADAPTERS:
            with self.subTest(adapter=adapter_cls.family):
                ctx = self._context(adapter_cls(), self._evaluation())
                failure = RuntimeError("MMMU inference failed")
                with patch.object(
                    mmmu_mod, "evaluate_mmmu", side_effect=failure
                ) as evaluate:
                    with self.assertRaisesRegex(
                        RuntimeError, "MMMU inference failed"
                    ) as raised:
                        ctx.adapter.evaluate(ctx)
                self.assertIs(raised.exception, failure)
                evaluate.assert_called_once()

    def test_mmmu_runs_before_ppl_with_adapter_cache_policy(self):
        for module, adapter_cls in self.ADAPTERS:
            with self.subTest(adapter=adapter_cls.family):
                evaluation = self._evaluation()
                evaluation["selected_tasks"] = ["ppl", "mmmu"]
                ctx = self._context(adapter_cls(), evaluation)
                events = []

                def evaluate_mmmu(**kwargs):
                    events.append("mmmu")
                    expected_cache = adapter_cls.family == "gemma4"
                    self.assertEqual(ctx.model.config.use_cache, expected_cache)
                    self.assertEqual(
                        ctx.model.config.text_config.use_cache, expected_cache
                    )
                    return {}

                def evaluate_ppl(**kwargs):
                    events.append("ppl")
                    return 1.0

                with patch.object(
                    mmmu_mod, "evaluate_mmmu", side_effect=evaluate_mmmu
                ), patch.object(
                    module, "evaluate_vlm_text_ppl", side_effect=evaluate_ppl
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    ctx.adapter.evaluate(ctx)
                self.assertEqual(events, ["mmmu", "ppl"])


if __name__ == "__main__":
    unittest.main()
