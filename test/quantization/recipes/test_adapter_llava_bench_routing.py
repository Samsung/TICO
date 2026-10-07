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
from unittest.mock import DEFAULT, patch

import tico.quantization.recipes.adapters.gemma4 as gemma4_mod
import tico.quantization.recipes.adapters.qwen3_vl as qwen3_vl_mod
import tico.quantization.recipes.evaluation.llava_bench as llava_bench_mod

import torch
from tico.quantization.recipes.context import RecipeContext

LEGACY_WARNING_FRAGMENT = "evaluation.llava_bench=true uses the legacy"
LEGACY_TITLE = "\n=== Llava Bench Evaluation ==="
NESTED_LEGACY_TITLE = "\n=== LLaVA Bench Legacy COCO-style Evaluation ==="


class TestAdapterLlavaBenchRouting(unittest.TestCase):
    """Shared LLaVA-Bench routing contract driven through both VLM adapters.

    Every case calls ``adapter.evaluate()`` so that the adapter -> shared
    routing path is executed for real. Only the evaluator boundaries that
    would download data or run inference are replaced.
    """

    ADAPTERS = (
        ("qwen3_vl", qwen3_vl_mod, qwen3_vl_mod.Qwen3VLAdapter),
        ("gemma4", gemma4_mod, gemma4_mod.Gemma4Adapter),
    )

    @staticmethod
    def _make_context(adapter: Any, evaluation: dict[str, Any]) -> RecipeContext:
        """Build a small VLM context with KV cache disabled as after load_model."""
        ctx = RecipeContext(
            cfg={
                "model": {"name_or_path": f"{adapter.family}-test"},
                "runtime": {"show_progress": False},
                "calibration": {"seq_len": 256},
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

    @contextlib.contextmanager
    def _patched_llava_evaluators(self):
        """Replace both LLaVA evaluator boundaries and capture printed output."""
        with patch.multiple(
            llava_bench_mod,
            evaluate_and_print_llava_bench_judge=DEFAULT,
            evaluate_llava_bench=DEFAULT,
            print_coco_score_results=DEFAULT,
        ) as mocks:
            mocks["evaluate_and_print_llava_bench_judge"].return_value = {"count": 1}
            mocks["evaluate_llava_bench"].return_value = {
                "CIDEr": 0.5,
                "total_count": 1,
                "skipped_count": 0,
            }
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                yield mocks, stdout

    def _run(self, adapter: Any, evaluation: dict[str, Any]):
        """Run adapter.evaluate() and return (mocks, stdout, ctx)."""
        ctx = self._make_context(adapter, evaluation)
        with self._patched_llava_evaluators() as (mocks, stdout):
            adapter.evaluate(ctx)
        return mocks, stdout, ctx

    def _assert_judge_only(self, mocks, *, times: int = 1) -> None:
        self.assertEqual(
            mocks["evaluate_and_print_llava_bench_judge"].call_count, times
        )
        mocks["evaluate_llava_bench"].assert_not_called()
        mocks["print_coco_score_results"].assert_not_called()

    def _assert_legacy_only(self, mocks, *, title: str) -> None:
        mocks["evaluate_llava_bench"].assert_called_once()
        mocks["evaluate_and_print_llava_bench_judge"].assert_not_called()
        mocks["print_coco_score_results"].assert_called_once_with(
            title, mocks["evaluate_llava_bench"].return_value
        )

    def _assert_none_called(self, mocks) -> None:
        mocks["evaluate_and_print_llava_bench_judge"].assert_not_called()
        mocks["evaluate_llava_bench"].assert_not_called()
        mocks["print_coco_score_results"].assert_not_called()

    # --- judge routing -------------------------------------------------

    def test_mapping_mode_omitted_defaults_to_judge_with_forwarded_args(self):
        """A nested mapping without mode routes to judge with identity preserved."""
        for family, _module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                adapter = adapter_cls()
                llava_cfg = {"enabled": True, "n_samples": 2}
                mocks, _stdout, ctx = self._run(
                    adapter,
                    {
                        "enabled": True,
                        "n_samples": 7,
                        "max_seq_len": 512,
                        "llava_bench": llava_cfg,
                    },
                )
                self._assert_judge_only(mocks)
                kwargs = mocks["evaluate_and_print_llava_bench_judge"].call_args.kwargs
                self.assertIs(kwargs["model"], ctx.model)
                self.assertIs(kwargs["processor"], ctx.processor)
                self.assertEqual(kwargs["device"], "cpu")
                self.assertIs(kwargs["llava_cfg"], llava_cfg)
                self.assertIs(kwargs["model_cfg"], ctx.cfg["model"])
                self.assertIs(kwargs["runtime_cfg"], ctx.cfg["runtime"])
                self.assertEqual(kwargs["default_n_samples"], 7)
                self.assertEqual(kwargs["default_max_seq_len"], 512)

    def test_judge_mode_aliases_route_to_judge_evaluator(self):
        """Both judge spellings route to the judge evaluator, case-insensitively."""
        for family, _module, adapter_cls in self.ADAPTERS:
            for mode in ("judge", "llm_judge", "Judge", "LLM_JUDGE"):
                with self.subTest(family=family, mode=mode):
                    mocks, _stdout, _ctx = self._run(
                        adapter_cls(),
                        {
                            "enabled": True,
                            "llava_bench": {"enabled": True, "mode": mode},
                        },
                    )
                    self._assert_judge_only(mocks)

    def test_selected_llava_with_null_or_false_config_uses_default_judge(self):
        """Explicit selection with null/false config runs judge with empty cfg."""
        for family, _module, adapter_cls in self.ADAPTERS:
            for raw_cfg in (None, False):
                with self.subTest(family=family, raw_cfg=raw_cfg):
                    mocks, _stdout, ctx = self._run(
                        adapter_cls(),
                        {
                            "enabled": True,
                            "selected_tasks": ["llava_bench"],
                            "n_samples": 3,
                            "max_seq_len": None,
                            "llava_bench": raw_cfg,
                        },
                    )
                    self._assert_judge_only(mocks)
                    kwargs = mocks[
                        "evaluate_and_print_llava_bench_judge"
                    ].call_args.kwargs
                    self.assertEqual(kwargs["llava_cfg"], {})
                    self.assertIs(kwargs["model_cfg"], ctx.cfg["model"])
                    self.assertIs(kwargs["runtime_cfg"], ctx.cfg["runtime"])
                    self.assertEqual(kwargs["default_n_samples"], 3)
                    self.assertIsNone(kwargs["default_max_seq_len"])

    def test_unselected_null_or_false_config_does_not_run(self):
        """Without explicit selection, null/false llava_bench stays off."""
        for family, _module, adapter_cls in self.ADAPTERS:
            for raw_cfg in (None, False):
                with self.subTest(family=family, raw_cfg=raw_cfg):
                    mocks, _stdout, _ctx = self._run(
                        adapter_cls(),
                        {"enabled": True, "llava_bench": raw_cfg},
                    )
                    self._assert_none_called(mocks)

    # --- legacy routing ------------------------------------------------

    def test_legacy_mode_aliases_route_to_legacy_evaluator_with_overrides(self):
        """legacy/coco/caption use the legacy evaluator with nested overrides."""
        for family, _module, adapter_cls in self.ADAPTERS:
            for mode in ("legacy", "coco", "caption", "Legacy"):
                with self.subTest(family=family, mode=mode):
                    mocks, stdout, ctx = self._run(
                        adapter_cls(),
                        {
                            "enabled": True,
                            "n_samples": 9,
                            "max_seq_len": 2048,
                            "llava_bench": {
                                "enabled": True,
                                "mode": mode,
                                "n_samples": "4",
                                "max_seq_len": 1024,
                            },
                        },
                    )
                    self._assert_legacy_only(mocks, title=NESTED_LEGACY_TITLE)
                    kwargs = mocks["evaluate_llava_bench"].call_args.kwargs
                    self.assertIs(kwargs["model"], ctx.model)
                    self.assertIs(kwargs["processor"], ctx.processor)
                    self.assertEqual(kwargs["device"], "cpu")
                    self.assertEqual(kwargs["n_samples"], 4)
                    self.assertEqual(kwargs["max_seq_len"], 1024)
                    self.assertNotIn(LEGACY_WARNING_FRAGMENT, stdout.getvalue())

    def test_legacy_mode_falls_back_to_top_level_defaults(self):
        """Nested legacy mode without overrides uses evaluation-level defaults."""
        for family, _module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                mocks, _stdout, _ctx = self._run(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "n_samples": 9,
                        "max_seq_len": 2048,
                        "llava_bench": {"enabled": True, "mode": "legacy"},
                    },
                )
                self._assert_legacy_only(mocks, title=NESTED_LEGACY_TITLE)
                kwargs = mocks["evaluate_llava_bench"].call_args.kwargs
                self.assertEqual(kwargs["n_samples"], 9)
                self.assertEqual(kwargs["max_seq_len"], 2048)

    def test_boolean_true_runs_legacy_with_warning(self):
        """llava_bench=true keeps the legacy path, its warning, and its title."""
        for family, _module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                mocks, stdout, _ctx = self._run(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "n_samples": 3,
                        "max_seq_len": 128,
                        "llava_bench": True,
                    },
                )
                self._assert_legacy_only(mocks, title=LEGACY_TITLE)
                kwargs = mocks["evaluate_llava_bench"].call_args.kwargs
                self.assertEqual(kwargs["n_samples"], 3)
                self.assertEqual(kwargs["max_seq_len"], 128)
                self.assertIn(LEGACY_WARNING_FRAGMENT, stdout.getvalue())

    # --- selection -----------------------------------------------------

    def test_evaluation_disabled_runs_nothing(self):
        """evaluation.enabled=false skips LLaVA even when it is selected."""
        for family, _module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                mocks, _stdout, _ctx = self._run(
                    adapter_cls(),
                    {
                        "enabled": False,
                        "selected_tasks": ["llava_bench"],
                        "llava_bench": {"enabled": True, "mode": "judge"},
                    },
                )
                self._assert_none_called(mocks)

    def test_nested_enabled_false_without_selection_does_not_run(self):
        """Nested enabled=false is honored when selected_tasks is absent."""
        for family, _module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                mocks, _stdout, _ctx = self._run(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "llava_bench": {"enabled": False, "mode": "judge"},
                    },
                )
                self._assert_none_called(mocks)

    def test_selected_tasks_override_nested_enabled_false(self):
        """Explicit selection runs LLaVA despite nested enabled=false."""
        for family, _module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                mocks, _stdout, _ctx = self._run(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "selected_tasks": ["llava_bench"],
                        "llava_bench": {"enabled": False, "mode": "legacy"},
                    },
                )
                self._assert_legacy_only(mocks, title=NESTED_LEGACY_TITLE)

    def test_selected_tasks_exclude_enabled_llava(self):
        """A selected_tasks list without llava_bench excludes an enabled config."""
        for family, module, adapter_cls in self.ADAPTERS:
            for selected in (["mmmu"], []):
                with self.subTest(family=family, selected=selected):
                    ctx = self._make_context(
                        adapter_cls(),
                        {
                            "enabled": True,
                            "selected_tasks": selected,
                            "llava_bench": {"enabled": True, "mode": "judge"},
                            "mmmu": {"enabled": False},
                        },
                    )
                    with patch.object(
                        module, "evaluate_and_print_mmmu"
                    ) as mmmu, self._patched_llava_evaluators() as (mocks, _):
                        ctx.adapter.evaluate(ctx)
                    self._assert_none_called(mocks)
                    self.assertEqual(mmmu.call_count, 1 if selected else 0)

    def test_unselected_llava_skips_mode_and_type_validation(self):
        """Invalid mode/type is not validated when LLaVA does not run."""
        for family, _module, adapter_cls in self.ADAPTERS:
            for evaluation in (
                {
                    "enabled": True,
                    "selected_tasks": [],
                    "llava_bench": {"enabled": True, "mode": "bogus"},
                },
                {
                    "enabled": True,
                    "llava_bench": {"enabled": False, "mode": "bogus"},
                },
                {
                    "enabled": True,
                    "selected_tasks": [],
                    "llava_bench": "not-a-mapping",
                },
            ):
                with self.subTest(family=family, evaluation=evaluation):
                    mocks, _stdout, _ctx = self._run(adapter_cls(), evaluation)
                    self._assert_none_called(mocks)

    # --- failure paths -------------------------------------------------

    def test_unknown_mode_raises_value_error_before_any_evaluator(self):
        """An unsupported mode fails with the documented message."""
        for family, _module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                ctx = self._make_context(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "llava_bench": {"enabled": True, "mode": "bogus"},
                    },
                )
                with self._patched_llava_evaluators() as (mocks, _stdout):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"evaluation\.llava_bench\.mode must be one of .* got 'bogus'",
                    ):
                        ctx.adapter.evaluate(ctx)
                self._assert_none_called(mocks)

    def test_unsupported_type_raises_type_error_before_any_evaluator(self):
        """A non-mapping, non-boolean value fails with the documented message."""
        for family, _module, adapter_cls in self.ADAPTERS:
            for raw_cfg in ("judge", 1, ["judge"]):
                with self.subTest(family=family, raw_cfg=raw_cfg):
                    ctx = self._make_context(
                        adapter_cls(),
                        {"enabled": True, "llava_bench": raw_cfg},
                    )
                    with self._patched_llava_evaluators() as (mocks, _stdout):
                        with self.assertRaisesRegex(
                            TypeError,
                            r"evaluation\.llava_bench must be a mapping, boolean, "
                            r"or null\.",
                        ):
                            ctx.adapter.evaluate(ctx)
                    self._assert_none_called(mocks)

    # --- state and ordering --------------------------------------------

    def test_original_config_is_not_mutated(self):
        """Routing must not write back into the recipe config."""
        for family, _module, adapter_cls in self.ADAPTERS:
            for raw_cfg in (
                {"enabled": True, "mode": "legacy", "n_samples": 2},
                {"enabled": True},
                True,
            ):
                with self.subTest(family=family, raw_cfg=raw_cfg):
                    evaluation = {
                        "enabled": True,
                        "n_samples": 3,
                        "max_seq_len": 64,
                        "llava_bench": raw_cfg,
                    }
                    ctx = self._make_context(adapter_cls(), evaluation)
                    snapshot = copy.deepcopy(ctx.cfg)
                    with self._patched_llava_evaluators():
                        ctx.adapter.evaluate(ctx)
                    self.assertEqual(ctx.cfg, snapshot)

    def test_gemma4_enables_cache_and_qwen_does_not(self):
        """Gemma4 keeps enabling KV cache in evaluate(); Qwen3-VL does not."""
        expected = {"gemma4": True, "qwen3_vl": False}
        for family, _module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                _mocks, _stdout, ctx = self._run(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "llava_bench": {"enabled": True, "mode": "judge"},
                    },
                )
                self.assertIs(ctx.model.config.use_cache, expected[family])
                self.assertIs(ctx.model.config.text_config.use_cache, expected[family])

    def test_llava_runs_between_coco_and_mapping_targets(self):
        """LLaVA keeps its position after COCO and before Video-MME/MMMU/PPL."""
        for family, module, adapter_cls in self.ADAPTERS:
            with self.subTest(family=family):
                order: list[str] = []

                def record(name):
                    def side_effect(*args, **kwargs):
                        order.append(name)
                        return {"CIDEr": 1.0} if name == "coco" else 2.0

                    return side_effect

                ctx = self._make_context(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "n_samples": 2,
                        "max_seq_len": 64,
                        "coco": True,
                        "llava_bench": {"enabled": True, "mode": "judge"},
                        "videomme": {"enabled": True},
                        "mmmu": {"enabled": True},
                        "ppl": {"enabled": True},
                    },
                )
                with patch.multiple(
                    module,
                    evaluate_vqa_tasks=DEFAULT,
                    evaluate_coco=DEFAULT,
                    evaluate_and_print_video_mme=DEFAULT,
                    evaluate_and_print_mmlu=DEFAULT,
                    evaluate_and_print_hellaswag=DEFAULT,
                    evaluate_and_print_mmmu=DEFAULT,
                    evaluate_vlm_text_ppl=DEFAULT,
                ) as mocks, self._patched_llava_evaluators() as (llava_mocks, _):
                    mocks["evaluate_coco"].side_effect = record("coco")
                    mocks["evaluate_and_print_video_mme"].side_effect = record(
                        "videomme"
                    )
                    mocks["evaluate_and_print_mmmu"].side_effect = record("mmmu")
                    mocks["evaluate_vlm_text_ppl"].side_effect = record("ppl")
                    llava_mocks[
                        "evaluate_and_print_llava_bench_judge"
                    ].side_effect = record("llava_judge")
                    ctx.adapter.evaluate(ctx)

                self.assertEqual(
                    order, ["coco", "llava_judge", "videomme", "mmmu", "ppl"]
                )
                mocks["evaluate_vqa_tasks"].assert_not_called()
                mocks["evaluate_and_print_mmlu"].assert_not_called()
                mocks["evaluate_and_print_hellaswag"].assert_not_called()
                llava_mocks["evaluate_llava_bench"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
