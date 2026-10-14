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
import tico.quantization.recipes.evaluation.vlm as vlm_mod

import torch
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.data.dataset_usage import EVALUATION_ROLE

VQA_TITLE = "VQA evaluation"
EMPTY_AFTER_NORMALIZATION_MESSAGE = (
    r"evaluation\.vlm_tasks must be non-empty when vqa runs\."
)
INVALID_TYPE_MESSAGE = (
    r"evaluation\.vlm_tasks must be a sequence or comma-separated string\."
)
NON_STRING_ELEMENT_MESSAGE = r"evaluation\.vlm_tasks must contain only strings\."
REQUIRED_FOR_SELECTION_MESSAGE = (
    r"Evaluation target 'vqa' requires non-empty evaluation\.vlm_tasks\."
)


class TestAdapterVqaRouting(unittest.TestCase):
    """Shared VQA selection/normalization contract driven through both VLM adapters.

    Every case calls ``adapter.evaluate()`` so that the adapter -> VQA
    selection -> evaluator path is executed for real. Only the evaluator
    boundary that would download data or run inference is replaced, at the
    module that owns the ``evaluate_vqa_tasks`` call.
    """

    # (family, adapter module, adapter class, module owning the VQA call)
    ADAPTERS = (
        ("qwen3_vl", qwen3_vl_mod, qwen3_vl_mod.Qwen3VLAdapter, vlm_mod),
        ("gemma4", gemma4_mod, gemma4_mod.Gemma4Adapter, vlm_mod),
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
    def _patched_vqa_evaluator(self, owner: Any, *, real_printer: bool = False):
        """Replace the VQA evaluator boundary and capture printed output."""
        targets: dict[str, Any] = {"evaluate_vqa_tasks": DEFAULT}
        if not real_printer:
            targets["print_vqa_results"] = DEFAULT
        with patch.multiple(owner, **targets) as mocks:
            mocks["evaluate_vqa_tasks"].return_value = {"vqav2": (3, 4)}
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                yield mocks, stdout

    def _run(self, adapter: Any, owner: Any, evaluation: dict[str, Any]):
        """Run adapter.evaluate() and return (mocks, stdout, ctx)."""
        ctx = self._make_context(adapter, evaluation)
        with self._patched_vqa_evaluator(owner) as (mocks, stdout):
            adapter.evaluate(ctx)
        return mocks, stdout, ctx

    def _assert_vqa_ran_once(self, mocks) -> None:
        mocks["evaluate_vqa_tasks"].assert_called_once()
        mocks["print_vqa_results"].assert_called_once_with(
            VQA_TITLE, mocks["evaluate_vqa_tasks"].return_value
        )

    def _assert_vqa_not_run(self, mocks) -> None:
        mocks["evaluate_vqa_tasks"].assert_not_called()
        mocks["print_vqa_results"].assert_not_called()

    # --- normalization -------------------------------------------------

    def test_task_list_normalization_preserves_order_and_duplicates(self):
        """Comma strings, lists, and tuples normalize identically in both adapters."""
        cases = (
            (" vqav2, , textvqa, vqav2 ", ["vqav2", "textvqa", "vqav2"]),
            ("vqav2", ["vqav2"]),
            (["vqav2", "textvqa"], ["vqav2", "textvqa"]),
            (("textvqa", "vqav2"), ["textvqa", "vqav2"]),
            ([" vqav2 ", "", "   ", "textvqa"], ["vqav2", "textvqa"]),
            # Strings inside a sequence are stripped but never re-split.
            (["vqav2,textvqa", " textvqa "], ["vqav2,textvqa", "textvqa"]),
            # Case and dataset names are passed through untouched.
            (["VQAv2", "vqav2", "VQAv2"], ["VQAv2", "vqav2", "VQAv2"]),
            # Non-"vqa" names are forwarded; filtering is the evaluator's policy.
            (["textvqa", "coco"], ["textvqa", "coco"]),
        )
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            for raw_tasks, expected in cases:
                with self.subTest(family=family, raw_tasks=raw_tasks):
                    mocks, _stdout, _ctx = self._run(
                        adapter_cls(),
                        owner,
                        {"enabled": True, "vlm_tasks": raw_tasks},
                    )
                    self._assert_vqa_ran_once(mocks)
                    tasks = mocks["evaluate_vqa_tasks"].call_args.kwargs["tasks"]
                    self.assertIs(type(tasks), list)
                    self.assertEqual(tasks, expected)

    def test_forwards_model_processor_and_explicit_arguments(self):
        """Model/processor identity and configured evaluation arguments are kept."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            with self.subTest(family=family):
                ctx = self._make_context(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "vlm_tasks": ["vqav2"],
                        "n_samples": 7,
                        "max_seq_len": 512,
                        "verbose": True,
                    },
                )
                ctx.cfg["runtime"]["show_progress"] = False
                with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                    ctx.adapter.evaluate(ctx)
                self._assert_vqa_ran_once(mocks)
                kwargs = mocks["evaluate_vqa_tasks"].call_args.kwargs
                self.assertIs(kwargs["model"], ctx.model)
                self.assertIs(kwargs["processor"], ctx.processor)
                self.assertEqual(kwargs["device"], "cpu")
                self.assertEqual(kwargs["tasks"], ["vqav2"])
                self.assertEqual(kwargs["n_samples"], 7)
                self.assertEqual(kwargs["max_seq_len"], 512)
                self.assertIs(kwargs["verbose"], True)
                self.assertIs(kwargs["show_progress"], False)

    def test_argument_defaults_are_preserved(self):
        """Omitted n_samples/max_seq_len/verbose/runtime keep their defaults."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            with self.subTest(family=family):
                ctx = self._make_context(
                    adapter_cls(), {"enabled": True, "vlm_tasks": "textvqa"}
                )
                del ctx.cfg["runtime"]
                with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                    ctx.adapter.evaluate(ctx)
                self._assert_vqa_ran_once(mocks)
                kwargs = mocks["evaluate_vqa_tasks"].call_args.kwargs
                self.assertEqual(kwargs["n_samples"], 50)
                self.assertIsNone(kwargs["max_seq_len"])
                self.assertIs(kwargs["verbose"], False)
                self.assertIs(kwargs["show_progress"], True)
                # max_seq_len=None must not be replaced by calibration.seq_len.
                self.assertEqual(ctx.cfg["calibration"]["seq_len"], 256)

    def test_results_reach_the_real_printer_with_title_and_scores(self):
        """The real printer emits the existing title and EM/n score format."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            with self.subTest(family=family):
                ctx = self._make_context(
                    adapter_cls(), {"enabled": True, "vlm_tasks": "vqav2,textvqa"}
                )
                with self._patched_vqa_evaluator(owner, real_printer=True) as (
                    mocks,
                    stdout,
                ):
                    mocks["evaluate_vqa_tasks"].return_value = {
                        "vqav2": (3, 4),
                        "textvqa": (1, 8),
                    }
                    ctx.adapter.evaluate(ctx)
                self.assertEqual(
                    stdout.getvalue(),
                    "VQA evaluation\n"
                    "vqav2: EM=0.7500  (n=4)\n"
                    "textvqa: EM=0.1250  (n=8)\n",
                )

    # --- selection -----------------------------------------------------

    def test_runs_when_vlm_tasks_present_without_or_with_null_selection(self):
        """Absent or null selected_tasks falls back to vlm_tasks truthiness."""
        evaluations: tuple[dict[str, Any], ...] = (
            {"enabled": True, "vlm_tasks": ["vqav2"]},
            {"enabled": True, "selected_tasks": None, "vlm_tasks": ["vqav2"]},
            {"enabled": True, "selected_tasks": ["vqa"], "vlm_tasks": ["vqav2"]},
        )
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            for evaluation in evaluations:
                with self.subTest(family=family, evaluation=evaluation):
                    mocks, _stdout, _ctx = self._run(adapter_cls(), owner, evaluation)
                    self._assert_vqa_ran_once(mocks)
                    self.assertEqual(
                        mocks["evaluate_vqa_tasks"].call_args.kwargs["tasks"],
                        ["vqav2"],
                    )

    def test_falsy_vlm_tasks_without_selection_does_not_run(self):
        """None/false/empty vlm_tasks stays off and raises nothing."""
        falsy_inputs: tuple[Any, ...] = (None, False, [], (), "")
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            for raw_tasks in falsy_inputs:
                with self.subTest(family=family, raw_tasks=raw_tasks):
                    mocks, _stdout, _ctx = self._run(
                        adapter_cls(),
                        owner,
                        {"enabled": True, "vlm_tasks": raw_tasks},
                    )
                    self._assert_vqa_not_run(mocks)
            with self.subTest(family=family, raw_tasks="<missing>"):
                mocks, _stdout, _ctx = self._run(
                    adapter_cls(), owner, {"enabled": True}
                )
                self._assert_vqa_not_run(mocks)

    def test_selected_tasks_excluding_vqa_skips_configured_tasks(self):
        """An allow-list without vqa excludes configured vlm_tasks."""
        for family, module, adapter_cls, owner in self.ADAPTERS:
            for selected in (["mmmu"], []):
                with self.subTest(family=family, selected=selected):
                    ctx = self._make_context(
                        adapter_cls(),
                        {
                            "enabled": True,
                            "selected_tasks": selected,
                            "vlm_tasks": ["vqav2"],
                            "mmmu": {
                                "enabled": False,
                                "dataset": "MMMU/MMMU_Pro",
                                "subjects": ["vision"],
                            },
                        },
                    )
                    with patch.object(
                        module, "evaluate_and_print_mmmu"
                    ) as mmmu, self._patched_vqa_evaluator(owner) as (mocks, _):
                        ctx.adapter.evaluate(ctx)
                    self._assert_vqa_not_run(mocks)
                    self.assertEqual(mmmu.call_count, 1 if selected else 0)

    def test_evaluation_disabled_runs_nothing(self):
        """evaluation.enabled=false skips VQA even when it is selected."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            with self.subTest(family=family):
                mocks, _stdout, _ctx = self._run(
                    adapter_cls(),
                    owner,
                    {
                        "enabled": False,
                        "selected_tasks": ["vqa"],
                        "vlm_tasks": ["vqav2"],
                    },
                )
                self._assert_vqa_not_run(mocks)

    def test_unselected_vqa_skips_task_normalization_and_validation(self):
        """Invalid vlm_tasks is not inspected when VQA does not run."""
        for family, module, adapter_cls, owner in self.ADAPTERS:
            for selected, raw_tasks in (
                ([], 123),
                ([], [1, 2]),
                ([], "   "),
                ([], {"vqav2": True}),
                (["mmmu"], 123),
                (["mmmu"], [None]),
            ):
                with self.subTest(family=family, selected=selected, raw=raw_tasks):
                    ctx = self._make_context(
                        adapter_cls(),
                        {
                            "enabled": True,
                            "selected_tasks": selected,
                            "vlm_tasks": raw_tasks,
                            "mmmu": {
                                "enabled": False,
                                "dataset": "MMMU/MMMU_Pro",
                                "subjects": ["vision"],
                            },
                        },
                    )
                    with patch.object(
                        module, "evaluate_and_print_mmmu"
                    ), self._patched_vqa_evaluator(owner) as (mocks, _):
                        ctx.adapter.evaluate(ctx)
                    self._assert_vqa_not_run(mocks)

    # --- failure paths -------------------------------------------------

    def test_empty_after_normalization_raises_when_vqa_runs(self):
        """Truthy input that normalizes to nothing fails with the existing error."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            for selected in ("<absent>", ["vqa"]):
                for raw_tasks in ("   ", " , ,", ["", "   "], ("",)):
                    with self.subTest(family=family, selected=selected, raw=raw_tasks):
                        evaluation: dict[str, Any] = {
                            "enabled": True,
                            "vlm_tasks": raw_tasks,
                        }
                        if selected != "<absent>":
                            evaluation["selected_tasks"] = selected
                        ctx = self._make_context(adapter_cls(), evaluation)
                        with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                            with self.assertRaisesRegex(
                                ValueError, EMPTY_AFTER_NORMALIZATION_MESSAGE
                            ):
                                ctx.adapter.evaluate(ctx)
                        self._assert_vqa_not_run(mocks)

    def test_explicit_vqa_selection_requires_vlm_tasks_before_execution(self):
        """Selecting vqa with empty vlm_tasks fails in adapter pre-validation."""
        empty_inputs: tuple[Any, ...] = (None, [], "", "<missing>")
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            for raw_tasks in empty_inputs:
                with self.subTest(family=family, raw_tasks=raw_tasks):
                    evaluation: dict[str, Any] = {
                        "enabled": True,
                        "selected_tasks": ["vqa"],
                    }
                    if raw_tasks != "<missing>":
                        evaluation["vlm_tasks"] = raw_tasks
                    ctx = self._make_context(adapter_cls(), evaluation)
                    with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                        with self.assertRaisesRegex(
                            ValueError, REQUIRED_FOR_SELECTION_MESSAGE
                        ):
                            ctx.adapter.evaluate(ctx)
                    self._assert_vqa_not_run(mocks)
                    # Pre-validation runs before any adapter-side mutation.
                    self.assertIs(ctx.model.config.use_cache, False)

    def test_invalid_vlm_tasks_type_raises_type_error(self):
        """Non-string, non-sequence values fail with the existing TypeError."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            for raw_tasks in (123, 1.5, True, {"vqav2": True}):
                with self.subTest(family=family, raw_tasks=raw_tasks):
                    ctx = self._make_context(
                        adapter_cls(), {"enabled": True, "vlm_tasks": raw_tasks}
                    )
                    with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                        with self.assertRaisesRegex(TypeError, INVALID_TYPE_MESSAGE):
                            ctx.adapter.evaluate(ctx)
                    self._assert_vqa_not_run(mocks)

    def test_non_string_sequence_element_raises_type_error(self):
        """Sequence elements are not coerced with str(); they must be strings."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            for raw_tasks in ([1, "vqav2"], ["vqav2", None], (b"vqav2",), [["vqav2"]]):
                with self.subTest(family=family, raw_tasks=raw_tasks):
                    ctx = self._make_context(
                        adapter_cls(), {"enabled": True, "vlm_tasks": raw_tasks}
                    )
                    with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                        with self.assertRaisesRegex(
                            TypeError, NON_STRING_ELEMENT_MESSAGE
                        ):
                            ctx.adapter.evaluate(ctx)
                    self._assert_vqa_not_run(mocks)

    def test_evaluator_failure_propagates_and_skips_later_targets(self):
        """An evaluator error is not swallowed and later targets do not run."""
        for family, module, adapter_cls, owner in self.ADAPTERS:
            with self.subTest(family=family):
                ctx = self._make_context(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "vlm_tasks": ["vqav2"],
                        "coco": True,
                        "llava_bench": {"enabled": True, "mode": "judge"},
                        "mmmu": {
                            "enabled": True,
                            "dataset": "MMMU/MMMU_Pro",
                            "subjects": ["vision"],
                        },
                    },
                )
                with patch.multiple(
                    module, evaluate_coco=DEFAULT, evaluate_and_print_mmmu=DEFAULT
                ) as later, patch.object(
                    llava_bench_mod, "evaluate_and_print_llava_bench_judge"
                ) as llava, self._patched_vqa_evaluator(
                    owner
                ) as (
                    mocks,
                    _stdout,
                ):
                    mocks["evaluate_vqa_tasks"].side_effect = RuntimeError("vqa boom")
                    with self.assertRaisesRegex(RuntimeError, "vqa boom"):
                        ctx.adapter.evaluate(ctx)
                mocks["evaluate_vqa_tasks"].assert_called_once()
                mocks["print_vqa_results"].assert_not_called()
                later["evaluate_coco"].assert_not_called()
                later["evaluate_and_print_mmmu"].assert_not_called()
                llava.assert_not_called()

    # --- state and ordering --------------------------------------------

    def test_original_config_is_not_mutated(self):
        """Normalization must not write back into the recipe config."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            for raw_tasks in (
                " vqav2, , textvqa ",
                [" vqav2 ", "", "textvqa"],
                ("textvqa",),
            ):
                with self.subTest(family=family, raw_tasks=raw_tasks):
                    evaluation = {
                        "enabled": True,
                        "n_samples": 3,
                        "max_seq_len": 64,
                        "vlm_tasks": raw_tasks,
                    }
                    ctx = self._make_context(adapter_cls(), evaluation)
                    snapshot = copy.deepcopy(ctx.cfg)
                    with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                        ctx.adapter.evaluate(ctx)
                    self._assert_vqa_ran_once(mocks)
                    self.assertEqual(ctx.cfg, snapshot)
                    self.assertIs(ctx.cfg["evaluation"], evaluation)
                    self.assertIs(evaluation["vlm_tasks"], raw_tasks)
                    self.assertIsNot(
                        mocks["evaluate_vqa_tasks"].call_args.kwargs["tasks"],
                        raw_tasks,
                    )

    def test_repeated_runs_do_not_accumulate_tasks_or_results(self):
        """Running evaluate() twice yields independent task lists and results."""
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            with self.subTest(family=family):
                ctx = self._make_context(
                    adapter_cls(), {"enabled": True, "vlm_tasks": "vqav2, textvqa"}
                )
                with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                    mocks["evaluate_vqa_tasks"].side_effect = (
                        {"vqav2": (1, 2)},
                        {"textvqa": (2, 2)},
                    )
                    ctx.adapter.evaluate(ctx)
                    ctx.adapter.evaluate(ctx)
                self.assertEqual(mocks["evaluate_vqa_tasks"].call_count, 2)
                for call in mocks["evaluate_vqa_tasks"].call_args_list:
                    self.assertEqual(call.kwargs["tasks"], ["vqav2", "textvqa"])
                self.assertEqual(
                    [call.args for call in mocks["print_vqa_results"].call_args_list],
                    [
                        (VQA_TITLE, {"vqav2": (1, 2)}),
                        (VQA_TITLE, {"textvqa": (2, 2)}),
                    ],
                )

    def test_vqa_runs_before_coco_and_llava(self):
        """VQA keeps its position ahead of COCO and LLaVA-Bench."""
        for family, module, adapter_cls, owner in self.ADAPTERS:
            with self.subTest(family=family):
                order: list[str] = []

                def record(name, value):
                    def side_effect(*args, **kwargs):
                        order.append(name)
                        return value

                    return side_effect

                ctx = self._make_context(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "n_samples": 2,
                        "max_seq_len": 64,
                        "vlm_tasks": ["vqav2"],
                        "coco": True,
                        "llava_bench": {"enabled": True, "mode": "judge"},
                    },
                )
                with patch.multiple(
                    module, evaluate_coco=DEFAULT, print_coco_score_results=DEFAULT
                ) as coco_mocks, patch.object(
                    llava_bench_mod, "evaluate_and_print_llava_bench_judge"
                ) as llava, self._patched_vqa_evaluator(
                    owner
                ) as (
                    mocks,
                    _stdout,
                ):
                    mocks["evaluate_vqa_tasks"].side_effect = record(
                        "vqa", {"vqav2": (1, 2)}
                    )
                    mocks["print_vqa_results"].side_effect = record("vqa_print", None)
                    coco_mocks["evaluate_coco"].side_effect = record(
                        "coco", {"CIDEr": 1.0}
                    )
                    llava.side_effect = record("llava_judge", {"count": 1})
                    ctx.adapter.evaluate(ctx)

                self.assertEqual(order, ["vqa", "vqa_print", "coco", "llava_judge"])

    def test_gemma4_enables_cache_and_qwen_does_not(self):
        """Gemma4 keeps enabling KV cache before VQA; Qwen3-VL does not mutate it."""
        expected = {"gemma4": True, "qwen3_vl": False}
        for family, _module, adapter_cls, owner in self.ADAPTERS:
            with self.subTest(family=family):
                ctx = self._make_context(
                    adapter_cls(), {"enabled": True, "vlm_tasks": ["vqav2"]}
                )
                observed: dict[str, Any] = {}

                def capture(**kwargs):
                    observed["use_cache"] = kwargs["model"].config.use_cache
                    return {"vqav2": (1, 1)}

                with self._patched_vqa_evaluator(owner) as (mocks, _stdout):
                    mocks["evaluate_vqa_tasks"].side_effect = capture
                    ctx.adapter.evaluate(ctx)
                self.assertIs(observed["use_cache"], expected[family])
                self.assertIs(ctx.model.config.use_cache, expected[family])
                self.assertIs(ctx.model.config.text_config.use_cache, expected[family])

    # --- real evaluator with replaced data/inference boundaries -----------

    def test_real_evaluator_receives_role_samples_and_arguments(self):
        """The real evaluate_vqa_tasks() runs with only dataset/inference replaced."""
        for family, _module, adapter_cls, _owner in self.ADAPTERS:
            with self.subTest(family=family):
                ctx = self._make_context(
                    adapter_cls(),
                    {
                        "enabled": True,
                        "vlm_tasks": " vqav2, textvqa ",
                        "n_samples": 5,
                        "max_seq_len": 128,
                    },
                )
                dataset_calls: list[tuple[Any, ...]] = []
                accuracy_calls: list[dict[str, Any]] = []
                dataset_adapter = object()
                samples = ["s1", "s2"]

                def fake_get_dataset(name, *, role, n):
                    dataset_calls.append((name, role, n))
                    return samples, dataset_adapter

                def fake_get_accuracy_on_dataset(
                    model, processor, dataset_iter, adapter, device, **kwargs
                ):
                    accuracy_calls.append(
                        {
                            "model": model,
                            "processor": processor,
                            "items": list(dataset_iter),
                            "adapter": adapter,
                            "device": device,
                            **kwargs,
                        }
                    )
                    return (len(accuracy_calls), 4)

                with patch.object(
                    vlm_mod, "get_dataset", fake_get_dataset
                ), patch.object(
                    vlm_mod, "get_accuracy_on_dataset", fake_get_accuracy_on_dataset
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ) as stdout:
                    ctx.adapter.evaluate(ctx)

                self.assertEqual(
                    dataset_calls,
                    [("vqav2", EVALUATION_ROLE, 5), ("textvqa", EVALUATION_ROLE, 5)],
                )
                self.assertEqual(len(accuracy_calls), 2)
                for call in accuracy_calls:
                    self.assertIs(call["model"], ctx.model)
                    self.assertIs(call["processor"], ctx.processor)
                    self.assertEqual(call["items"], samples)
                    self.assertIs(call["adapter"], dataset_adapter)
                    self.assertEqual(call["device"], "cpu")
                    self.assertEqual(call["max_seq_len"], 128)
                    self.assertIs(call["verbose"], False)
                self.assertEqual(
                    stdout.getvalue(),
                    "VQA evaluation\n"
                    "vqav2: EM=0.2500  (n=4)\n"
                    "textvqa: EM=0.5000  (n=4)\n",
                )


if __name__ == "__main__":
    unittest.main()
