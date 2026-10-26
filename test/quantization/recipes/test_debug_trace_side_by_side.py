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
Characterization tests for the multi-batch side-by-side comparison helpers.

``merge_diff_stats`` and ``trace_parity`` compare two already prepared models
over several calibration inputs. Expected values below are derived by hand so
the per-batch summary aggregation (equal batch weight, not element-pooled
statistics) is pinned.
"""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import contextlib
import io
import math
import re
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import tico.quantization.examples.compare_side_by_side as example_mod
import tico.quantization.recipes.debug.trace as trace_mod

import torch
import torch.nn as nn
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.trace import merge_diff_stats, trace_parity
from tico.quantization.wrapq.utils.introspection import DifferenceStatistics

_SUMMARY_RE = re.compile(
    r"^(?P<name>.*): mean\|diff\|=(?P<mean>[^,]+), max\|diff\|=(?P<max>[^,]+), "
    r"peir=(?P<peir>.+)%$"
)


def _stats(mean: float, min: float, max: float, stddev: float, peir: float):
    return DifferenceStatistics(mean=mean, min=min, max=max, stddev=stddev, peir=peir)


class _Scale(nn.Module):
    """Multiply by a constant; a tensor-producing leaf module for the hooks."""

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class _TwoStage(nn.Module):
    """``second(first(x)) * scale`` with a shared event log for ordering checks."""

    def __init__(self, tag: str, factor: float, log: list[Any]):
        super().__init__()
        self.tag = tag
        self.first = _Scale(factor)
        self.second = _Scale(1.0)
        self.log = log
        self.seen: list[tuple[Any, ...]] = []
        self.fail_on_call: int | None = None
        self.last_error: BaseException | None = None

    def forward(self, x, scale=1.0):
        if self.fail_on_call is not None and len(self.seen) == self.fail_on_call:
            self.last_error = RuntimeError(f"{self.tag} forward boom")
            raise self.last_error
        self.seen.append((x, scale))
        self.log.append(self.tag)
        return self.second(self.first(x)) * scale


def _make_ctx(inputs: list[Any]) -> RecipeContext:
    ctx = RecipeContext(cfg={}, adapter=None, calibration_inputs=inputs)
    ctx.device = torch.device("cpu")
    return ctx


def _parse_summary(output: str) -> list[tuple[str, str, str, str]]:
    rows = []
    for line in output.splitlines():
        match = _SUMMARY_RE.match(line)
        if match:
            rows.append((match["name"], match["mean"], match["max"], match["peir"]))
    return rows


class TestSideBySideOwnership(unittest.TestCase):
    def test_example_names_are_the_debug_functions(self):
        """The example re-exports the debug module's functions, not copies."""
        self.assertIs(example_mod.trace_parity, trace_mod.trace_parity)
        self.assertIs(example_mod.merge_diff_stats, trace_mod.merge_diff_stats)
        self.assertEqual(trace_mod.trace_parity.__module__, trace_mod.__name__)
        self.assertEqual(trace_mod.merge_diff_stats.__module__, trace_mod.__name__)


class TestMergeDiffStats(unittest.TestCase):
    def test_two_and_three_batch_running_merge(self):
        """Running merge of per-batch summaries with hand-computed expectations."""
        first = _stats(mean=1.0, min=0.0, max=2.0, stddev=1.0, peir=1.0)
        second = _stats(mean=2.0, min=0.0, max=8.0, stddev=3.0, peir=1.0)
        third = _stats(mean=5.0, min=-1.0, max=3.0, stddev=0.0, peir=0.5)
        snapshots = [(s, s._asdict().copy()) for s in (first, second, third)]

        merged2 = merge_diff_stats(first, second, n=2)
        with self.subTest("two_batches"):
            self.assertEqual(merged2.min, 0.0)
            self.assertEqual(merged2.max, 8.0)
            self.assertEqual(merged2.mean, 1.5)
            self.assertEqual(merged2.peir, 1.0)
            # sqrt(1^2 + (3^2 - 1^2) / 2) = sqrt(5)
            self.assertAlmostEqual(merged2.stddev, math.sqrt(5.0), places=12)

        merged3 = merge_diff_stats(merged2, third, n=3)
        with self.subTest("three_batches"):
            self.assertEqual(merged3.min, -1.0)
            self.assertEqual(merged3.max, 8.0)
            # 1.5 + (5 - 1.5) / 3
            self.assertAlmostEqual(merged3.mean, 1.5 + 3.5 / 3, places=12)
            # 1 + (0.5 - 1) / 3
            self.assertAlmostEqual(merged3.peir, 1.0 - 0.5 / 3, places=12)
            # sqrt(5 + (0 - 5) / 3) = sqrt(10 / 3)
            self.assertAlmostEqual(merged3.stddev, math.sqrt(10.0 / 3.0), places=12)

        with self.subTest("inputs_unchanged"):
            for stats, before in snapshots:
                self.assertEqual(stats._asdict(), before)
            self.assertIsInstance(merged3, DifferenceStatistics)

    def test_rejects_n_not_greater_than_one(self):
        first = _stats(mean=1.0, min=0.0, max=2.0, stddev=1.0, peir=1.0)
        for n in (0, 1):
            with self.subTest(n=n), self.assertRaises(AssertionError):
                merge_diff_stats(first, first, n=n)

    def test_nan_peir_propagates(self):
        """A NaN PEIR from a zero-range batch stays NaN in the running average."""
        finite = _stats(mean=1.0, min=0.0, max=2.0, stddev=1.0, peir=1.0)
        zero_range = _stats(mean=0.0, min=0.0, max=0.0, stddev=0.0, peir=float("nan"))
        for label, acc, new in (
            ("new_nan", finite, zero_range),
            ("acc_nan", zero_range, finite),
        ):
            with self.subTest(label):
                merged = merge_diff_stats(acc, new, n=2)
                self.assertTrue(math.isnan(merged.peir))
                self.assertEqual(merged.mean, 0.5)
                self.assertEqual(merged.max, 2.0)


class TestTraceParityMultiBatch(unittest.TestCase):
    def setUp(self):
        self.log: list[Any] = []
        self.fp = _TwoStage("fp", 1.0, self.log).eval()
        self.q = _TwoStage("q", 2.0, self.log).eval()
        self.memory_labels: list[str] = []
        self.tqdm_calls: list[dict[str, Any]] = []
        self.collected: list[Any] = []

        def fake_print_gpu_memory(label: str) -> None:
            self.log.append(f"mem:{label}")
            self.memory_labels.append(label)

        def fake_tqdm(iterable, **kwargs):
            self.tqdm_calls.append(kwargs)
            return iterable

        real_collect = trace_mod.collect_forward_outputs
        real_compare = trace_mod.compare_outputs

        def recording_collect(*args, **kwargs):
            outputs = real_collect(*args, **kwargs)
            self.collected.append(outputs)
            return outputs

        def recording_compare(*args, **kwargs):
            self.log.append("compare")
            return real_compare(*args, **kwargs)

        for target, replacement in (
            ("print_gpu_memory", fake_print_gpu_memory),
            ("tqdm", fake_tqdm),
            ("gc", SimpleNamespace(collect=lambda: self.log.append("gc"))),
            ("collect_forward_outputs", recording_collect),
            ("compare_outputs", recording_compare),
        ):
            patcher = patch.object(trace_mod, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, inputs: list[Any]) -> str:
        """Run the comparison, assert the ``None`` return, and give back stdout."""
        ctx = _make_ctx(inputs)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertIsNone(
                trace_parity(fp_model=self.fp, q_model=self.q, ctx=ctx)  # type: ignore[func-returns-value]
            )
        return buffer.getvalue()

    def _hook_counts(self, model: nn.Module) -> dict[str, int]:
        return {n: len(m._forward_hooks) for n, m in model.named_modules()}

    def test_equal_batch_weight_aggregation_and_output(self):
        """Batches of different element counts share equal weight in the summary."""
        inputs = [torch.tensor([0.0, 2.0]), torch.tensor([0.0, 0.0, 0.0, 8.0])]
        originals = [t.clone() for t in inputs]

        output = self._run(inputs)

        # Per-batch mean|diff| is 1 and 2; element-pooled mean would be 10/6.
        expected_row = ("1.50000000", "8.00000000", "100.000000")
        rows = _parse_summary(output)
        self.assertEqual([name for name, *_ in rows], ["first", "second", ""])
        for name, *values in rows:
            with self.subTest(module=name):
                self.assertEqual(tuple(values), expected_row)

        # FP runs before quantized in every batch; gc and memory log follow compare.
        self.assertEqual(
            self.log,
            [
                "fp",
                "q",
                "compare",
                "gc",
                "mem:Batch 0",
                "fp",
                "q",
                "compare",
                "gc",
                "mem:Batch 1",
            ],
        )
        self.assertEqual(len(self.fp.seen), 2)
        self.assertEqual(len(self.q.seen), 2)
        for i, sample in enumerate(inputs):
            self.assertIs(self.fp.seen[i][0], sample)
            self.assertIs(self.q.seen[i][0], sample)
        self.assertEqual(
            self.tqdm_calls, [{"unit": "batch", "desc": "side-by-side difference"}]
        )
        # Intermediate output dictionaries are cleared after each batch.
        self.assertEqual(len(self.collected), 4)
        self.assertTrue(all(len(outputs) == 0 for outputs in self.collected))
        # Original calibration inputs are untouched.
        for tensor, original in zip(inputs, originals):
            self.assertTrue(torch.equal(tensor, original))

    def test_repeated_runs_do_not_accumulate(self):
        inputs = [torch.tensor([0.0, 2.0]), torch.tensor([0.0, 0.0, 0.0, 8.0])]
        first = self._run(inputs)
        second = self._run(inputs)
        self.assertEqual(_parse_summary(first), _parse_summary(second))
        self.assertEqual(self.memory_labels, ["Batch 0", "Batch 1"] * 2)

    def test_input_dispatch_semantics(self):
        """Tensor, tuple, and mapping inputs keep their existing call semantics."""
        x = torch.tensor([0.0, 2.0])
        inputs: list[Any] = [x, (x, 2.0), {"x": x, "scale": 2.0}]
        mapping_before = dict(inputs[2])

        self._run(inputs)

        for model in (self.fp, self.q):
            with self.subTest(model=model.tag):
                self.assertEqual(len(model.seen), 3)
                self.assertIs(model.seen[0][0], x)
                self.assertEqual(model.seen[0][1], 1.0)
                self.assertIs(model.seen[1][0], x)
                self.assertEqual(model.seen[1][1], 2.0)
                self.assertIs(model.seen[2][0], x)
                self.assertEqual(model.seen[2][1], 2.0)
        self.assertEqual(inputs[2], mapping_before)
        self.assertIs(inputs[1][0], x)

    def test_empty_inputs_raise(self):
        with self.assertRaisesRegex(RuntimeError, "at least one calibration input"):
            self._run([])
        self.assertEqual(self.log, [])

    def test_forward_failure_propagates_and_leaves_no_hooks(self):
        """A forward failure in a later batch stops the loop and cleans up hooks."""
        sentinel_calls: list[int] = []
        sentinel = self.fp.first.register_forward_hook(
            lambda m, i, o: sentinel_calls.append(1)
        )
        self.addCleanup(sentinel.remove)
        baseline = (self._hook_counts(self.fp), self._hook_counts(self.q))
        inputs = [torch.tensor([0.0, 2.0])] * 3
        self.q.fail_on_call = 1

        ctx = _make_ctx(inputs)
        buffer = io.StringIO()
        with self.assertRaises(RuntimeError) as cm, contextlib.redirect_stdout(buffer):
            trace_parity(fp_model=self.fp, q_model=self.q, ctx=ctx)

        self.assertIs(cm.exception, self.q.last_error)
        self.assertEqual(self.log, ["fp", "q", "compare", "gc", "mem:Batch 0", "fp"])
        self.assertEqual(_parse_summary(buffer.getvalue()), [])
        self.assertEqual(
            (self._hook_counts(self.fp), self._hook_counts(self.q)), baseline
        )
        self.assertEqual(len(sentinel_calls), 2)
        self.assertIn(sentinel.id, self.fp.first._forward_hooks)


if __name__ == "__main__":
    unittest.main()
