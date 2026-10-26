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

"""Hook-lifetime tests for ``collect_forward_outputs`` in the debug trace helper."""

import contextlib
import io
import unittest
from collections import OrderedDict
from unittest.mock import patch

import tico.quantization.recipes.debug.trace as trace_mod

import torch
import torch.nn as nn
from tico.quantization.recipes.debug.trace import collect_forward_outputs


class _Boom(nn.Module):
    """Submodule that always fails and remembers the exception it raised."""

    def __init__(self):
        super().__init__()
        self.last_error: BaseException | None = None

    def forward(self, x):
        self.last_error = RuntimeError("boom")
        raise self.last_error


class QuantPassthrough(nn.Module):
    """Identity module whose class name starts with ``Quant`` (skip_wrappers target)."""

    def forward(self, x):
        return x


class _TraceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(2, 2)
        self.blk = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        self.quant = QuantPassthrough()
        self.boom = _Boom()
        self.fail = False
        self.forward_calls = 0

    def forward(self, x, scale=1.0):
        self.forward_calls += 1
        x = self.pre(x) * scale
        if self.fail:
            x = self.boom(x)
        return self.quant(self.blk(x))


class TestCollectForwardOutputsHookLifetime(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _TraceModel().eval()
        self.x = torch.tensor([[0.5, -1.5]])

        # Sentinel hook owned by the "user"; must survive every tracing call.
        self.sentinel_calls: list[str] = []
        sentinel = self.model.pre.register_forward_hook(
            lambda m, i, o: self.sentinel_calls.append(type(m).__name__)
        )
        self.addCleanup(sentinel.remove)
        self.baseline_hook_counts = self._hook_counts()

        # Spy on the real registration API to learn which handles tracing created.
        self.recorded: list[tuple[nn.Module, torch.utils.hooks.RemovableHandle]] = []
        self.fail_registration_at: int | None = None
        self.registration_error: list[BaseException] = []
        original = nn.Module.register_forward_hook

        def spy(module, *args, **kwargs):
            if (
                self.fail_registration_at is not None
                and len(self.recorded) >= self.fail_registration_at
            ):
                self.registration_error.append(RuntimeError("register boom"))
                raise self.registration_error[-1]
            handle = original(module, *args, **kwargs)
            self.recorded.append((module, handle))
            return handle

        patcher = patch.object(nn.Module, "register_forward_hook", spy)
        patcher.start()
        self.addCleanup(patcher.stop)

        def remove_recorded_handles():
            for _, handle in self.recorded:
                handle.remove()

        self.addCleanup(remove_recorded_handles)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _hook_counts(self) -> dict[str, int]:
        return {n: len(m._forward_hooks) for n, m in self.model.named_modules()}

    def _assert_tracing_hooks_removed(self):
        self.assertGreater(len(self.recorded), 0, "spy did not observe registration")
        for module, handle in self.recorded:
            self.assertNotIn(handle.id, module._forward_hooks)
        self.assertEqual(self._hook_counts(), self.baseline_hook_counts)

    def _assert_plain_forward_is_clean(self):
        """A plain forward after tracing must only fire the sentinel, never a stale trace hook."""
        sentinel_before = len(self.sentinel_calls)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), torch.no_grad():
            self.model(self.x)
        self.assertEqual(buffer.getvalue(), "")
        self.assertEqual(len(self.sentinel_calls), sentinel_before + 1)

    def _collect_failing(self, **kwargs):
        buffer = io.StringIO()
        with self.assertRaises(RuntimeError) as cm, contextlib.redirect_stdout(buffer):
            collect_forward_outputs(self.model, self.x, print_trace=True, **kwargs)
        return cm.exception

    # ------------------------------------------------------------------ #
    # A. forward / callback failure and retry
    # ------------------------------------------------------------------ #
    def test_forward_failure_removes_hooks_and_retry_succeeds(self):
        self.model.fail = True
        for attempt in range(3):
            with self.subTest(attempt=attempt):
                error = self._collect_failing()
                self.assertIs(error, self.model.boom.last_error)
                self._assert_tracing_hooks_removed()
        self.assertEqual(self.model.forward_calls, 3)

        self.model.fail = False
        self._assert_plain_forward_is_clean()

        outputs = collect_forward_outputs(self.model, self.x, print_trace=False)
        self.assertEqual(list(outputs.keys()), ["pre", "blk.0", "blk.1", "blk", ""])
        with torch.no_grad():
            self.assertTrue(torch.equal(outputs["pre"], self.model.pre(self.x)))
            self.assertTrue(torch.equal(outputs[""], self.model(self.x)))
        self._assert_tracing_hooks_removed()
        self._assert_plain_forward_is_clean()

    def test_trace_callback_failure_removes_hooks(self):
        def broken_summarize(value):
            raise ValueError("summarize boom")

        buffer = io.StringIO()
        with patch.object(trace_mod, "_summarize", broken_summarize):
            with (
                self.assertRaisesRegex(ValueError, "summarize boom"),
                contextlib.redirect_stdout(buffer),
            ):
                collect_forward_outputs(self.model, self.x, print_trace=True)

        self.assertEqual(self.model.forward_calls, 1)
        self._assert_tracing_hooks_removed()
        self._assert_plain_forward_is_clean()

    # ------------------------------------------------------------------ #
    # B. failure in the middle of registration
    # ------------------------------------------------------------------ #
    def test_partial_registration_failure_removes_registered_hooks(self):
        # named_modules order: "", "pre", "blk", ... -> the third call fails.
        self.fail_registration_at = 2

        with self.assertRaises(RuntimeError) as cm:
            collect_forward_outputs(self.model, self.x, print_trace=False)

        self.assertEqual(len(self.registration_error), 1)
        self.assertIs(cm.exception, self.registration_error[0])
        self.assertEqual(len(self.recorded), 2)
        self.assertEqual(
            [module for module, _ in self.recorded], [self.model, self.model.pre]
        )
        self.assertEqual(self.model.forward_calls, 0)
        self._assert_tracing_hooks_removed()

        self.fail_registration_at = None
        self._assert_plain_forward_is_clean()

    # ------------------------------------------------------------------ #
    # E. normal behavior preserved
    # ------------------------------------------------------------------ #
    def test_input_dispatch(self):
        with torch.no_grad():
            expected_default = self.model(self.x)
            expected_scaled = self.model(self.x, scale=2.0)
        cases = {
            "tensor": (self.x, expected_default),
            "tuple": ((self.x, 2.0), expected_scaled),
            "mapping": ({"x": self.x, "scale": 2.0}, expected_scaled),
        }
        for label, (inputs, expected) in cases.items():
            with self.subTest(inputs=label):
                outputs = collect_forward_outputs(self.model, inputs, print_trace=False)
                self.assertTrue(torch.equal(outputs[""], expected))
                self._assert_tracing_hooks_removed()

    def test_filtering_order_identity_and_print_trace(self):
        outputs = collect_forward_outputs(self.model, self.x, print_trace=False)
        self.assertEqual(list(outputs.keys()), ["pre", "blk.0", "blk.1", "blk", ""])
        # Stored objects keep their identity: the root returns blk's tensor as is.
        self.assertIs(outputs[""], outputs["blk"])
        self.assertIs(outputs["blk"], outputs["blk.1"])

        outputs = collect_forward_outputs(
            self.model, self.x, print_trace=False, skip_wrappers=False
        )
        self.assertEqual(
            list(outputs.keys()), ["pre", "blk.0", "blk.1", "blk", "quant", ""]
        )

        outputs = collect_forward_outputs(
            self.model, self.x, interesting_modules=("blk",), print_trace=False
        )
        self.assertEqual(list(outputs.keys()), ["blk.0", "blk.1", "blk"])

        outputs = collect_forward_outputs(
            self.model, self.x, interesting_modules=("pre",), print_trace=False
        )
        self.assertEqual(list(outputs.keys()), ["pre"])

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            collect_forward_outputs(self.model, self.x, print_trace=True)
        printed = buffer.getvalue()
        self.assertIn("[pre] Tensor(shape=(1, 2)", printed)
        self.assertIn("[blk] Tensor(shape=(1, 2)", printed)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            collect_forward_outputs(self.model, self.x, print_trace=False)
        self.assertEqual(buffer.getvalue(), "")

        self._assert_tracing_hooks_removed()
        self._assert_plain_forward_is_clean()


class TestCollectForwardOutputsModuleSelection(unittest.TestCase):
    def _assert_selection(self, model, interesting, expected, *, skip_wrappers=True):
        value = torch.tensor([-1.0, 2.0])
        before = value.clone()
        outputs = collect_forward_outputs(
            model,
            value,
            interesting_modules=interesting,
            print_trace=False,
            skip_wrappers=skip_wrappers,
        )
        self.assertEqual(list(outputs), expected)
        self.assertTrue(torch.equal(value, before))
        for output in outputs.values():
            self.assertIs(output, value)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_numeric_component_selects_only_itself_and_descendants(self):
        model = nn.Sequential(
            OrderedDict(
                layers=nn.Sequential(*(nn.Sequential(nn.Identity()) for _ in range(12)))
            )
        )
        self._assert_selection(model, ("layers.1",), ["layers.1.0", "layers.1"])
        self._assert_selection(model, ("layers.1.0",), ["layers.1.0"])

    def test_named_component_excludes_siblings_with_a_shared_prefix(self):
        model = nn.Sequential(
            OrderedDict(
                block=nn.Sequential(nn.Identity()),
                block_extra=nn.Sequential(nn.Identity()),
                blocker=nn.Identity(),
            )
        )
        for interesting, expected in (
            (("block",), ["block.0", "block"]),
            (
                ("block_extra", "block"),
                ["block.0", "block", "block_extra.0", "block_extra"],
            ),
            (("missing",), []),
        ):
            with self.subTest(interesting=interesting):
                self._assert_selection(model, interesting, expected)

    def test_empty_filter_and_empty_prefix_keep_all_modules(self):
        model = nn.Sequential(
            OrderedDict(block=nn.Sequential(nn.Identity()), block_extra=nn.Identity())
        )
        expected = ["block.0", "block", "block_extra", ""]
        for interesting in ((), ("",), ("block", "")):
            with self.subTest(interesting=interesting):
                self._assert_selection(model, interesting, expected)

    def test_boundary_matching_uses_normalized_wrapper_paths(self):
        model = nn.Sequential(
            OrderedDict(
                block=nn.Sequential(OrderedDict(wrapped=QuantPassthrough())),
                block_extra=nn.Identity(),
            )
        )
        for skip_wrappers in (True, False):
            with self.subTest(skip_wrappers=skip_wrappers):
                self._assert_selection(
                    model, ("block",), ["block"], skip_wrappers=skip_wrappers
                )


if __name__ == "__main__":
    unittest.main()
