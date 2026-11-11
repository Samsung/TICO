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

"""Lifecycle contracts for generated conversion tests, without exporting models."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from test.modules.base import TestModuleBase
from test.pt2_to_circle_test import builder
from test.support import tag


class LazyGeneratedTests(unittest.TestCase):
    def _case(self, *models):
        test_builder = builder.NormalTestDictBuilder("test.modules.op")
        with mock.patch.object(test_builder, "_get_nnmodules", return_value=models):
            methods = test_builder.build("test.modules.op.synthetic")
        return type("Synthetic", (unittest.TestCase,), methods)

    def _run_case(self, case, pattern=None):
        loader = unittest.TestLoader()
        if pattern is not None:
            loader.testNamePatterns = [pattern]
        result = unittest.TestResult()
        loader.loadTestsFromTestCase(case).run(result)
        return result

    def test_discovery_and_filter_only_construct_selected_model(self):
        calls = []

        class Selected(TestModuleBase):
            def __init__(self):
                super().__init__()
                calls.append("Selected constructor")

            def get_example_inputs(self):
                calls.append("Selected inputs")
                return (torch.tensor([3.0]),), {}

        class Excluded(TestModuleBase):
            def __init__(self):
                raise AssertionError("excluded constructor called")

            def get_example_inputs(self):
                raise AssertionError("excluded inputs called")

        case = self._case(Selected, Excluded)
        all_tests = unittest.TestLoader().loadTestsFromTestCase(case)
        self.assertEqual(all_tests.countTestCases(), 2)
        self.assertEqual(calls, [])
        observed = []

        def run(runner, **options):
            observed.append((type(runner.nnmodule), runner.forward_args[0].item()))

        with mock.patch.object(builder.NNModuleTest, "_run", run):
            result = self._run_case(case, "*.Selected")
        self.assertTrue(result.wasSuccessful(), result.errors)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(calls, ["Selected constructor", "Selected inputs"])
        self.assertEqual(observed, [(Selected, 3.0)])

    def test_class_skip_never_constructs_model_or_inputs(self):
        @tag.skip("synthetic unavailable backend")
        class Skipped(TestModuleBase):
            def get_example_inputs(self):
                raise AssertionError("skipped inputs called")

        with mock.patch.object(Skipped, "__init__", side_effect=AssertionError("ctor")):
            case = self._case(Skipped)
            with mock.patch.object(builder.NNModuleTest, "_run") as run:
                result = self._run_case(case)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(
            [reason for _, reason in result.skipped], ["synthetic unavailable backend"]
        )
        run.assert_not_called()

    def test_factories_bind_distinct_models_and_reloads_use_fresh_state(self):
        instances = []

        class First(TestModuleBase):
            def __init__(self):
                super().__init__()
                self.history = []
                instances.append(self)

            def get_example_inputs(self):
                return (torch.tensor([1.0]),), {}

        class Second(First):
            def get_example_inputs(self):
                return (torch.tensor([2.0]),), {}

        observed = []

        def run(runner, **options):
            self.assertEqual(runner.nnmodule.history, [])
            observed.append((type(runner.nnmodule), runner.forward_args[0].item()))
            runner.nnmodule.history.append("changed")
            runner.forward_args[0].add_(100)

        case = self._case(First, Second)
        with mock.patch.object(builder.NNModuleTest, "_run", run):
            for _ in range(2):
                result = self._run_case(case)
                self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertEqual(observed, [(First, 1.0), (Second, 2.0)] * 2)
        self.assertEqual(len({id(instance) for instance in instances}), 4)

    def test_negative_fixture_errors_are_errors_even_when_diagnostic_matches(self):
        for stage in ("constructor", "inputs"):
            with self.subTest(stage=stage):

                @tag.test_negative("expected conversion diagnostic")
                class Broken(TestModuleBase):
                    def __init__(self):
                        super().__init__()
                        if stage == "constructor":
                            raise RuntimeError("expected conversion diagnostic")

                    def get_example_inputs(self):
                        raise RuntimeError("expected conversion diagnostic")

                case = self._case(Broken)
                with mock.patch.object(builder.NNModuleTest, "_run") as run:
                    result = self._run_case(case)
                self.assertEqual(len(result.errors), 1)
                self.assertEqual(result.failures, [])
                self.assertIn("expected conversion diagnostic", result.errors[0][1])
                run.assert_not_called()

    def test_invalid_example_input_contract_is_preparation_error(self):
        inputs: tuple[object, object]
        for inputs in (([], {}), ((), [])):
            with self.subTest(inputs=inputs):

                @tag.test_negative("invalid example inputs")
                class InvalidInputs(TestModuleBase):
                    def get_example_inputs(self):
                        return inputs

                result = self._run_case(self._case(InvalidInputs))
                self.assertEqual(len(result.errors), 1)
                self.assertEqual(result.failures, [])
                self.assertIn("invalid example inputs", result.errors[0][1])

    def test_negative_execution_still_requires_expected_failure(self):
        @tag.test_negative("conversion diagnostic")
        class Negative(TestModuleBase):
            def get_example_inputs(self):
                return (), {}

        case = self._case(Negative)
        with mock.patch.object(
            builder.NNModuleTest,
            "_run",
            side_effect=RuntimeError("conversion diagnostic"),
        ) as run:
            result = self._run_case(case)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        run.assert_called_once_with(without_pt2=True)
        with mock.patch.object(builder.NNModuleTest, "_run"):
            result = self._run_case(case)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("Exception not raised", result.failures[0][1])

    def test_instance_metadata_is_resolved_after_construction(self):
        class InstanceOptions(TestModuleBase):
            def __init__(self):
                super().__init__()
                self.rtol, self.atol = 0.002, 0.003
                for option in (
                    "test_without_pt2",
                    "test_without_inference",
                    "with_golden",
                    "use_onert",
                ):
                    setattr(self, "__tag_" + option, True)

            def get_example_inputs(self):
                return (), {}

        observed = []

        def run(runner, **options):
            observed.append((runner.tolerance, options, runner.runtime()))

        case = self._case(InstanceOptions)
        with (
            mock.patch.dict(os.environ, {"CCEX_RUNTIME": "circle-interpreter"}),
            mock.patch.object(builder.NNModuleTest, "_run", run),
        ):
            result = self._run_case(case)
        self.assertTrue(result.wasSuccessful(), result.errors)
        self.assertEqual(
            observed,
            [
                (
                    {"rtol": 0.002, "atol": 0.003},
                    {
                        "without_pt2": True,
                        "without_inference": True,
                        "with_golden": True,
                    },
                    "onert",
                )
            ],
        )

    def test_runtime_specific_negative_runs_normally_on_other_runtime(self):
        @tag.test_negative("backend diagnostic", runtime="onert")
        class BackendNegative(TestModuleBase):
            def get_example_inputs(self):
                return (), {}

        case = self._case(BackendNegative)
        with (
            mock.patch.dict(os.environ, {"CCEX_RUNTIME": "reference"}),
            mock.patch.object(builder.NNModuleTest, "_run") as run,
        ):
            result = self._run_case(case)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        run.assert_called_once_with(
            without_pt2=False, without_inference=False, with_golden=False
        )

    def test_one_input_creation_is_used_through_the_execution(self):
        calls = []

        class Inputs(TestModuleBase):
            def get_example_inputs(self):
                calls.append("inputs")
                return (torch.tensor([7.0]),), {}

        observed = []

        def check_inputs(*args, forward_args, forward_kwargs, **kwargs):
            observed.append(forward_args[0].item())
            self.assertEqual(forward_kwargs, {})
            return [forward_args[0]]

        def infer(*args, forward_args, forward_kwargs, **kwargs):
            check_inputs(forward_args=forward_args, forward_kwargs=forward_kwargs)
            return [forward_args[0].numpy()]

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(builder, "infer_nnmodule", side_effect=check_inputs),
            mock.patch.object(
                builder, "convert_nnmodule_to_pt2", side_effect=check_inputs
            ),
            mock.patch.object(builder, "convert_pt2_to_circle"),
            mock.patch.object(builder, "verify_circle"),
            mock.patch.object(builder, "infer_circle", side_effect=infer),
        ):
            runner = builder.NNModuleTest("test.modules.op.synthetic.Inputs", Inputs())
            runner.test_dir = Path(temp_dir)
            runner._run()
        self.assertEqual(calls, ["inputs"])
        self.assertEqual(observed, [7.0, 7.0, 7.0])
