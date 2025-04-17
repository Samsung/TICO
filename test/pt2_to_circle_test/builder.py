# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import contextlib
import io
import os
import unittest
from pathlib import Path

import torch

from test.pt2_to_circle_test.test_pt2_to_circle import (
    convert_nnmodule_to_circle,
    convert_nnmodule_to_pt2,
    convert_pt2_to_circle,
    infer_circle,
    infer_nnmodule,
    resize_circle,
    validate_result,
    verify_circle,
)
from test.utils.base_builders import TestDictBuilderBase, TestRunnerBase

from test.utils.tag import is_tagged


class NNModuleTest(TestRunnerBase):
    def __init__(self, test_name: str, nnmodule: torch.nn.Module):
        super().__init__(test_name, nnmodule)
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "artifacts"

        # Get tags
        self.test_without_pt2: bool = is_tagged(self.nnmodule, "test_without_pt2")

        # Set tolerance
        self.tolerance = {}
        if hasattr(self.nnmodule, "rtol"):
            self.tolerance["rtol"] = self.nnmodule.rtol
        if hasattr(self.nnmodule, "atol"):
            self.tolerance["atol"] = self.nnmodule.atol

    def make(self):
        if self.skip:

            @unittest.skip(self.skip_reason)
            def wrapper(s):
                self._run()

            return wrapper
        elif self.test_negative:

            def wrapper(s):
                # Suppress the error message by redirecting stdout and discarding it.
                # Since the argument of `redirect_stdout` should have `isatty()` method, `io.StringIO()` is used.
                with contextlib.redirect_stdout(io.StringIO()):
                    with s.assertRaises(Exception) as e:
                        self._run(without_pt2=True)
                    assert self.expected_err in str(
                        e.exception
                    ), f"\nExpected the error message: {self.expected_err}\nbut the actual error message: {str(e.exception)}"

            return wrapper
        else:

            def wrapper(s):
                self._run(without_pt2=self.test_without_pt2)

            return wrapper

    def _run(self, without_pt2=False):
        test_prefix = self.test_dir / self.test_name.replace(
            "test.modules.", ""
        ).replace(".", "/")

        os.makedirs(os.path.dirname(test_prefix), exist_ok=True)

        circle_model_path = str(test_prefix) + ".circle"
        resized_circle_model_path = str(test_prefix) + ".resized.circle"
        opt_circle_model_path = str(test_prefix) + ".opt.circle"
        pt2_model_path = str(test_prefix) + ".pt2"

        if without_pt2:
            # torch.nn.Module --> ExportedProgram --> pt2 ----- (ExportedProgram) ------- > circle
            #                                       (--> load_from_pt2_file -->)
            convert_nnmodule_to_circle(
                self.nnmodule, self.example_inputs, circle_model_path
            )
        else:
            # torch.nn.Module --> ExportedProgram ----------------------------------------> circle
            convert_nnmodule_to_pt2(
                self.nnmodule, self.example_inputs, pt2_model_path, self.dynamic_shapes
            )
            convert_pt2_to_circle(pt2_model_path, circle_model_path)

        if self.dynamic_shapes:
            resize_circle(
                circle_model_path, resized_circle_model_path, self.example_inputs
            )

        verify_circle(
            resized_circle_model_path if self.dynamic_shapes else circle_model_path,
            opt_circle_model_path,
        )

        torch_result = infer_nnmodule(self.nnmodule, self.example_inputs)
        circle_result = infer_circle(opt_circle_model_path, self.example_inputs)
        validate_result(torch_result, circle_result, **self.tolerance)


class NormalTestDictBuilder(TestDictBuilderBase):
    def __init__(self, namespace: str):
        super().__init__(namespace)

    def build(self, submodule):
        """
        Return a dictionary of tests for a submodule
        key: module name (must match the module name in the source code, e.g., test.modules.op.add.SimpleAdd)
        value: a function that runs the test for the module
        """
        testdict = {}
        for nnmodule_cls in self._get_nnmodules(submodule):
            nnmodule_name = f"{submodule}.{nnmodule_cls.__name__}"

            testdict[nnmodule_name] = NNModuleTest(nnmodule_name, nnmodule_cls()).make()
        return testdict
