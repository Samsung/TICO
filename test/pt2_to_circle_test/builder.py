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
from copy import deepcopy
from pathlib import Path
from typing import Optional

from tico.config.base import CompileConfigBase
from tico.utils.signature import ModelInputSpec

from test.modules.base import TestModuleBase

from test.pt2_to_circle_test.test_pt2_to_circle import (
    convert_nnmodule_to_circle,
    convert_nnmodule_to_pt2,
    convert_pt2_to_circle,
    infer_circle,
    infer_nnmodule,
    validate_result,
    verify_circle,
)
from test.utils.base_builders import TestDictBuilderBase, TestRunnerBase
from test.utils.tag import is_tagged


@contextlib.contextmanager
def suppress_stderr_fd():
    """Suppress C/C++ level stderr output (fd=2)"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)  # duplicate current stderr fd
    try:
        os.dup2(devnull, 2)  # redirect fd=2 to /dev/null
        yield
    finally:
        os.dup2(old_stderr, 2)  # restore original stderr
        os.close(devnull)
        os.close(old_stderr)


class NNModuleTest(TestRunnerBase):
    def __init__(self, test_name: str, nnmodule: TestModuleBase):
        super().__init__(test_name, nnmodule)
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "artifacts"

        # Get tags
        self.test_without_pt2: bool = is_tagged(self.nnmodule, "test_without_pt2")
        self.test_without_inference: bool = is_tagged(
            self.nnmodule, "test_without_inference"
        )
        self.with_golden: bool = is_tagged(self.nnmodule, "with_golden")

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
                with contextlib.redirect_stdout(io.StringIO()), suppress_stderr_fd():
                    with s.assertRaises(Exception) as e:
                        self._run(without_pt2=True)
                    assert self.expected_err in str(
                        e.exception
                    ), f"\nExpected the error message: {self.expected_err}\nbut the actual error message: {str(e.exception)}"

            return wrapper
        else:

            def wrapper(s):
                self._run(
                    without_pt2=self.test_without_pt2,
                    without_inference=self.test_without_inference,
                    with_golden=self.with_golden,
                )

            return wrapper

    def _run(
        self,
        without_pt2=False,
        without_inference=False,
        with_golden=False,
    ):
        dynamic_shapes = None

        assert hasattr(self.nnmodule, "get_example_inputs")
        self.forward_args, self.forward_kwargs = self.nnmodule.get_example_inputs()

        if hasattr(self.nnmodule, "get_dynamic_shapes"):
            dynamic_shapes = self.nnmodule.get_dynamic_shapes()
            if dynamic_shapes is not None:
                assert (
                    self.use_onert
                ), "Dynamic shapes are only supported with onert runtime. Please set 'use_onert' to True."

        compile_config: Optional[CompileConfigBase] = None
        if hasattr(self.nnmodule, "get_compile_config"):
            get_compile_config = getattr(self.nnmodule, "get_compile_config")
            compile_config = get_compile_config()

        test_prefix = self.test_dir / self.test_name.replace(
            "test.modules.", ""
        ).replace(".", "/")

        os.makedirs(os.path.dirname(test_prefix), exist_ok=True)

        circle_model_path = str(test_prefix) + ".circle"
        opt_circle_model_path = str(test_prefix) + ".opt.circle"
        pt2_model_path = str(test_prefix) + ".pt2"

        # Let's infer torch model before `export`
        # WHY?
        #   Some model changes its state during export (e.g., EfficientFormerL1)
        #   See https://github.com/pytorch/pytorch/issues/155114
        torch_result = infer_nnmodule(
            self.nnmodule,
            forward_args=deepcopy(self.forward_args),
            forward_kwargs=deepcopy(self.forward_kwargs),
        )

        if without_pt2:
            # torch.nn.Module --> ExportedProgram --> pt2 ----- (ExportedProgram) ------- > circle
            #                                       (--> load_from_pt2_file -->)
            convert_nnmodule_to_circle(
                self.nnmodule,
                forward_args=deepcopy(self.forward_args),
                forward_kwargs=deepcopy(self.forward_kwargs),
                circle_model_path=circle_model_path,
                dynamic_shapes=dynamic_shapes,
                config=compile_config,
            )
        else:
            # torch.nn.Module --> ExportedProgram ----------------------------------------> circle
            convert_nnmodule_to_pt2(
                self.nnmodule,
                forward_args=deepcopy(self.forward_args),
                forward_kwargs=deepcopy(self.forward_kwargs),
                pt2_model_path=pt2_model_path,
                dynamic_shapes=dynamic_shapes,
            )
            convert_pt2_to_circle(
                pt2_model_path=pt2_model_path,
                circle_model_path=circle_model_path,
                config=compile_config,
            )

        verify_circle(circle_model_path, opt_circle_model_path)

        if dynamic_shapes:

            def has_symbolic_input(circle_model_path: str) -> bool:
                ispec = ModelInputSpec.load(circle_model_path)
                for idx, shape_sig in enumerate(ispec.shape_signatures):
                    if shape_sig is None:
                        continue
                    else:
                        assert any(
                            dim == -1 for dim in shape_sig
                        ), "Unexpected shape signature: {shape_sig} in {ispec.names[idx]}"
                        return True
                return False

            if not has_symbolic_input(circle_model_path):
                raise RuntimeError(
                    f"Dynamic shapes were not applied to {circle_model_path} but expected. Check your dynamic shapes."
                )

        if without_inference:
            return

        USE_ONERT = os.environ.get("CCEX_RUNTIME") == "onert"
        if self.use_onert or USE_ONERT:
            circle_result = infer_circle(
                circle_model_path,
                forward_args=deepcopy(self.forward_args),
                forward_kwargs=deepcopy(self.forward_kwargs),
                runtime="onert",
            )
            torch_shape = torch_result[0].shape
            circle_result[0] = circle_result[0].reshape(torch_shape)
        else:
            circle_result = infer_circle(
                circle_model_path,
                forward_args=deepcopy(self.forward_args),
                forward_kwargs=deepcopy(self.forward_kwargs),
                runtime="circle-interpreter",
            )
        if with_golden:
            assert hasattr(self.nnmodule, "get_golden_outputs")

            get_golden_outputs = self.nnmodule.get_golden_outputs  # type: ignore[operator]
            validate_result(get_golden_outputs(), circle_result, **self.tolerance)  # type: ignore[operator]
        else:
            # trim None outputs
            torch_result = [res for res in torch_result if res is not None]
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
            base_name = f"{submodule}.{nnmodule_cls.__name__}"
            module_instance = nnmodule_cls()

            testdict[base_name] = NNModuleTest(base_name, module_instance).make()

        return testdict
