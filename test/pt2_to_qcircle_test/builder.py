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

import os
import unittest
from pathlib import Path

import tico
from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config import PT2EConfig
from tico.experimental.quantization.evaluation.backend import BACKEND
from tico.experimental.quantization.evaluation.evaluate import evaluate

from test.modules.base import TestModuleBase

from test.utils.base_builders import TestDictBuilderBase, TestRunnerBase

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


class QuantizationTest(TestRunnerBase):
    def __init__(
        self,
        test_name: str,
        mod: TestModuleBase,
        config: PT2EConfig,
        backend: BACKEND,
    ):
        super().__init__(test_name, mod)
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "artifacts"

        self.config = config
        self.backend = backend

        # Set tolerance for PEIR as a default value 5, with the percentage unit(%).
        self.tolerance = {"peir": 5}
        if hasattr(mod, "quantized_peir_tolerance"):
            self.tolerance["peir"] = mod.quantized_peir_tolerance  # type: ignore[assignment]

    def make(self):
        @unittest.skipIf(
            not IS_INTERNAL_TEST,
            "Internal test — run only if --include-internal is set",
        )
        def wrapper(s):
            self._run()

        return wrapper

    def _run(self):
        mod = self.nnmodule.eval()
        original_mod = mod

        cal_args_0, cal_kwargs_0 = next(original_mod.get_calibration_data())  # type: ignore[operator]
        mod = prepare(mod, self.config, args=cal_args_0, kwargs=cal_kwargs_0)

        for c_args, c_kwargs in original_mod.get_calibration_data():  # type: ignore[operator]
            mod(*c_args, **c_kwargs)

        mod = convert(mod)

        # pt2e module doesn't have `eval()` api.
        mod.training = False
        cm = tico.convert(mod, cal_args_0, cal_kwargs_0)

        test_prefix = self.test_dir / self.test_name.replace(
            "test.modules.", ""
        ).replace(".", "/")

        os.makedirs(os.path.dirname(test_prefix), exist_ok=True)

        qcircle_model_path = str(test_prefix) + ".q.circle"
        cm.save(qcircle_model_path)

        results = evaluate(original_mod, cm, backend=self.backend, mode="return")
        assert results is not None

        assert "peir" in results
        assert len(results["peir"]) == 1
        assert (
            results["peir"][0] < self.tolerance["peir"]
        ), f"PEIR exceeds tolerance. PEIR:{results['peir'][0]}%, tolerance: {self.tolerance['peir']}%"


class QuantizationTestDictBuilder(TestDictBuilderBase):
    def __init__(self, namespace: str, backend: BACKEND = BACKEND.TRIV24):
        super().__init__(namespace)
        self.backend = backend

    def build(self, submodule):
        """
        Return a dictionary of quantization tests for a submodule
        key: module name (must match the module name in the source code, e.g., test.modules..op.add.SimpleAdd)
        value: a function that runs the test for the module
        """
        testdict = {}
        for nnmodule_cls in self._get_nnmodules(submodule):
            if not hasattr(nnmodule_cls, "get_calibration_data"):
                # Only build the modules for quantization test
                continue

            nnmodule_name = f"{submodule}.{nnmodule_cls.__name__}"

            config = PT2EConfig()
            backend = self.backend
            testdict[nnmodule_name] = QuantizationTest(
                nnmodule_name, nnmodule_cls(), config, backend
            ).make()
        return testdict
