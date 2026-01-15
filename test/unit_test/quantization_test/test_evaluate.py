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
from typing import Any

import tico
import torch
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.backend import BACKEND
from tico.quantization.evaluation.evaluate import evaluate
from tico.utils.utils import SuppressWarning

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


class TwoLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.ModuleList()
        self.m.append(torch.nn.Linear(5, 10))
        self.m.append(torch.nn.Linear(10, 20))

    def forward(self, x):
        z = self.m[0](x)
        z = self.m[1](z)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 5),), {}


class EvaluateTest(unittest.TestCase):
    # This test needs triv24-toolchain package.
    @unittest.skipIf(
        not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
    )
    def test_evaluate_simple_linear(self):
        q_m: Any = TwoLinear().eval()
        ori_m = q_m
        args, kwargs = q_m.get_example_inputs()

        prepare(q_m.m, PTQConfig())

        # 3. Calibration
        for i in range(100):
            cal_args, cal_kwargs = ori_m.get_example_inputs()
            q_m(*cal_args, **cal_kwargs)

        convert(q_m.m)

        # Export circle
        with SuppressWarning(FutureWarning, ".*LeafSpec*"):
            ep = torch.export.export(q_m, args, kwargs)
            cm = tico.convert_from_exported_program(ep)
        results = evaluate(q_m, cm, BACKEND.TRIV24, mode="return")
        assert results is not None
        self.assertTrue("peir" in results)
        self.assertEqual(len(results["peir"]), 1)
        """
        The PEIR threshold is set to 10 or lower for the test.

        This value is chosen conservatively to account for possible outliers
         caused by the randomness of the input values. In certain cases, random
        inputs may produce spikes in error, and a stricter threshold might
        result in unncessary test failures.

        Setting the threshold at 10 ensures robustness while still maintaining 
         a reasonable level of accuracy for the test.
        """
        self.assertLess(results["peir"][0], 10)
