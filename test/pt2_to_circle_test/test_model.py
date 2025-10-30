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

import fnmatch
import os
import unittest
from pathlib import Path
from unittest import TestSuite

from test.pt2_to_circle_test.builder import NormalTestDictBuilder

# NOTE Thie file's name must start with `test_` to be found by unittest

testdir = Path(os.path.dirname(os.path.dirname(__file__)))


def load_tests(loader, tests, pattern):
    """
    Decide how to collect tests.
    https://docs.python.org/ko/3.13/library/unittest.html#load-tests-protocol

    Load only tests which matches the pattern given by the user, among them which are defined in `model` directory.
    For exact model matching, use exact directory name instead of pattern matching.
    """
    suite = TestSuite()

    test_model = os.environ.get("CCEX_TEST_MODEL")

    assert test_model
    matches = fnmatch.filter(os.listdir(str(testdir) + "/modules/model"), test_model)

    if len(matches) == 0:
        raise Exception(
            f"No test files matching {test_model} found in {testdir}/modules/model"
        )

    for match in matches:
        builder = NormalTestDictBuilder(namespace=f"test.modules.model.{match}")
        tests = build_tests(builder)

        for test_class in tests:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
    return suite


def build_tests(builder: NormalTestDictBuilder):
    """
    NOTE: COPY THE WHOLE FUNCTION TO USE
    This function should be placed under the test-defining directory to get globals() locally.

    Lv1     | Category          | EX. test.modules.model.ResNet18
    Lv2     | Submodule (File)  | EX. test.modules.model.ResNet18.model
    LV3     | NNmodule (Class)  | EX. test.modules.model.ResNet18.model.ResNet18
    """
    tests = []
    for submodule in builder.submodules:
        testdict = builder.build(submodule)
        if len(testdict) > 0:
            # Here, a class is declared in this file grammatically
            # NOTE Replacing '.' with '_' to avoid potential namespace corruption
            tests.append(type(submodule, (unittest.TestCase,), testdict))
    return tests
