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

import unittest
from unittest.mock import Mock

import torch
from tico.utils.passes import PassBase

from test.support.pass_value_test import MultiPassValueTest, SinglePassValueTest


class TestPassValueComparison(unittest.TestCase):
    def _compare(self, helper_type, expected, actual):
        # Stub only graph execution: exercise each helper's real comparison path.
        helper = helper_type()
        helper.initialized = True
        helper.forward_args, helper.forward_kwargs = (), {}
        helper.ep = Mock()
        helper.ep.module.side_effect = [lambda: expected, lambda: actual]
        test_pass = Mock(spec=PassBase)
        if helper_type is SinglePassValueTest:
            helper.run_value_test(test_pass)
        else:
            helper.run_value_test([test_pass])
        test_pass.call.assert_called_once_with(helper.ep)

    def test_integer_difference_is_not_hidden_by_relative_tolerance(self):
        for helper in (SinglePassValueTest, MultiPassValueTest):
            with self.subTest(helper=helper.__name__):
                with self.assertRaises(AssertionError):
                    self._compare(
                        helper, torch.tensor([100000]), torch.tensor([100001])
                    )

    def test_boolean_difference_fails(self):
        for helper in (SinglePassValueTest, MultiPassValueTest):
            with self.subTest(helper=helper.__name__):
                with self.assertRaises(AssertionError):
                    self._compare(helper, torch.tensor([True]), torch.tensor([False]))

    def test_identical_integer_and_boolean_values_pass(self):
        for helper in (SinglePassValueTest, MultiPassValueTest):
            for expected in (torch.tensor([2**53 + 1]), torch.tensor([False, True])):
                with self.subTest(helper=helper.__name__, dtype=expected.dtype):
                    self._compare(helper, expected, expected.clone())

    def test_shape_and_dtype_contracts_are_explicit(self):
        expected = torch.tensor([1], dtype=torch.int64)
        for helper in (SinglePassValueTest, MultiPassValueTest):
            for actual, diagnostic in (
                (expected.reshape(1, 1), "Shape"),
                (expected.to(torch.int32), "Dtype"),
            ):
                with self.subTest(helper=helper.__name__, diagnostic=diagnostic):
                    with self.assertRaisesRegex(AssertionError, diagnostic):
                        self._compare(helper, expected, actual)

    def test_existing_float_relative_and_absolute_tolerances_are_preserved(self):
        for helper in (SinglePassValueTest, MultiPassValueTest):
            with self.subTest(helper=helper.__name__):
                self._compare(
                    helper,
                    torch.tensor([1.0, 0.0]),
                    torch.tensor([1.000005, 0.0000005]),
                )
                with self.assertRaises(AssertionError):
                    self._compare(helper, torch.tensor([1.0]), torch.tensor([1.0001]))

    def test_existing_nonfinite_policy_is_preserved(self):
        for helper in (SinglePassValueTest, MultiPassValueTest):
            with self.subTest(helper=helper.__name__):
                self._compare(
                    helper,
                    torch.tensor([float("inf"), -float("inf")]),
                    torch.tensor([float("inf"), -float("inf")]),
                )
                for expected, actual in (
                    (float("nan"), float("nan")),
                    (float("inf"), -float("inf")),
                ):
                    with self.assertRaises(AssertionError):
                        self._compare(
                            helper, torch.tensor([expected]), torch.tensor([actual])
                        )
