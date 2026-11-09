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

import numpy as np
import torch

from test.pt2_to_circle_test.test_pt2_to_circle import validate_result


class TestResultValidation(unittest.TestCase):
    def test_integer_tensor_off_by_one_is_not_relative_error(self):
        with self.assertRaises(AssertionError):
            validate_result(
                [torch.tensor([100000], dtype=torch.int64)],
                [np.array([100001], dtype=np.int64)],
            )

    def test_discrete_tensors_ignore_float_tolerances(self):
        for expected, actual in [
            (torch.tensor([2**53 + 1]), np.array([2**53 + 2], dtype=np.int64)),
            (torch.tensor([True]), np.array([False])),
        ]:
            with self.subTest(dtype=expected.dtype), self.assertRaises(AssertionError):
                validate_result([expected], [actual], rtol=100, atol=100)
        validate_result([torch.tensor([True])], [np.array([True])])

    def test_float_scalar_cannot_become_discrete(self):
        # In particular, uint8 must not truncate 1.5 into an accepted 1.
        for actual in [
            np.array([1], dtype=np.uint8),
            np.array(1, dtype=np.int64),
            np.array(True),
        ]:
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                validate_result([1.5], [actual])

    def test_large_integer_scalar_does_not_round_through_float(self):
        expected = 2**53 + 1
        validate_result([expected], [np.array(expected, dtype=np.int64)])
        with self.assertRaises(AssertionError):
            validate_result([expected], [np.array([expected + 1], dtype=np.int64)])
        # Also avoid overflow/promotion when the output uses uint64.
        validate_result([2**64 - 1], [np.array([2**64 - 1], dtype=np.uint64)])
        with self.assertRaises(AssertionError):
            validate_result([2**64 - 1], [np.array([2**64 - 2], dtype=np.uint64)])

    def test_integer_scalar_is_not_cast_to_narrow_or_float_output(self):
        for expected, actual in [
            (257, np.array([1], dtype=np.uint8)),
            (-1, np.array([255], dtype=np.uint8)),
            (2**53 + 1, np.array([2**53], dtype=np.float64)),
        ]:
            with self.subTest(expected=expected), self.assertRaises(AssertionError):
                validate_result([expected], [actual], rtol=100, atol=100)

    def test_bool_scalar_has_its_own_kind_and_exact_value(self):
        validate_result([True], [np.array([True])])
        for actual in [np.array([False]), np.array([1], dtype=np.int32)]:
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                validate_result([True], [actual], rtol=100, atol=100)
        with self.assertRaises(AssertionError):
            validate_result([1], [np.array([True])])

    def test_output_count_includes_extra_outputs(self):
        with self.assertRaisesRegex(AssertionError, "Number of outputs"):
            validate_result([1], [np.array(1), np.array(2)])
        with self.assertRaisesRegex(AssertionError, "Number of outputs"):
            validate_result([1], [])

    def test_tensor_shape_and_dtype_cannot_change(self):
        with self.assertRaisesRegex(AssertionError, "Shape mismatches"):
            validate_result([torch.tensor([1.0])], [np.array(1.0, dtype=np.float32)])
        with self.assertRaisesRegex(AssertionError, "dtype"):
            validate_result([torch.tensor([1.0])], [np.array([1.0], dtype=np.float64)])

    def test_float_tensor_uses_callers_tolerance(self):
        expected = torch.tensor([1.0])
        actual = np.array([1.01], dtype=np.float32)
        validate_result([expected], [actual], rtol=0, atol=0.02)
        with self.assertRaises(AssertionError):
            validate_result([expected], [actual], rtol=0, atol=0.001)

    def test_scalar_and_single_element_output_representations(self):
        for actual in [np.array(3, dtype=np.int32), np.array([3], dtype=np.uint8)]:
            validate_result([3], [actual])
        # float32 rounding is legitimate; exact float64 representability is not
        # required of Python floats serialized as float32 constants.
        for actual in [
            np.array(0.1, dtype=np.float32),
            np.array([[0.1]], dtype=np.float32),
        ]:
            validate_result([0.1], [actual])
        with self.assertRaises(AssertionError):
            validate_result([3], [np.array([3, 3])])

    def test_matching_nan_and_signed_infinities_are_preserved(self):
        values = [float("nan"), float("inf"), -float("inf")]
        validate_result([torch.tensor(values)], [np.array(values, dtype=np.float32)])
        for value in values:
            validate_result([value], [np.array([value], dtype=np.float32)])

    def test_nonmatching_nonfinite_values_are_rejected(self):
        for expected, actual in [
            (float("inf"), -float("inf")),
            (1.0, float("nan")),
            (float("nan"), 1.0),
            (1e300, float("inf")),
        ]:
            with self.subTest(expected=expected, actual=actual):
                with self.assertRaises(AssertionError):
                    validate_result([expected], [np.array([actual], dtype=np.float32)])
                with self.assertRaises(AssertionError):
                    validate_result(
                        [torch.tensor([expected], dtype=torch.float64)],
                        [np.array([actual], dtype=np.float64)],
                    )
