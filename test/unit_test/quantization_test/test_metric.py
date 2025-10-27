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

import unittest

import torch
from tico.quantization.evaluation.metric import (
    compute_max_abs_diff,
    compute_peir,
    MetricCalculator,
    mse,
)


class TestMetricKernels(unittest.TestCase):
    def test_max_abs_diff_basic(self):
        a = torch.tensor([1.0, -2.0, 3.0])
        b = torch.tensor([1.5, -1.0, 2.0])
        expected = 1.0  # |-2.0 – (-1.0)| == 1.0
        self.assertAlmostEqual(compute_max_abs_diff(a, b), expected, places=6)

    def test_max_abs_diff_shape_mismatch(self):
        a = torch.randn(3)
        b = torch.randn(4)
        with self.assertRaises(AssertionError):
            _ = compute_max_abs_diff(a, b)

    def test_peir_basic(self):
        a = torch.tensor([0.0, 1.0, 2.0])
        b = torch.tensor([0.1, 1.5, 2.0])
        peak_err = 0.5  # max(|1.0-1.5|)
        interval = 2.0  # max(a)-min(a) == 2
        expected = peak_err / interval
        self.assertAlmostEqual(compute_peir(a, b), expected, places=6)

    def test_peir_zero_interval(self):
        a = torch.tensor([5.0, 5.0, 5.0])
        b = torch.tensor([6.0, 4.0, 5.0])
        peak_err = 1.0  # |5-6|
        expected = peak_err / 1.0
        self.assertAlmostEqual(compute_peir(a, b), expected, places=6)

    def test_peir_shape_mismatch(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 4)
        with self.assertRaises(AssertionError):
            _ = compute_peir(a, b)

    def test_mse_basic(self):
        a = torch.tensor([2.0, 2.0])
        b = torch.tensor([0.0, 0.0])
        expected_mse = (4.0 + 4.0) / 2
        self.assertAlmostEqual(mse(a, b), expected_mse, places=6)


class TestMetricCalculator(unittest.TestCase):
    def setUp(self):
        self.fp = [torch.tensor([0.0, 1.0, 2.0])]
        self.q = [torch.tensor([0.1, 1.5, 2.0])]

    def test_compute_selected_metrics(self):
        calc = MetricCalculator()
        res = calc.compute(self.fp, self.q, metrics=["diff"])
        self.assertIn("diff", res)
        self.assertNotIn("peir", res)

    def test_compute_all_metrics(self):
        calc = MetricCalculator()
        res = calc.compute(self.fp, self.q)  # metrics=None
        self.assertTrue({"diff", "peir"} <= res.keys())

    def test_unknown_metric_error(self):
        calc = MetricCalculator()
        with self.assertRaises(RuntimeError):
            _ = calc.compute(self.fp, self.q, metrics=["not_a_metric"])

    def test_custom_metric_and_duplicate_rejection(self):
        def l1_norm(x, y):
            return torch.sum(torch.abs(x - y)).item()

        # Legit custom metric
        calc = MetricCalculator(custom_metrics={"l1_sum": l1_norm})
        res = calc.compute(self.fp, self.q, metrics=["l1_sum"])
        self.assertIn("l1_sum", res)
        # Duplicate name should raise at construction time
        with self.assertRaises(RuntimeError):
            _ = MetricCalculator(custom_metrics={"diff": l1_norm})
