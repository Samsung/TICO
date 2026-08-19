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

"""Tests for WrapperSmokeResult failure tagging and summary formatting."""

import unittest

from tico.quantization.recipes.debug.wrapper_smoke import WrapperSmokeResult


class TestWrapperSmokeResultFailures(unittest.TestCase):
    """Validate failure kind tags in status, dict, and exception output."""

    def test_pass_status_has_no_failure_kinds(self):
        """A passing result should render a bare PASS status."""
        result = WrapperSmokeResult(case="nn_linear", passed=True)
        self.assertEqual(result.status_text(), "PASS")
        self.assertEqual(result.failures, [])
        self.assertIn("│ Status           : PASS", result.format_text())

    def test_add_failure_marks_failed_and_records_kind(self):
        """add_failure should flip passed and record the kind and message."""
        result = WrapperSmokeResult(case="nn_linear", passed=True)
        result.add_failure("EXPORT", "Circle export failed: boom")
        self.assertFalse(result.passed)
        self.assertEqual(result.failures, ["EXPORT"])
        self.assertEqual(result.messages, ["Circle export failed: boom"])
        self.assertEqual(result.status_text(), "FAIL (EXPORT)")
        self.assertIn("│ Status           : FAIL (EXPORT)", result.format_text())

    def test_multiple_kinds_render_in_first_reported_order(self):
        """Distinct kinds should appear once each, in first-reported order."""
        result = WrapperSmokeResult(case="nn_linear", passed=True)
        result.add_failure("ACCURACY", "mean_abs_diff 0.2 exceeds 0.1")
        result.add_failure("ACCURACY", "PEIR 0.9 exceeds 0.5")
        result.add_failure("EXPORT", "unsupported export artifact: onnx")
        self.assertEqual(result.failures, ["ACCURACY", "EXPORT"])
        self.assertEqual(result.status_text(), "FAIL (ACCURACY, EXPORT)")
        self.assertEqual(len(result.messages), 3)

    def test_fail_without_kind_renders_bare_fail(self):
        """Legacy direct passed=False results should still render FAIL."""
        result = WrapperSmokeResult(case="nn_linear", passed=False)
        self.assertEqual(result.status_text(), "FAIL")
        self.assertIn("│ Status           : FAIL", result.format_text())

    def test_to_dict_includes_failures(self):
        """JSON serialization should expose the failure kind list."""
        result = WrapperSmokeResult(case="nn_linear", passed=True)
        result.add_failure("SHAPE", "output shape mismatch")
        payload = result.to_dict()
        self.assertEqual(payload["failures"], ["SHAPE"])
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["messages"], ["output shape mismatch"])

    def test_raise_if_failed_includes_kinds_and_details(self):
        """The strict-mode exception should name the failure kinds."""
        result = WrapperSmokeResult(case="nn_linear", passed=True)
        result.add_failure("NON-FINITE", "quantized output has non-finite values")
        with self.assertRaisesRegex(
            RuntimeError, r"failed \(NON-FINITE\): quantized output"
        ):
            result.raise_if_failed()

    def test_raise_if_failed_without_kind_reports_unknown(self):
        """A failure without a recorded kind should raise with 'unknown'."""
        result = WrapperSmokeResult(case="nn_linear", passed=False)
        with self.assertRaisesRegex(RuntimeError, r"failed \(unknown\): no details"):
            result.raise_if_failed()


if __name__ == "__main__":
    unittest.main()
