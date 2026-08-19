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

"""Failure-kind tagging tests for the wrapper smoke runner."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.debug.wrapper_smoke.runner as runner


class _UnavailableCase:
    """Fake case whose environment check reports unavailability."""

    name = "fake_unavailable"

    def availability(self):
        return SimpleNamespace(available=False, reason="optional dependency missing")


class _InvalidConfigCase:
    """Fake case that rejects every configuration."""

    name = "fake_invalid_config"

    def availability(self):
        return SimpleNamespace(available=True, reason=None)

    def validate_config(self, cfg):
        raise ValueError("unknown option 'foo'")


class TestWrapperSmokeRunnerFailureKinds(unittest.TestCase):
    """Validate that runner failure paths record the expected failure kinds."""

    def test_unavailable_case_is_tagged_unavailable(self):
        """An unavailable case should fail with the UNAVAILABLE kind."""
        with patch.object(runner, "get_case", lambda name: _UnavailableCase()):
            result = runner.run_wrapper_smoke("fake_unavailable")
        self.assertFalse(result.passed)
        self.assertEqual(result.failures, ["UNAVAILABLE"])
        self.assertEqual(result.messages, ["optional dependency missing"])

    def test_invalid_config_is_tagged_config(self):
        """A config validation error should fail with the CONFIG kind."""
        stdout = io.StringIO()
        with patch.object(
            runner, "get_case", lambda name: _InvalidConfigCase()
        ), redirect_stdout(stdout):
            result = runner.run_wrapper_smoke("fake_invalid_config")
        self.assertFalse(result.passed)
        self.assertEqual(result.failures, ["CONFIG"])
        self.assertEqual(
            result.messages, ["Invalid case configuration: unknown option 'foo'"]
        )
        self.assertIn("Status           : FAIL (CONFIG)", stdout.getvalue())

    def test_strict_unavailable_raises_with_kind(self):
        """Strict mode should surface the failure kind in the exception."""
        with patch.object(runner, "get_case", lambda name: _UnavailableCase()):
            with self.assertRaisesRegex(RuntimeError, r"failed \(UNAVAILABLE\)"):
                runner.run_wrapper_smoke("fake_unavailable", strict=True)


if __name__ == "__main__":
    unittest.main()
