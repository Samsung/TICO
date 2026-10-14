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

import sys
import unittest

from pathlib import Path


def _test_ids(suite: unittest.TestSuite) -> list[str]:
    result = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            result.extend(_test_ids(test))
        else:
            assert isinstance(test, unittest.TestCase)
            result.append(test.id())
    return result


class CircleAnalysisDiscoveryTest(unittest.TestCase):
    def test_analysis_tests_are_discovered(self) -> None:
        """Keep analysis tests reachable from the normal parent discovery path."""

        circle_dir = Path(__file__).resolve().parent
        repo_root = circle_dir.parents[2]
        original_path = sys.path.copy()
        try:
            for name in ("test_purity", "test_tensor_contract"):
                with self.subTest(module=name):
                    explicit_loader = unittest.TestLoader()
                    explicit = explicit_loader.loadTestsFromName(
                        f"test.unit_test.circle.analysis.{name}"
                    )
                    discovery_loader = unittest.TestLoader()
                    discovered = discovery_loader.discover(
                        start_dir=str(circle_dir),
                        pattern=f"{name}.py",
                        top_level_dir=str(repo_root),
                    )

                    self.assertEqual(explicit_loader.errors, [])
                    self.assertEqual(discovery_loader.errors, [])
                    expected_ids = sorted(_test_ids(explicit))
                    self.assertTrue(expected_ids)
                    self.assertEqual(sorted(_test_ids(discovered)), expected_ids)
        finally:
            sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
