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

"""Exercise coverage command status handling without requiring coverage in CI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CcexCoverageTest(unittest.TestCase):
    def _run_coverage(
        self, mode: str, *, test_fails: bool = False, report_status: int = 0
    ) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_root = Path(temp_dir)
            for relative_path in (
                "ccex",
                "infra/command/coverage",
                "infra/scripts/coverage.sh",
            ):
                destination = fixture_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project_root / relative_path, destination)

            test_dir = fixture_root / "test"
            test_dir.mkdir()
            (test_dir / "test_fixture.py").write_text(
                "import unittest\n\n"
                "class FixtureTest(unittest.TestCase):\n"
                "    def test_executed(self):\n"
                + (
                    "        self.fail('intentional fixture failure')\n"
                    if test_fails
                    else "        self.assertEqual(1 + 1, 2)\n"
                ),
                encoding="utf-8",
            )

            # Only replace external tools; run the real shell and unittest discovery.
            bin_dir = fixture_root / "bin"
            bin_dir.mkdir()
            tools = {
                "pip": "print('coverage 7.6.1')\n",
                "coverage": (
                    "import os\n"
                    "import subprocess\n"
                    "import sys\n"
                    "from pathlib import Path\n\n"
                    "if sys.argv[1] == 'run':\n"
                    "    sys.exit(subprocess.call([sys.executable, *sys.argv[2:]]))\n"
                    "if sys.argv[1] == 'report':\n"
                    "    print('fixture coverage report')\n"
                    "elif sys.argv[1] == 'xml':\n"
                    "    Path(sys.argv[-1]).write_text('<coverage />', encoding='utf-8')\n"
                    "else:\n"
                    "    raise AssertionError(sys.argv)\n"
                    "sys.exit(int(os.environ['FIXTURE_REPORT_STATUS']))\n"
                ),
            }
            for name, source in tools.items():
                executable = bin_dir / name
                executable.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
                executable.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                    "PYTHONPATH": str(fixture_root),
                    "FIXTURE_REPORT_STATUS": str(report_status),
                }
            )
            arguments = [] if mode == "terminal" else ["-f", mode]
            completed = subprocess.run(
                ["bash", str(fixture_root / "ccex"), "coverage", *arguments],
                cwd=fixture_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = completed.stdout + completed.stderr
            self.assertIn("Ran 1 test", output)
            if test_fails:
                self.assertIn("FAILED (failures=1)", output)
                self.assertIn("intentional fixture failure", output)
            else:
                self.assertIn("\nOK\n", output)

            if mode == "terminal":
                self.assertIn("fixture coverage report", output)
            else:
                report = test_dir / "reports" / "cov" / f"coverage.{mode}"
                expected = (
                    "<coverage />" if mode == "xml" else "fixture coverage report\n"
                )
                self.assertEqual(report.read_text(encoding="utf-8"), expected)
            return completed

    def test_successful_tests_and_reports_return_zero(self) -> None:
        for mode in ("terminal", "txt", "xml"):
            with self.subTest(mode=mode):
                completed = self._run_coverage(mode)
                self.assertEqual(completed.returncode, 0)

    def test_failed_tests_keep_reports_and_return_test_status(self) -> None:
        for mode in ("terminal", "txt", "xml"):
            with self.subTest(mode=mode):
                completed = self._run_coverage(mode, test_fails=True)
                self.assertEqual(completed.returncode, 1)

    def test_failed_report_returns_report_status(self) -> None:
        for mode in ("terminal", "txt", "xml"):
            with self.subTest(mode=mode):
                completed = self._run_coverage(mode, report_status=17)
                self.assertEqual(completed.returncode, 17)

    def test_failed_test_status_takes_precedence_over_report_failure(self) -> None:
        for mode in ("terminal", "txt", "xml"):
            with self.subTest(mode=mode):
                completed = self._run_coverage(mode, test_fails=True, report_status=17)
                self.assertEqual(completed.returncode, 1)


if __name__ == "__main__":
    unittest.main()
