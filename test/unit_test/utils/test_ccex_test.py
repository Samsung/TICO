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

"""Run the real ccex test command against small, isolated unittest fixtures."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CcexTestTest(unittest.TestCase):
    def _run_tests(
        self,
        *arguments: str,
        project_name: str = "project with spaces",
        direct_runner: bool = False,
        source: str = "",
        environment: dict[str, str] | None = None,
        test_file: str = "test_fixture.py",
    ) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_root = Path(temp_dir) / project_name
            for relative_path in (
                "ccex",
                "infra/command/test",
                "infra/scripts/test_run.py",
            ):
                destination = fixture_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project_root / relative_path, destination)

            fixture_file = fixture_root / "test" / test_file
            fixture_file.parent.mkdir(parents=True, exist_ok=True)
            for directory in (fixture_file.parent, fixture_root / "test"):
                (directory / "__init__.py").touch()
            fixture_file.write_text(
                "import os\nimport unittest\n\n"
                "class FixtureTest(unittest.TestCase):\n"
                "    def test_selected(self):\n"
                + (source or "        self.assertEqual(1 + 1, 2)\n"),
                encoding="utf-8",
            )

            env = os.environ.copy()
            for name in (
                "CCEX_RUNTIME",
                "CCEX_TEST_MODEL",
                "RUN_INTERNAL_TESTS",
                "TICO_LOG",
            ):
                env.pop(name, None)
            env.update(
                {
                    "CCEX_PROJECT_PATH": str(fixture_root),
                    "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}",
                    "PYTHONPATH": str(fixture_root),
                }
            )
            env.update(environment or {})
            if direct_runner:
                command = [
                    sys.executable,
                    str(fixture_root / "infra/scripts/test_run.py"),
                ]
            else:
                command = ["bash", str(fixture_root / "ccex"), "test"]
            return subprocess.run(
                [*command, *arguments],
                cwd=fixture_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

    def _assert_one_test_passed(
        self, completed: subprocess.CompletedProcess[str]
    ) -> None:
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Ran 1 test", completed.stderr)
        self.assertIn("\nOK\n", completed.stderr)

    def test_discovery_preserves_project_path(self) -> None:
        for project_name in ("project", "project with spaces"):
            for direct_runner in (False, True):
                for arguments in ((), ("--all",)):
                    with self.subTest(
                        project_name=project_name,
                        direct_runner=direct_runner,
                        arguments=arguments,
                    ):
                        self._assert_one_test_passed(
                            self._run_tests(
                                *arguments,
                                project_name=project_name,
                                direct_runner=direct_runner,
                            )
                        )

    def test_keyword_preserves_project_path(self) -> None:
        for direct_runner in (False, True):
            with self.subTest(direct_runner=direct_runner):
                self._assert_one_test_passed(
                    self._run_tests(
                        "-k",
                        "selected",
                        direct_runner=direct_runner,
                        source=(
                            "        self.assertEqual(1 + 1, 2)\n"
                            "    def test_excluded(self):\n"
                            "        self.fail('keyword filter was lost')\n"
                        ),
                    )
                )

    def test_keyword_with_spaces_remains_one_argument(self) -> None:
        self._assert_one_test_passed(
            self._run_tests(
                "-k",
                "selected case",
                project_name="project",
                source=(
                    "        self.assertEqual(1 + 1, 2)\n\n"
                    "setattr(FixtureTest, 'test_selected case', FixtureTest.test_selected)\n"
                    "del FixtureTest.test_selected\n"
                ),
            )
        )

    def test_failed_test_returns_nonzero(self) -> None:
        completed = self._run_tests(source="        self.fail('fixture failure')\n")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Ran 1 test", completed.stderr)
        self.assertIn("FAILED (failures=1)", completed.stderr)
        self.assertIn("fixture failure", completed.stderr)

    def test_runtime_internal_and_logging_options_reach_tests(self) -> None:
        cases: list[tuple[tuple[str, ...], dict[str, str], dict[str, str]]] = [
            ((), {}, {"CCEX_RUNTIME": "reference"}),
            ((), {"CCEX_RUNTIME": "onert"}, {"CCEX_RUNTIME": "onert"}),
            (
                ("-r", "reference", "-i", "-v"),
                {"CCEX_RUNTIME": "onert"},
                {
                    "CCEX_RUNTIME": "reference",
                    "RUN_INTERNAL_TESTS": "1",
                    "TICO_LOG": "4",
                },
            ),
        ]
        for arguments, environment, expected in cases:
            with self.subTest(arguments=arguments, environment=environment):
                source = "".join(
                    f"        self.assertEqual(os.environ[{name!r}], {value!r})\n"
                    for name, value in expected.items()
                )
                self._assert_one_test_passed(
                    self._run_tests(*arguments, source=source, environment=environment)
                )

    def test_model_pattern_reaches_model_loader(self) -> None:
        self._assert_one_test_passed(
            self._run_tests(
                "-m",
                "Small Model*",
                test_file="pt2_to_circle_test/test_model.py",
                source=(
                    "        self.assertEqual(os.environ['CCEX_TEST_MODEL'], 'Small Model*')\n"
                ),
            )
        )

    def test_module_keyword_shorthands_are_preserved(self) -> None:
        for keyword in ("op", "net"):
            with self.subTest(keyword=keyword):
                completed = self._run_tests(
                    "-k",
                    keyword,
                    source=(
                        "        self.assertEqual(1 + 1, 2)\n\n"
                        f"FixtureTest.__module__ = 'test.modules.{keyword}'\n"
                    ),
                )
                self._assert_one_test_passed(completed)
                self.assertIn(
                    f"RUN unit tests with -k test.modules.{keyword}", completed.stdout
                )


if __name__ == "__main__":
    unittest.main()
