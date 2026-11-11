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

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
                "infra/scripts/unittest_run.py",
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

    def test_durations_preserve_selection_and_report_full_ids(self) -> None:
        for arguments in (("-k", "selected"), ("-k", "*FixtureTest.test_selected")):
            for durations in (0, 1):
                with self.subTest(arguments=arguments, durations=durations):
                    completed = self._run_tests(
                        *arguments,
                        "--durations",
                        str(durations),
                        source=(
                            "        self.assertEqual(1 + 1, 2)\n"
                            "    def test_excluded(self):\n"
                            "        self.fail('keyword filter was lost')\n"
                        ),
                    )
                    self._assert_one_test_passed(completed)
                    self.assertIn(
                        "Collection (worker import through discovery):",
                        completed.stderr,
                    )
                    self.assertIn(
                        "Suite (including class/module fixtures):", completed.stderr
                    )
                    self.assertIn("Slowest test durations", completed.stderr)
                    self.assertRegex(
                        completed.stderr,
                        r"[0-9.]+s +test_fixture.FixtureTest.test_selected",
                    )

    def test_durations_preserve_failure_import_and_no_match_status(self) -> None:
        cases = (
            ((), "        self.fail('fixture failure')\n", "FAILED (failures=1)"),
            (
                (),
                "        pass\n\nraise ImportError('fixture import')\n",
                "FAILED (errors=1)",
            ),
            (("-k", "__no_matching_test__"), "", "Ran 0 tests"),
        )
        for arguments, source, expected in cases:
            results = [
                self._run_tests(*arguments, *timing, source=source)
                for timing in ((), ("--durations", "1"))
            ]
            with self.subTest(arguments=arguments, expected=expected):
                # Older stdlib versions may succeed for an empty selection;
                # the timing option must preserve that interpreter's behavior.
                if expected != "Ran 0 tests":
                    self.assertNotEqual(results[0].returncode, 0)
                self.assertEqual(results[0].returncode, results[1].returncode)
                for completed in results:
                    self.assertIn(expected, completed.stderr)

    def test_durations_preserve_skip_expected_failure_and_subtests(self) -> None:
        completed = self._run_tests(
            "--durations",
            "1",
            source=(
                "        with self.subTest(value=1):\n"
                "            self.assertEqual(1, 1)\n"
                "    @unittest.skip('fixture skip')\n"
                "    def test_skipped(self):\n"
                "        self.fail('skip was lost')\n"
                "    @unittest.expectedFailure\n"
                "    def test_expected(self):\n"
                "        self.fail('expected failure')\n"
            ),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Ran 3 tests", completed.stderr)
        self.assertIn("OK (skipped=1, expected failures=1)", completed.stderr)
        durations = completed.stderr.split("Slowest test durations", 1)[1].split(
            "\n\n", 1
        )[0]
        self.assertEqual(durations.count("test_fixture.FixtureTest."), 1)

    def test_durations_forward_model_and_shorthand_selectors(self) -> None:
        self._assert_one_test_passed(
            self._run_tests(
                "--durations",
                "1",
                "-m",
                "Small Model*",
                test_file="pt2_to_circle_test/test_model.py",
                source="        self.assertEqual(os.environ['CCEX_TEST_MODEL'], 'Small Model*')\n",
            )
        )
        for keyword in ("op", "net"):
            self._assert_one_test_passed(
                self._run_tests(
                    "--durations",
                    "1",
                    "-k",
                    keyword,
                    source=(
                        "        self.assertEqual(1 + 1, 2)\n\n"
                        f"FixtureTest.__module__ = 'test.modules.{keyword}'\n"
                    ),
                )
            )

    def test_duration_compatibility_result_preserves_failfast_and_outcomes(
        self,
    ) -> None:
        from infra.scripts import unittest_run

        class Fixture(unittest.TestCase):
            def test_success(self):
                with self.subTest(value=1):
                    self.assertEqual(1, 1)

            @unittest.skip("fixture skip")
            def test_skip(self):
                self.fail("skip was lost")

            @unittest.expectedFailure
            def test_expected(self):
                self.fail("expected failure")

            def test_failure(self):
                with self.subTest(value=2):
                    self.fail("subtest failure")

        for failfast in (False, True):
            stream = io.StringIO()
            with (
                patch.object(unittest_run, "_NATIVE_DURATIONS", False),
                patch.object(
                    unittest_run._DurationRunner,
                    "resultclass",
                    unittest_run._LegacyDurationResult,
                ),
            ):
                result = unittest_run._DurationRunner(
                    stream=stream,
                    failfast=failfast,
                    durations=0,
                ).run(
                    unittest.TestSuite(
                        Fixture(method)
                        for method in (
                            "test_success",
                            "test_skip",
                            "test_expected",
                            "test_failure",
                            "test_success",
                        )
                    )
                )
            with self.subTest(failfast=failfast):
                self.assertFalse(result.wasSuccessful())
                self.assertEqual(result.testsRun, 4 if failfast else 5)
                self.assertEqual(len(result.skipped), 1)
                self.assertEqual(len(result.expectedFailures), 1)
                self.assertEqual(len(result.failures), 1)
                self.assertEqual(len(result.collectedDurations), 3 if failfast else 4)
                self.assertEqual(
                    result.collectedDurations[0][0], Fixture("test_success").id()
                )
                self.assertIn("Slowest test durations", stream.getvalue())

    def test_duration_results_preserve_buffer_and_class_setup_errors(self) -> None:
        from infra.scripts import unittest_run

        class BufferedFixture(unittest.TestCase):
            def test_failure(self):
                print("fixture output")
                self.fail("fixture failure")

        class ClassSetupErrorFixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("fixture class setup failure")

            def test_unreachable(self):
                self.fail("class setup error was lost")

        modes = (False, True) if unittest_run._NATIVE_DURATIONS else (False,)
        for native in modes:
            with (
                self.subTest(native=native),
                patch.object(unittest_run, "_NATIVE_DURATIONS", native),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                failed = unittest_run._DurationRunner(
                    stream=io.StringIO(),
                    buffer=True,
                    durations=1,
                ).run(unittest.TestSuite([BufferedFixture("test_failure")]))
                self.assertEqual(len(failed.failures), 1)
                self.assertIn("Stdout:\nfixture output", failed.failures[0][1])
                self.assertEqual(len(failed.collectedDurations), 1)
                setup_error = unittest_run._DurationRunner(
                    stream=io.StringIO(),
                    durations=1,
                ).run(unittest.TestSuite([ClassSetupErrorFixture("test_unreachable")]))
                self.assertEqual(setup_error.testsRun, 0)
                self.assertEqual(len(setup_error.errors), 1)
                self.assertIn("fixture class setup failure", setup_error.errors[0][1])
                self.assertEqual(setup_error.collectedDurations, [])


if __name__ == "__main__":
    unittest.main()
