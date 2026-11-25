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

"""Exercise selection, allocation, fixtures, and failures through fresh workers."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from infra.scripts import unittest_run

_SCRIPTS = Path(__file__).resolve().parents[1]


class UnittestShardingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tico shard tests ")
        self.addCleanup(directory.cleanup)
        self.project = Path(directory.name)
        (self.project / "test").mkdir()
        self.write("test/__init__.py", "")
        (self.project / "infra/scripts").mkdir(parents=True)
        shutil.copy2(_SCRIPTS / "unittest_run.py", self.project / "infra/scripts")
        self.env = os.environ.copy()
        for name in ("RUN_INTERNAL_TESTS", "CCEX_TEST_MODEL", "CCEX_RUNTIME"):
            self.env.pop(name, None)
        self.env["CCEX_PROJECT_PATH"] = str(self.project)
        self.env["PYTHONPATH"] = str(self.project)
        self.env["EVENTS"] = str(self.project / "events.jsonl")

    def write(self, name, source):
        path = self.project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")

    def worker(self, *options, env=None):
        return subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "unittest_run.py"),
                "discover",
                "-s",
                "test",
                "-v",
                *options,
            ],
            cwd=self.project,
            env=self.env if env is None else env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

    def command(self, *options):
        return subprocess.run(
            [sys.executable, str(_SCRIPTS / "test_run.py"), *options],
            cwd=self.project,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

    def inventory(self, *options, env=None):
        result = self.worker("--list-tests", *options, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def simple_suite(self):
        self.write(
            "test/test_generated.py",
            """
            import unittest

            print("generated module imported")

            def success(self):
                pass

            # Both the class and method names contain literal dots. These IDs
            # cannot be reconstructed with loadTestsFromName.
            Alpha = type("generated.alpha", (unittest.TestCase,), {
                "test.first.with.dots": success,
                "test.second": success,
                "__module__": __name__,
            })
            Beta = type("generated.beta", (unittest.TestCase,), {
                "test.other": success,
                "__module__": __name__,
            })
            Gamma = type("generated.gamma", (unittest.TestCase,), {
                "test.last": success,
                "__module__": __name__,
            })

            def load_tests(loader, tests, pattern):
                alpha = list(loader.loadTestsFromTestCase(Alpha))
                return unittest.TestSuite([
                    unittest.TestSuite(alpha),
                    loader.loadTestsFromTestCase(Beta),
                    loader.loadTestsFromTestCase(Gamma),
                    # A repeated occurrence must remain in the inventory.
                    unittest.TestSuite(alpha[:1]),
                ])
            """,
        )

    def test_exact_once_inventory_including_duplicate_and_generated_ids(self):
        self.simple_suite()
        whole = self.inventory()
        self.assertEqual(len(whole), 5)
        self.assertEqual(len(set(whole)), 4)
        self.assertTrue(any("generated.alpha.test.first.with.dots" in x for x in whole))
        partitions = [
            self.inventory("--shard-index", str(index), "--shard-count", "4")
            for index in range(4)
        ]
        self.assertEqual(
            Counter(whole), sum((Counter(p) for p in partitions), Counter())
        )
        for test_id in set(whole):
            self.assertEqual(sum(test_id in p for p in partitions), 1)
        for partition in partitions:
            # Assignment selects existing objects and preserves discovery order.
            self.assertEqual(partition, [x for x in whole if x in partition])
        for index in range(4):
            result = self.worker("--shard-index", str(index), "--shard-count", "4")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_assignment_is_deterministic_across_fresh_processes_and_hash_seeds(self):
        self.simple_suite()
        self.write(
            "test/test_independent.py",
            """
            import unittest
            class First(unittest.TestCase):
                def test_first(self):
                    pass
            class Second(unittest.TestCase):
                def test_second(self):
                    pass
            """,
        )
        nonempty = 0
        for index in range(4):
            options = ("--shard-index", str(index), "--shard-count", "4")
            first = self.inventory(*options, env={**self.env, "PYTHONHASHSEED": "1"})
            second = self.inventory(*options, env={**self.env, "PYTHONHASHSEED": "42"})
            self.assertEqual(first, second)
            nonempty += bool(first)
        self.assertGreaterEqual(nonempty, 2)

    def test_newly_discovered_tests_are_assigned_automatically(self):
        self.simple_suite()
        before = Counter(self.inventory())
        self.write(
            "test/test_new.py",
            """
            import unittest
            class NewTest(unittest.TestCase):
                def test_new(self):
                    pass
            """,
        )
        after = Counter(self.inventory())
        self.assertEqual(sum((after - before).values()), 1)
        partitions = [
            self.inventory("--shard-index", str(index), "--shard-count", "4")
            for index in range(4)
        ]
        self.assertEqual(after, sum((Counter(p) for p in partitions), Counter()))

    def fixture_suite(self, module_fixtures):
        self.write(
            "test/test_fixtures.py",
            """
            import json
            import os
            import unittest

            def record(event):
                with open(os.environ["EVENTS"], "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\\n")

            class First(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    record([cls.__name__, "setup"])
                @classmethod
                def tearDownClass(cls):
                    record([cls.__name__, "teardown"])
                def test_a(self):
                    record([self.__class__.__name__, "a"])
                def test_b(self):
                    record([self.__class__.__name__, "b"])

            class Second(First):
                pass

            def load_tests(loader, tests, pattern):
                # Preserve this intentional non-alphabetical class/method order.
                return unittest.TestSuite([
                    Second("test_b"), Second("test_a"),
                    First("test_b"), First("test_a"),
                ])
            """,
        )
        if module_fixtures:
            with (self.project / "test/test_fixtures.py").open("a") as stream:
                stream.write(
                    "\ndef setUpModule():\n    record(['module', 'setup'])\n"
                    "\ndef tearDownModule():\n    record(['module', 'teardown'])\n"
                )

    def events(self):
        path = Path(self.env["EVENTS"])
        if not path.exists():
            return []
        events = [json.loads(line) for line in path.read_text().splitlines()]
        path.unlink()
        return events

    def test_classes_stay_together_with_fixture_and_method_order_preserved(self):
        self.fixture_suite(module_fixtures=False)
        expected = [
            [[name, "setup"], [name, "b"], [name, "a"], [name, "teardown"]]
            for name in ("First", "Second")
        ]
        observed = []
        for index in range(4):
            result = self.worker("--shard-index", str(index), "--shard-count", "4")
            self.assertEqual(result.returncode, 0, result.stderr)
            events = self.events()
            for offset in range(0, len(events), 4):
                observed.append(events[offset : offset + 4])
        self.assertCountEqual(observed, expected)

    def test_module_fixtures_group_classes_and_preserve_original_order(self):
        self.fixture_suite(module_fixtures=True)
        result = self.worker()
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = self.events()
        self.assertEqual(expected[0], ["module", "setup"])
        self.assertEqual(expected[-1], ["module", "teardown"])
        nonempty = []
        for index in range(4):
            result = self.worker("--shard-index", str(index), "--shard-count", "4")
            self.assertEqual(result.returncode, 0, result.stderr)
            events = self.events()
            if events:
                nonempty.append(events)
        self.assertEqual(nonempty, [expected])

    def assert_shards_preserve_fixture_reentry(self):
        result = self.worker()
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = self.events()
        nonempty = []
        for index in range(4):
            result = self.worker("--shard-index", str(index), "--shard-count", "4")
            self.assertEqual(result.returncode, 0, result.stderr)
            events = self.events()
            if events:
                nonempty.append(events)
        self.assertEqual(nonempty, [expected])
        return expected

    def test_interleaved_classes_preserve_fixture_reentry(self):
        self.fixture_suite(module_fixtures=False)
        with (self.project / "test/test_fixtures.py").open("a") as stream:
            stream.write(textwrap.dedent("""
                def load_tests(loader, tests, pattern):
                    return unittest.TestSuite([
                        First("test_b"), Second("test_b"), First("test_a")
                    ])
                """))
        events = self.assert_shards_preserve_fixture_reentry()
        self.assertEqual(events.count(["First", "setup"]), 2)
        self.assertEqual(events.count(["First", "teardown"]), 2)

    def test_interleaved_modules_preserve_fixture_reentry(self):
        for name in ("first", "second"):
            self.write(
                f"test/fixtures/{name}.py",
                f"""
                import json
                import os
                import unittest

                def record(event):
                    with open(os.environ["EVENTS"], "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(["{name}", event]) + "\\n")

                def setUpModule():
                    record("setup")

                def tearDownModule():
                    record("teardown")

                class Example(unittest.TestCase):
                    def test_a(self):
                        record("a")
                    def test_b(self):
                        record("b")
                """,
            )
        self.write(
            "test/fixtures/__init__.py",
            """
            import unittest
            from . import first, second

            def load_tests(loader, tests, pattern):
                return unittest.TestSuite([
                    first.Example("test_b"), second.Example("test_b"),
                    first.Example("test_a"),
                ])
            """,
        )
        events = self.assert_shards_preserve_fixture_reentry()
        self.assertEqual(events.count(["first", "setup"]), 2)
        self.assertEqual(events.count(["first", "teardown"]), 2)

    def test_invalid_arguments_and_unsupported_performance_mode(self):
        cases = (
            ("--shard-index", "0"),
            ("--shard-count", "4"),
            ("--shard-index", "0", "--shard-count", "0"),
            ("--shard-index", "0", "--shard-count", "-1"),
            ("--shard-index", "-1", "--shard-count", "4"),
            ("--shard-index", "4", "--shard-count", "4"),
            ("--shard-index", "x", "--shard-count", "4"),
            ("--shard-index", "0", "--shard-count", "4", "--perf"),
            ("--list-tests", "--perf"),
        )
        for options in cases:
            with self.subTest(options=options):
                result = self.command(*options)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("error", (result.stdout + result.stderr).lower())

    def test_discovery_import_and_load_tests_errors_fail_every_shard_and_listing(self):
        for source in (
            "raise ImportError('synthetic import failure')",
            "def load_tests(loader, tests, pattern):\n    raise RuntimeError('synthetic hook failure')",
        ):
            self.write("test/test_broken.py", source)
            for index in range(4):
                for listing in ((), ("--list-tests",)):
                    with self.subTest(source=source, index=index, listing=listing):
                        result = self.worker(
                            *listing, "--shard-index", str(index), "--shard-count", "4"
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("synthetic", result.stdout + result.stderr)
            # Avoid same-second pycache reuse between different synthetic modules.
            shutil.rmtree(self.project / "test/__pycache__", ignore_errors=True)

    def test_keyword_cannot_hide_a_discovery_failure(self):
        self.write(
            "test/test_broken.py", "raise ImportError('unfiltered discovery failure')"
        )
        result = self.worker(
            "--list-tests",
            "--shard-index",
            "3",
            "--shard-count",
            "4",
            "-k",
            "no_such_test",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unfiltered discovery failure", result.stderr)

    def test_failure_exit_status_reaches_public_command(self):
        self.write(
            "test/test_failure.py",
            """
            import unittest
            class Failure(unittest.TestCase):
                def test_fails(self):
                    self.fail("synthetic assertion")
            """,
        )
        results = [
            self.command("--shard-index", str(index), "--shard-count", "4")
            for index in range(4)
        ]
        failures = [result for result in results if result.returncode]
        self.assertEqual(len(failures), 1)
        self.assertIn("synthetic assertion", failures[0].stderr)

    def test_empty_partition_is_explicit_and_successful(self):
        self.write(
            "test/test_one.py",
            "import unittest\nclass One(unittest.TestCase):\n    def test_one(self):\n        pass\n",
        )
        empty = []
        for index in range(4):
            options = ("--shard-index", str(index), "--shard-count", "4")
            if not self.inventory(*options):
                result = self.worker(*options)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("empty", result.stderr.lower())
                empty.append(index)
        self.assertEqual(len(empty), 3)

    def test_empty_full_discovery_fails_but_empty_keyword_selection_is_reported(self):
        for options in (
            ("--shard-index", "0", "--shard-count", "4"),
            ("--list-tests",),
        ):
            result = self.worker(*options)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no tests", result.stderr.lower())
        self.simple_suite()
        result = self.command(
            "-k", "no_such_test", "--shard-index", "0", "--shard-count", "4"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("empty", result.stderr.lower())

    def test_keyword_duration_and_runtime_internal_options_are_preserved(self):
        self.write(
            "test/test_options.py",
            """
            import os
            import unittest
            class Options(unittest.TestCase):
                def test_selected(self):
                    self.assertEqual(os.environ["CCEX_RUNTIME"], "onert")
                    self.assertEqual(os.environ["RUN_INTERNAL_TESTS"], "1")
                    self.assertEqual(os.environ["TICO_LOG"], "4")
                def test_excluded(self):
                    self.fail("keyword filter was ignored")
            """,
        )
        executed = 0
        for index in range(4):
            result = self.command(
                "-k",
                "selected",
                "-i",
                "-v",
                "-r",
                "onert",
                "--durations",
                "0",
                "--shard-index",
                str(index),
                "--shard-count",
                "4",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Collection", result.stderr)
            self.assertIn("Suite (including class/module fixtures)", result.stderr)
            executed += "test_selected (" in result.stderr
        self.assertEqual(executed, 1)

    def test_listing_does_not_execute_tests_and_preserves_skipped_inventory(self):
        self.write(
            "test/test_skipped.py",
            """
            import unittest
            class Skipped(unittest.TestCase):
                @unittest.skip("intentional synthetic skip")
                def test_skipped(self):
                    self.fail("must remain skipped")
                def test_selected(self):
                    self.fail("listing must not execute tests")
            """,
        )
        result = self.command("--list-tests", "--durations", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        inventory = json.loads(result.stdout)
        self.assertEqual(len(inventory), 2)
        self.assertTrue(any(test_id.endswith("test_skipped") for test_id in inventory))
        skipped = 0
        for index in range(4):
            result = self.command(
                "-k",
                "*.test_skipped",
                "--shard-index",
                str(index),
                "--shard-count",
                "4",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            skipped += "skipped 'intentional synthetic skip'" in result.stderr
        self.assertEqual(skipped, 1)

    def test_model_opt_in_boundary_and_sharded_model_selection(self):
        self.write(
            "test/pt2_to_circle_test/__init__.py",
            """
            def load_tests(loader, tests, pattern):
                return loader.loadTestsFromName(__name__ + ".test_default")
            """,
        )
        self.write(
            "test/pt2_to_circle_test/test_default.py",
            """
            import unittest
            class DefaultCase(unittest.TestCase):
                def test_default(self):
                    pass
            """,
        )
        self.write(
            "test/pt2_to_circle_test/test_model.py",
            """
            import os
            import unittest
            class ModelCase(unittest.TestCase):
                def test_model(self):
                    self.assertEqual(os.environ["CCEX_TEST_MODEL"], "Synthetic*")
            """,
        )
        default = self.command("--list-tests")
        self.assertEqual(default.returncode, 0, default.stdout + default.stderr)
        default_inventory = json.loads(default.stdout)
        self.assertEqual(len(default_inventory), 1)
        self.assertIn("DefaultCase", default_inventory[0])
        model = self.command("--list-tests", "-m", "Synthetic*")
        self.assertEqual(model.returncode, 0, model.stdout + model.stderr)
        model_inventory = json.loads(model.stdout)
        self.assertEqual(len(model_inventory), 1)
        self.assertIn("ModelCase", model_inventory[0])
        partitions = []
        for index in range(4):
            options = (
                "-m",
                "Synthetic*",
                "--shard-index",
                str(index),
                "--shard-count",
                "4",
            )
            listed = self.command("--list-tests", *options)
            self.assertEqual(listed.returncode, 0, listed.stdout + listed.stderr)
            partitions.extend(json.loads(listed.stdout))
            executed = self.command(*options)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
        self.assertEqual(Counter(partitions), Counter(model_inventory))

    def test_legacy_duration_collection_retains_full_ids_and_excludes_skips(self):
        class Example(unittest.TestCase):
            def test_pass(self):
                pass

            @unittest.skip("intentional duration skip")
            def test_skip(self):
                self.fail("skip was ignored")

        stream = io.StringIO()
        with patch.object(unittest_run, "_NATIVE_DURATIONS", False):
            result = unittest_run._DurationRunner(stream=stream, durations=0).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(Example)
            )
        self.assertTrue(result.wasSuccessful(), stream.getvalue())
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(
            [test_id for test_id, _ in result.collectedDurations],
            [Example("test_pass").id()],
        )
        self.assertTrue(all(elapsed >= 0 for _, elapsed in result.collectedDurations))
        self.assertIn("Slowest test durations", stream.getvalue())
        self.assertIn(Example("test_pass").id(), stream.getvalue())


if __name__ == "__main__":
    unittest.main()
