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

"""Timing, inventory inspection and optional sharding after unittest selection."""

import time

# Start before importing unittest or any discovered modules. Process startup and
# imports preceding this worker remain part of separately measured process wall time.
_COLLECTION_STARTED = time.perf_counter()

import argparse
import contextlib
import json
import os
import sys
import unittest

_NATIVE_DURATIONS = hasattr(unittest.TestResult, "addDuration")


class _DurationResult(unittest.TextTestResult):
    def addDuration(self, test, elapsed):
        # The native runner otherwise stores str(test), rather than the full ID.
        self.collectedDurations.append((test.id(), elapsed))


class _LegacyDurationResult(unittest.TextTestResult):
    """Python 3.10/3.11 counterpart of native per-test duration collection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collectedDurations: list[tuple[str, float]] = []
        self._test_started: float | None = None

    def startTest(self, test):
        super().startTest(test)
        method = getattr(test, test._testMethodName)
        skipped = getattr(test.__class__, "__unittest_skip__", False) or getattr(
            method, "__unittest_skip__", False
        )
        self._test_started = None if skipped else time.perf_counter()

    def stopTest(self, test):
        if self._test_started is not None:
            self.collectedDurations.append(
                (test.id(), time.perf_counter() - self._test_started)
            )
        super().stopTest(test)

    def addDuration(self, test, elapsed):
        # Allows exercising this compatibility result on Python >= 3.12 too.
        pass


class _DurationRunner(unittest.TextTestRunner):
    def __init__(self, *args, durations=10, **kwargs):
        kwargs["resultclass"] = (
            _DurationResult if _NATIVE_DURATIONS else _LegacyDurationResult
        )
        if _NATIVE_DURATIONS:
            kwargs["durations"] = durations
        super().__init__(*args, **kwargs)
        self._duration_count = durations

    def run(self, test):
        result = super().run(test)
        if not _NATIVE_DURATIONS and result.collectedDurations:
            durations = sorted(
                result.collectedDurations, key=lambda item: item[1], reverse=True
            )
            if self._duration_count:
                durations = durations[: self._duration_count]
            self.stream.writeln("Slowest test durations")
            self.stream.writeln(result.separator2)
            for test_id, elapsed in durations:
                self.stream.writeln(f"{elapsed:.3f}s     {test_id}")
            self.stream.writeln()
        return result


def iter_tests(suite):
    """Yield discovered objects in execution order, retaining repeated occurrences."""
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from iter_tests(test)
        else:
            yield test


def _group_key(test):
    cls = type(test)
    module = sys.modules.get(cls.__module__)
    # unittest's module fixtures must run in just one process. Classes otherwise
    # remain independent, including generated classes with opaque dotted names.
    if any(
        getattr(module, fixture, None) is not None
        for fixture in ("setUpModule", "tearDownModule")
    ):
        return ("module", cls.__module__)
    return ("class", cls)


def _fixture_groups(suite):
    tests = list(iter_tests(suite))
    last = {_group_key(test): index for index, test in enumerate(tests)}
    groups = []
    start = end = 0
    for index, test in enumerate(tests):
        end = max(end, last[_group_key(test)])
        if index == end:
            # A load_tests hook can return A, B, A. Keep the intervening B with
            # A so filtering cannot collapse A's two fixture lifecycles to one.
            groups.append(tests[start : end + 1])
            start = end + 1
    return groups


def partition_suite(suite, count):
    """Greedily balance class/module groups by count; never reload tests by ID.

    Largest groups go first, ties follow discovery order. The least populated
    shard wins, breaking ties by index. Retain original stream order and all
    duplicate occurrences, including fixture transitions in interleaved suites.
    """
    groups = _fixture_groups(suite)
    loads = [0] * count
    assignments = {}
    for index in sorted(range(len(groups)), key=lambda index: -len(groups[index])):
        shard = min(range(count), key=lambda shard: loads[shard])
        assignments[index] = shard
        loads[shard] += len(groups[index])
    partitions = [unittest.TestSuite() for _ in range(count)]
    for index, group in enumerate(groups):
        partitions[assignments[index]].addTests(group)
    return partitions


class _TimingProgram(unittest.TestProgram):
    testLoader: unittest.TestLoader
    test: unittest.TestSuite

    def __init__(self, *args, options, **kwargs):
        self.options = options
        super().__init__(*args, **kwargs)

    def createTests(self, *args, **kwargs):
        # Imported test modules may print; inventories must still be valid JSON.
        with (
            contextlib.redirect_stdout(sys.stderr)
            if self.options.list_tests
            else contextlib.nullcontext()
        ):
            super().createTests(*args, **kwargs)
        inspect = self.options.list_tests or self.options.shard_count is not None
        if inspect:
            # Loader failures are not ordinary groups: every shard (and listing)
            # must fail even if its partition would omit the failed import.
            if self.testLoader.errors:
                for error in self.testLoader.errors:
                    print(error, file=sys.stderr)
                raise SystemExit(1)
            total = self.test.countTestCases()
            if total == 0 and not self.testLoader.testNamePatterns:
                print("Error: full suite discovery selected no tests", file=sys.stderr)
                raise SystemExit(1)
            print(f"Selected suite: {total} tests", file=sys.stderr)
            if self.options.shard_count is not None:
                partitions = partition_suite(self.test, self.options.shard_count)
                counts = [part.countTestCases() for part in partitions]
                print(f"Shard plan (test counts): {counts}", file=sys.stderr)
                self.test = partitions[self.options.shard_index]
                groups = len(_fixture_groups(self.test))
                print(
                    f"Shard {self.options.shard_index}/{self.options.shard_count}: "
                    f"{self.test.countTestCases()} tests, {groups} fixture groups",
                    file=sys.stderr,
                )
                if self.test.countTestCases() == 0:
                    print(
                        f"Empty shard: legitimate partition of selected suite ({total} tests)",
                        file=sys.stderr,
                    )
            if total == 0:
                print("Empty selection: no tests matched the keyword", file=sys.stderr)
        self._collection_elapsed = time.perf_counter() - _COLLECTION_STARTED

    def runTests(self):
        if self.options.list_tests:
            json.dump([test.id() for test in iter_tests(self.test)], sys.stdout)
            print()
            print(
                f"Collection (worker import through discovery): {self._collection_elapsed:.3f}s",
                file=sys.stderr,
            )
            return
        started = time.perf_counter()
        empty_partition = (
            self.options.shard_count is not None and self.test.countTestCases() == 0
        )
        if empty_partition:
            # Python 3.12+ exits with NO_TESTS for an empty suite. After verified
            # discovery, an empty partition is expected and should succeed.
            self.exit = False
        try:
            super().runTests()
            if empty_partition:
                raise SystemExit(not self.result.wasSuccessful())
        finally:
            elapsed = time.perf_counter() - started
            print(
                f"Collection (worker import through discovery): {self._collection_elapsed:.3f}s",
                file=sys.stderr,
            )
            print(
                f"Suite (including class/module fixtures): {elapsed:.3f}s",
                file=sys.stderr,
            )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--durations", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--list-tests", action="store_true")
    options, arguments = parser.parse_known_args()
    if options.durations is not None and options.durations < 0:
        parser.error("--durations must be nonnegative")
    if options.shard_index is not None or options.shard_count is not None:
        if options.shard_index is None or options.shard_count is None:
            parser.error("--shard-index and --shard-count must be specified together")
        if options.shard_count <= 0:
            parser.error("--shard-count must be positive")
        if not 0 <= options.shard_index < options.shard_count:
            parser.error("--shard-index must be in [0, shard_count)")

    class ConfiguredDurationRunner(_DurationRunner):
        def __init__(self, *args, **kwargs):
            kwargs["durations"] = options.durations
            super().__init__(*args, **kwargs)

    # Match the initial import path of `python -m unittest`, including when this
    # worker is invoked via an absolute path from a checkout containing spaces.
    sys.path[0] = os.getcwd()
    _TimingProgram(
        module=None,
        argv=[sys.argv[0], *arguments],
        testRunner=(
            ConfiguredDurationRunner if options.durations is not None else None
        ),
        options=options,
    )


if __name__ == "__main__":
    main()
