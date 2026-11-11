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

"""Optional timing around the standard unittest program, without changing selection."""

import time

# Start before importing unittest or any discovered modules. Process startup and
# imports preceding this worker remain part of separately measured process wall time.
_COLLECTION_STARTED = time.perf_counter()

import argparse
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


class _TimingProgram(unittest.TestProgram):
    def createTests(self, *args, **kwargs):
        super().createTests(*args, **kwargs)
        self._collection_elapsed = time.perf_counter() - _COLLECTION_STARTED

    def runTests(self):
        started = time.perf_counter()
        try:
            super().runTests()
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
    parser.add_argument("--durations", type=int, required=True)
    options, arguments = parser.parse_known_args()
    if options.durations < 0:
        parser.error("--durations must be nonnegative")

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
        testRunner=ConfiguredDurationRunner,
    )


if __name__ == "__main__":
    main()
