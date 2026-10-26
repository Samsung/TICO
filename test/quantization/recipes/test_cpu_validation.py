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

"""Small saved-array and CLI contracts for connected CPU replay."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tico.quantization.recipes.debug.cpu_validation import (
    _load_replay_evidence,
    _same_replay_arrays,
    main,
)


class TestReplayArrays(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.archive = Path(self.directory.name) / "arrays.npz"
        self.record = {"input_count": 1, "output_count": 1}
        self.values = {
            "input_0": np.array([1.0], dtype=np.float32),
            "torch_0": np.array([2.0], dtype=np.float32),
            "circle_0": np.array([2.0], dtype=np.float32),
        }

    def compare(self, saved=None, current=None, record=None):
        np.savez(self.archive, **(self.values if saved is None else saved))
        with np.load(self.archive, allow_pickle=False) as previous:
            return _same_replay_arrays(
                previous,
                self.values if current is None else current,
                self.record if record is None else record,
            )

    def test_complete_identical_arrays_match(self):
        self.assertTrue(self.compare())

    def test_equal_values_with_different_dtypes_never_match(self):
        for key in self.values:
            with self.subTest(key=key):
                current = dict(self.values)
                current[key] = current[key].astype(np.float64)
                self.assertFalse(self.compare(current=current))

    def test_different_shapes_and_values_never_match(self):
        for key in self.values:
            for value in (self.values[key].reshape(1, 1), self.values[key] + 1):
                with self.subTest(key=key, shape=value.shape):
                    current = dict(self.values, **{key: value})
                    self.assertFalse(self.compare(current=current))

    def test_nonfinite_current_or_saved_arrays_never_match(self):
        for key in self.values:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(key=key, value=value):
                    invalid = dict(self.values, **{key: np.array([value], np.float32)})
                    self.assertFalse(self.compare(current=invalid))
                    self.assertFalse(self.compare(saved=invalid))
                    self.assertFalse(self.compare(saved=invalid, current=invalid))

    def test_missing_or_extra_arrays_are_rejected_even_when_both_sides_match(self):
        for key in self.values:
            missing = dict(self.values)
            missing.pop(key)
            for saved, current in (
                (missing, self.values),
                (self.values, missing),
                (missing, missing),
            ):
                with self.subTest(missing=key), self.assertRaisesRegex(
                    ValueError, "array keys"
                ):
                    self.compare(saved=saved, current=current)
        extra = dict(self.values, unexpected=np.ones(1, np.float32))
        for saved, current in (
            (extra, self.values),
            (self.values, extra),
            (extra, extra),
        ):
            with self.subTest(extra=True), self.assertRaisesRegex(
                ValueError, "array keys"
            ):
                self.compare(saved=saved, current=current)

    def test_empty_or_invalid_output_counts_cannot_pass(self):
        for count in (0, -1, True, 1.5, "1", None):
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError, "output_count"
            ):
                self.compare(record={"input_count": 1, "output_count": count})
        with self.assertRaisesRegex(ValueError, "input_count"):
            self.compare(record={"input_count": -1, "output_count": 1})

    def test_zero_inputs_are_valid_when_output_evidence_is_complete(self):
        arrays = {key: value for key, value in self.values.items() if key != "input_0"}
        self.assertTrue(
            self.compare(
                saved=arrays,
                current=arrays,
                record={"input_count": 0, "output_count": 1},
            )
        )


class TestReplaySummary(unittest.TestCase):
    def test_empty_or_malformed_stage_lists_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries: tuple[dict[str, object], ...] = (
                {},
                {"stages": []},
                {"stages": {}},
                {"stages": [None]},
            )
            for summary in summaries:
                with self.subTest(summary=summary):
                    (root / "summary.json").write_text(json.dumps(summary))
                    with self.assertRaisesRegex(ValueError, "stage evidence"):
                        _load_replay_evidence(root)

    def test_zero_output_stage_is_rejected_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(
                json.dumps({"stages": [{"input_count": 1, "output_count": 0}]})
            )
            with self.assertRaisesRegex(ValueError, "output_count"):
                _load_replay_evidence(root)


class TestReplayCLI(unittest.TestCase):
    def test_replay_arguments_are_visible_in_help(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(
            SystemExit
        ) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)
        for argument in (
            "replay",
            "--repo",
            "--checkpoint",
            "--evidence",
            "--artifacts",
        ):
            self.assertIn(argument, output.getvalue())

    def test_missing_replay_arguments_are_reported_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(
                SystemExit
            ) as result:
                main(["--stage", "replay", "--output", str(output)])
            self.assertEqual(result.exception.code, 2)
            for argument in ("--repo", "--checkpoint", "--evidence", "--artifacts"):
                self.assertIn(argument, error.getvalue())
            self.assertFalse(output.exists())

    def test_public_replay_dispatch_requires_no_unused_capacity_argument(self):
        """UNIT_MOCK: validate CLI dispatch without replacing a model or runtime."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "trusted.pt"
            checkpoint.touch()
            argv = ["--stage", "replay", "--output", str(root / "new")]
            for name, value in (
                ("repo", root),
                ("checkpoint", checkpoint),
                ("evidence", root),
                ("artifacts", root),
            ):
                argv += ["--" + name, str(value)]
            with patch(
                "tico.quantization.recipes.debug.cpu_validation._replay_connected",
                return_value=0,
            ) as replay:
                self.assertEqual(main(argv), 0)
            self.assertEqual(replay.call_args.args[0].evidence, str(root))


if __name__ == "__main__":
    unittest.main()
