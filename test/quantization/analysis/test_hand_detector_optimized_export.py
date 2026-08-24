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

"""Tests for optimized hand-detector Circle export metadata."""

from __future__ import annotations

import tempfile
import unittest

from pathlib import Path
from typing import cast
from unittest import mock

from examples.hand_detector._support.optimized_export import (
    build_export_manifest,
    default_manifest_path,
    export_full_integer_circle,
    write_export_manifest,
)

from torch import nn


class OptimizedExportTest(unittest.TestCase):
    def test_manifest_distinguishes_internal_and_full_profiles(self) -> None:
        internal = {
            "regressors": {"mae": 1.5},
            "classifiers": {"mae": 0.4},
        }
        full = {
            "regressors": {"mae": 1.6},
            "classifiers": {"mae": 0.42},
        }
        manifest = build_export_manifest(
            bit_width=8,
            circle_summary={"path": "model.circle"},
            optimization_report_path="adaround.json",
            activation_report_path="activation.json",
            optimization_metadata={"steps": 2000},
            baseline_internal_full=internal,
            final_internal_full=internal,
            final_full=full,
            steps=(
                {
                    "window": {"name": "feature_block_28"},
                    "adaround": {
                        "accepted": True,
                        "best_step": 1800,
                        "weight_groups": ["w0", "w1"],
                        "weight_statistics": [],
                    },
                },
            ),
        )
        self.assertEqual(manifest["circle_export_profile"], "D:full")
        evaluation = cast(dict, manifest["evaluation"])
        self.assertEqual(evaluation["final_full"], full)
        recipe = cast(dict, manifest["recipe"])
        accepted = recipe["accepted_adaround_windows"]
        self.assertEqual(accepted[0]["name"], "feature_block_28")
        self.assertEqual(accepted[0]["best_step"], 1800)

    def test_export_writes_and_verifies_circle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.circle"

            def fake_export(_model, path):
                destination = Path(path)
                destination.write_bytes(b"0000CIR0optimized")
                return destination

            with (
                mock.patch(
                    "examples.hand_detector._support.optimized_export."
                    "export_quantized_circle",
                    side_effect=fake_export,
                ),
                mock.patch(
                    "examples.hand_detector._support.optimized_export."
                    "verify_quantized_circle",
                    return_value={"verified": True},
                ),
                mock.patch(
                    "examples.hand_detector._support.optimized_export."
                    "verify_circle_layout",
                    return_value={"transpose_count": 0},
                ),
            ):
                summary = export_full_integer_circle(
                    nn.Identity(),
                    output,
                    bit_width=8,
                )
            self.assertEqual(summary["path"], str(output))
            self.assertFalse(summary["verification_skipped"])
            self.assertEqual(
                summary["quantization_verification"],
                {"verified": True},
            )

    def test_manifest_sidecar_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            circle = Path(directory) / "model.circle"
            manifest_path = default_manifest_path(circle)
            self.assertEqual(manifest_path.name, "model.circle.manifest.json")
            output = write_export_manifest(manifest_path, {"path": circle})
            self.assertTrue(output.is_file())
            self.assertIn(str(circle), output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
