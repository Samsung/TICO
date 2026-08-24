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

"""Tests for palm-detector cumulative sensitivity reporting."""

from __future__ import annotations

import io
import unittest

from contextlib import redirect_stdout
from typing import cast

from examples.hand_detector._support.cumulative_sensitivity import (
    build_activation_sensitivity_path_report,
    print_activation_sensitivity_path,
    select_activation_sensitivity_groups,
)
from examples.hand_detector._support.sensitivity import ActivationSensitivityGroup
from tico.quantization.analysis import (
    QuantizationGroup,
    SensitivityPathResult,
    SiteSelector,
)


class HandDetectorCumulativeSensitivityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = (
            _group("stem", "stem", (0, 1), (141, 142), ("stem.out",)),
            _group(
                "feature_block_00",
                "feature",
                (2, 3, 4, 5),
                (143, 144, 145, 146),
                ("block0.in", "block0.out"),
            ),
        )

    def test_group_selection_preserves_requested_order(self) -> None:
        selected = select_activation_sensitivity_groups(
            self.groups,
            ("feature_block_00", "stem"),
        )
        self.assertEqual(
            tuple(group.name for group in selected),
            ("feature_block_00", "stem"),
        )

    def test_group_selection_rejects_unknown_and_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown activation"):
            select_activation_sensitivity_groups(self.groups, ("missing",))
        with self.assertRaisesRegex(ValueError, "repeat"):
            select_activation_sensitivity_groups(self.groups, ("stem", "stem"))

    def test_path_report_contains_cumulative_and_incremental_gains(self) -> None:
        baseline = _outputs(2.0, 0.8)
        steps = (
            SensitivityPathResult(
                step=1,
                group="stem",
                selected_groups=("stem",),
                outputs=_outputs(1.5, 0.6),
                score=1.5,
                cumulative_sensitivity=0.5,
                incremental_sensitivity=0.5,
                matched_sites=("stem.out",),
                selected_sites=("stem.out",),
            ),
            SensitivityPathResult(
                step=2,
                group="feature_block_00",
                selected_groups=("stem", "feature_block_00"),
                outputs=_outputs(1.2, 0.55),
                score=1.2,
                cumulative_sensitivity=0.8,
                incremental_sensitivity=0.3,
                matched_sites=("block0.in", "block0.out"),
                selected_sites=("stem.out", "block0.in", "block0.out"),
            ),
        )

        report = build_activation_sensitivity_path_report(
            baseline=baseline,
            results=steps,
            groups=self.groups,
        )

        self.assertEqual(report[0]["kind"], "stem")
        self.assertEqual(report[1]["kind"], "feature")
        self.assertAlmostEqual(cast(float, report[1]["regressor_mae_improvement"]), 0.8)
        self.assertAlmostEqual(
            cast(float, report[1]["incremental_regressor_mae_improvement"]),
            0.3,
        )
        self.assertAlmostEqual(
            cast(float, report[1]["incremental_classifier_mae_improvement"]),
            0.05,
        )

    def test_path_console_reports_empty_greedy_selection(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            print_activation_sensitivity_path(
                strategy="greedy",
                dtype_name="uint8",
                percentile=99.99,
                baseline=_outputs(2.0, 0.8),
                results=(),
                baseline_site_count=291,
                score_output="regressors",
            )
        text = output.getvalue()
        self.assertIn("Baseline E:internal-full", text)
        self.assertIn("No group satisfied", text)


def _group(
    name: str,
    kind: str,
    positions: tuple[int, ...],
    tensor_ids: tuple[int, ...],
    site_paths: tuple[str, ...],
) -> ActivationSensitivityGroup:
    return ActivationSensitivityGroup(
        group=QuantizationGroup(name, SiteSelector.none()),
        kind=kind,
        operation_positions=positions,
        operation_indices=positions,
        operation_names=tuple("TEST" for _ in positions),
        tensor_ids=tensor_ids,
        site_paths=site_paths,
    )


def _outputs(
    regressor_mae: float,
    classifier_mae: float,
) -> dict[str, dict[str, float]]:
    return {
        "regressors": {"mae": regressor_mae},
        "classifiers": {"mae": classifier_mae},
    }


if __name__ == "__main__":
    unittest.main()
