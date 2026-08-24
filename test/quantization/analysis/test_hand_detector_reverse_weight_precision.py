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

"""Tests for reverse FP-to-W8 greedy and beam diagnostics."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from typing import cast

from examples.hand_detector._support.reverse_weight_precision import (
    _build_independent_report,
    _enrich_search_result,
    _select_beam_groups,
)
from tico.quantization.analysis.reverse_precision_search import (
    ReversePrecisionGroup,
    run_reverse_beam,
    run_reverse_greedy,
)


def _outputs(reg: float, cls: float):
    return {
        "regressors": {"mae": reg},
        "classifiers": {"mae": cls},
    }


class ReversePrecisionSearchTest(unittest.TestCase):
    def test_greedy_quantizes_lowest_cost_groups_until_target_boundary(self) -> None:
        groups = (
            ReversePrecisionGroup("a", 10),
            ReversePrecisionGroup("b", 20),
            ReversePrecisionGroup("c", 30),
        )
        table = {
            frozenset(): _outputs(0.01, 0.02),
            frozenset({"a"}): _outputs(0.04, 0.03),
            frozenset({"b"}): _outputs(0.02, 0.03),
            frozenset({"c"}): _outputs(0.08, 0.04),
            frozenset({"a", "b"}): _outputs(0.06, 0.04),
            frozenset({"b", "c"}): _outputs(0.11, 0.05),
            frozenset({"a", "c"}): _outputs(0.12, 0.05),
            frozenset({"a", "b", "c"}): _outputs(0.13, 0.06),
        }
        result = run_reverse_greedy(
            groups,
            table[frozenset()],
            table.__getitem__,
            primary_output="regressors",
            auxiliary_output="classifiers",
            target_primary=0.1,
            target_auxiliary=0.1,
        )
        self.assertEqual(
            [step.added_group for step in result.steps],
            ["b", "a"],
        )
        self.assertEqual(result.remaining_groups, ("c",))
        self.assertEqual(result.stop_reason, "no_target_feasible_transition")
        self.assertEqual(result.final.quantized_element_count, 30)

    def test_parameter_efficiency_prefers_large_low_cost_group(self) -> None:
        groups = (
            ReversePrecisionGroup("small", 10),
            ReversePrecisionGroup("large", 1_000),
        )
        table = {
            frozenset(): _outputs(0.01, 0.01),
            frozenset({"small"}): _outputs(0.011, 0.011),
            frozenset({"large"}): _outputs(0.03, 0.02),
        }
        result = run_reverse_greedy(
            groups,
            table[frozenset()],
            table.__getitem__,
            primary_output="regressors",
            auxiliary_output="classifiers",
            target_primary=0.1,
            target_auxiliary=0.1,
            max_steps=1,
            selection_objective="parameter-efficiency",
        )
        self.assertEqual(result.steps[0].added_group, "large")

    def test_beam_can_cross_temporary_violation_and_recover(self) -> None:
        groups = (
            ReversePrecisionGroup("a", 100),
            ReversePrecisionGroup("b", 100),
            ReversePrecisionGroup("c", 5),
        )
        table = {
            frozenset(): _outputs(0.01, 0.01),
            frozenset({"a"}): _outputs(0.11, 0.05),
            frozenset({"b"}): _outputs(0.12, 0.05),
            frozenset({"c"}): _outputs(0.02, 0.02),
            frozenset({"a", "b"}): _outputs(0.05, 0.04),
            frozenset({"a", "c"}): _outputs(0.12, 0.05),
            frozenset({"b", "c"}): _outputs(0.13, 0.05),
            frozenset({"a", "b", "c"}): _outputs(0.06, 0.05),
        }
        result = run_reverse_beam(
            groups,
            table[frozenset()],
            table.__getitem__,
            primary_output="regressors",
            auxiliary_output="classifiers",
            target_primary=0.1,
            target_auxiliary=0.1,
            search_primary_ceiling=0.2,
            search_auxiliary_ceiling=0.2,
            beam_width=3,
            exploration_slots=2,
        )
        self.assertEqual(
            frozenset(result.best.selected_groups),
            frozenset({"a", "b", "c"}),
        )
        self.assertEqual(result.best.quantized_element_count, 205)

    def test_beam_prefers_more_quantized_parameters_within_target(self) -> None:
        groups = (
            ReversePrecisionGroup("small", 10),
            ReversePrecisionGroup("large", 100),
        )
        table = {
            frozenset(): _outputs(0.01, 0.01),
            frozenset({"small"}): _outputs(0.02, 0.02),
            frozenset({"large"}): _outputs(0.09, 0.02),
            frozenset({"small", "large"}): _outputs(0.12, 0.03),
        }
        result = run_reverse_beam(
            groups,
            table[frozenset()],
            table.__getitem__,
            primary_output="regressors",
            auxiliary_output="classifiers",
            target_primary=0.1,
            target_auxiliary=0.1,
            search_primary_ceiling=0.2,
            search_auxiliary_ceiling=0.2,
            beam_width=2,
            exploration_slots=1,
        )
        self.assertEqual(result.best.selected_groups, ("large",))
        self.assertEqual(result.best.quantized_element_count, 100)

    def test_duplicate_group_names_are_rejected(self) -> None:
        groups = (
            ReversePrecisionGroup("same", 10),
            ReversePrecisionGroup("same", 20),
        )
        with self.assertRaises(ValueError):
            run_reverse_greedy(
                groups,
                _outputs(0.01, 0.01),
                lambda _selected: _outputs(0.02, 0.02),
                primary_output="regressors",
                auxiliary_output="classifiers",
                target_primary=0.1,
                target_auxiliary=0.1,
            )


class HandDetectorReverseReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = (
            SimpleNamespace(
                name="a",
                parameter_element_count=10,
                site_paths=("a.weight",),
                to_dict=lambda: {
                    "group": "a",
                    "parameter_element_count": 10,
                },
            ),
            SimpleNamespace(
                name="b",
                parameter_element_count=20,
                site_paths=("b.weight",),
                to_dict=lambda: {
                    "group": "b",
                    "parameter_element_count": 20,
                },
            ),
        )

    def test_independent_report_sorts_by_reverse_cost(self) -> None:
        class Evaluator:
            def evaluate(self, selected):
                if selected == frozenset({"a"}):
                    return _outputs(0.03, 0.02)
                return _outputs(0.02, 0.02)

        report = _build_independent_report(
            self.groups,  # type: ignore[arg-type]
            _outputs(0.01, 0.01),
            Evaluator(),  # type: ignore[arg-type]
            target_regressor_mae=0.1,
            target_classifier_mae=0.1,
            full_parameter_element_count=30,
        )
        self.assertEqual([row["group"] for row in report], ["b", "a"])
        self.assertTrue(all(row["target_feasible"] for row in report))

    def test_enrichment_reports_remaining_must_optimize_groups(self) -> None:
        payload: dict[str, object] = {
            "final": {
                "selected_groups": ["a"],
                "quantized_element_count": 10,
                "outputs": _outputs(0.05, 0.04),
            },
            "steps": [],
        }
        result = _enrich_search_result(
            payload,
            self.groups,  # type: ignore[arg-type]
            full_parameter_element_count=30,
        )
        final = cast(dict, result["final"])
        self.assertEqual(final["remaining_float_groups"], ["b"])
        self.assertAlmostEqual(final["quantized_parameter_ratio"], 1 / 3)

    def test_beam_candidate_count_uses_lowest_reverse_cost_rows(self) -> None:
        independent = (
            {"group": "b"},
            {"group": "a"},
        )
        selected = _select_beam_groups(
            self.groups,  # type: ignore[arg-type]
            independent,
            1,
        )
        self.assertEqual([group.name for group in selected], ["b"])


if __name__ == "__main__":
    unittest.main()
