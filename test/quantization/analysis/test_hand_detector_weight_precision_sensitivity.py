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

"""Tests for W8/A16 hand-detector parameter sensitivity helpers."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from typing import cast
from unittest import mock

from examples.hand_detector._support import weight_precision_sensitivity as module
from examples.hand_detector._support.weight_precision_sensitivity import (
    _run_constrained_greedy,
    build_independent_report,
    build_path_report,
    build_weight_sensitivity_groups,
    parameter_totals,
    select_weight_sensitivity_groups,
)
from tico.quantization.analysis import SensitivityPathResult, SensitivityResult
from tico.quantization.wrapq.control import SiteRole

from torch import nn


def _site(path: str, layer: int, wrapped: nn.Module):
    return SimpleNamespace(
        path=path,
        module_path=f"runtime.layers.{layer}",
        observer_name="weight",
        role=SiteRole.PARAMETER,
        module=SimpleNamespace(
            fp_name=f"detector.layers.{layer}",
            module=wrapped,
        ),
    )


class HandDetectorWeightPrecisionSensitivityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = SimpleNamespace(
            operations=(
                {"index": 2, "name": "CONV_2D"},
                {"index": 4, "name": "PRELU"},
                {"index": 7, "name": "DEPTHWISE_CONV_2D"},
            )
        )
        self.operation_groups = (
            SimpleNamespace(name="stem", kind="stem", positions=(0, 1)),
            SimpleNamespace(
                name="feature_block_00",
                kind="feature",
                positions=(2,),
            ),
        )
        regular = nn.Conv2d(3, 4, 1, bias=False)
        prelu = nn.PReLU(4)
        depthwise = nn.Conv2d(4, 4, 3, groups=4, bias=False)
        self.sites = (
            _site("runtime.layers.0.weight", 0, regular),
            _site("runtime.layers.1.weight", 1, prelu),
            _site("runtime.layers.2.weight", 2, depthwise),
        )

    def _patch_graph(self):
        return (
            mock.patch.object(module, "_find_detector", return_value=self.detector),
            mock.patch.object(
                module,
                "_partition_operations",
                return_value=self.operation_groups,
            ),
            mock.patch.object(
                module,
                "iter_quantization_sites",
                return_value=self.sites,
            ),
        )

    def test_semantic_groups_cover_conv_depthwise_and_prelu(self) -> None:
        first, second, third = self._patch_graph()
        with first, second, third:
            groups = build_weight_sensitivity_groups(object())
        self.assertEqual([group.name for group in groups], ["stem", "feature_block_00"])
        self.assertEqual(groups[0].site_count, 2)
        self.assertEqual(groups[0].parameter_element_count, 16)
        self.assertEqual(
            {value.kind for value in groups[0].parameter_breakdown},
            {"conv2d_weight", "prelu_slope"},
        )
        self.assertEqual(
            groups[1].parameter_breakdown[0].kind,
            "depthwise_conv2d_weight",
        )
        self.assertEqual(parameter_totals(groups), (3, 52))

    def test_site_granularity_assigns_stable_names(self) -> None:
        first, second, third = self._patch_graph()
        with first, second, third:
            groups = build_weight_sensitivity_groups(
                object(),
                granularity="site",
            )
        self.assertEqual(
            [group.name for group in groups],
            [
                "layer_000_conv2d_weight",
                "layer_001_prelu_slope",
                "layer_002_depthwise_conv2d_weight",
            ],
        )
        self.assertTrue(all(group.site_count == 1 for group in groups))

    def test_select_groups_preserves_requested_order(self) -> None:
        first, second, third = self._patch_graph()
        with first, second, third:
            groups = build_weight_sensitivity_groups(object())
        selected = select_weight_sensitivity_groups(
            groups,
            ("feature_block_00", "stem"),
        )
        self.assertEqual(
            [group.name for group in selected],
            ["feature_block_00", "stem"],
        )
        with self.assertRaises(KeyError):
            select_weight_sensitivity_groups(groups, ("missing",))

    def test_independent_report_attaches_element_normalized_gain(self) -> None:
        first, second, third = self._patch_graph()
        with first, second, third:
            groups = build_weight_sensitivity_groups(object())
        baseline = {
            "regressors": {"mae": 0.5},
            "classifiers": {"mae": 0.2},
        }
        result = SensitivityResult(
            group="stem",
            outputs={
                "regressors": {"mae": 0.3},
                "classifiers": {"mae": 0.15},
            },
            score=0.3,
            sensitivity=0.2,
            matched_sites=groups[0].site_paths,
        )
        report = build_independent_report(
            baseline=baseline,
            results=(result,),
            groups=groups,
            target_regressor_mae=0.1,
            auxiliary_tolerance=0.0,
        )
        self.assertAlmostEqual(cast(float, report[0]["regressor_mae_improvement"]), 0.2)
        self.assertAlmostEqual(
            cast(float, report[0]["classifier_mae_improvement"]), 0.05
        )
        self.assertFalse(report[0]["regressor_target_reached"])
        self.assertTrue(report[0]["eligible"])
        self.assertGreater(
            cast(float, report[0]["regressor_gain_per_million_parameters"]), 0
        )

    def test_greedy_skips_classifier_regressing_candidate(self) -> None:
        class DummyState:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def set_all(self, enabled):
                del enabled

            def set_where(self, selector, enabled):
                del selector, enabled

        group_a = SimpleNamespace(
            name="a",
            site_paths=("a.weight",),
        )
        group_b = SimpleNamespace(
            name="b",
            site_paths=("b.weight",),
        )
        outputs = [
            {
                "regressors": {"mae": 0.5},
                "classifiers": {"mae": 0.2},
            },
            {
                "regressors": {"mae": 0.3},
                "classifiers": {"mae": 0.25},
            },
            {
                "regressors": {"mae": 0.4},
                "classifiers": {"mae": 0.19},
            },
            {
                "regressors": {"mae": 0.2},
                "classifiers": {"mae": 0.18},
            },
        ]
        sites = (
            SimpleNamespace(path="a.weight"),
            SimpleNamespace(path="b.weight"),
        )
        with (
            mock.patch.object(module, "FakeQuantState", return_value=DummyState()),
            mock.patch.object(module, "evaluate_models", side_effect=outputs),
            mock.patch.object(module, "iter_quantization_sites", return_value=sites),
        ):
            path = module._run_constrained_greedy(
                object(),
                object(),
                (object(),),
                (group_a, group_b),  # type: ignore[arg-type]
                output_adapter=object(),  # type: ignore[arg-type]
                max_steps=2,
                minimum_improvement=0.0,
                auxiliary_tolerance=0.0,
                target_regressor_mae=0.1,
            )
        self.assertEqual([step.group for step in path], ["b", "a"])

    def test_constrained_greedy_rejects_classifier_regression(self) -> None:
        first, second, third = self._patch_graph()
        with first, second, third:
            groups = build_weight_sensitivity_groups(object())

        class FakeState:
            def __init__(self, _model):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def set_all(self, _enabled):
                pass

            def set_where(self, _selector, _enabled):
                pass

        outputs = [
            {"regressors": {"mae": 0.5}, "classifiers": {"mae": 0.2}},
            # Stem improves REG more but violates the zero CLS tolerance.
            {"regressors": {"mae": 0.3}, "classifiers": {"mae": 0.25}},
            {"regressors": {"mae": 0.4}, "classifiers": {"mae": 0.19}},
            # After block 00, adding stem is now balanced and reaches target.
            {"regressors": {"mae": 0.08}, "classifiers": {"mae": 0.18}},
        ]
        with (
            mock.patch.object(module, "FakeQuantState", FakeState),
            mock.patch.object(
                module,
                "iter_quantization_sites",
                return_value=self.sites,
            ),
            mock.patch.object(module, "evaluate_models", side_effect=outputs),
        ):
            path = _run_constrained_greedy(
                object(),
                object(),
                (object(),),
                groups,
                output_adapter=object(),  # type: ignore[arg-type]
                max_steps=0,
                minimum_improvement=0.0,
                auxiliary_tolerance=0.0,
                target_regressor_mae=0.1,
            )
        self.assertEqual([step.group for step in path], ["feature_block_00", "stem"])
        self.assertLess(path[-1].score, 0.1)

    def test_run_can_keep_independently_negative_groups_for_greedy(self) -> None:
        first, second, third = self._patch_graph()
        with first, second, third:
            groups = build_weight_sensitivity_groups(object())

        class FakeRunner:
            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, *_args, **_kwargs):
                return (
                    {
                        "regressors": {"mae": 0.5},
                        "classifiers": {"mae": 0.2},
                    },
                    [
                        SensitivityResult(
                            group="stem",
                            outputs={
                                "regressors": {"mae": 0.3},
                                "classifiers": {"mae": 0.19},
                            },
                            score=0.3,
                            sensitivity=0.2,
                            matched_sites=groups[0].site_paths,
                        ),
                        SensitivityResult(
                            group="feature_block_00",
                            outputs={
                                "regressors": {"mae": 0.6},
                                "classifiers": {"mae": 0.21},
                            },
                            score=0.6,
                            sensitivity=-0.1,
                            matched_sites=groups[1].site_paths,
                        ),
                    ],
                )

        captured: list[str] = []

        def fake_greedy(*_args, **kwargs):
            captured.extend(group.name for group in _args[3])
            return []

        with (
            mock.patch.object(
                module,
                "build_w8a16_candidate",
                return_value=(object(), {}),
            ),
            mock.patch.object(
                module,
                "build_weight_sensitivity_groups",
                return_value=groups,
            ),
            mock.patch.object(module, "QuantizationSensitivity", FakeRunner),
            mock.patch.object(module, "_run_constrained_greedy", fake_greedy),
        ):
            report = module.run_weight_precision_sensitivity(
                object(),
                (object(),),
                (object(),),
                uint8_percentile=99.99,
                int16_observer="minmax",
                int16_percentile=99.99,
                max_samples=128,
                samples_per_batch=64,
                sampling_seed=1,
                requested_groups=None,
                granularity="semantic",
                run_greedy=True,
                greedy_include_all_groups=True,
                greedy_candidate_count=0,
                max_greedy_steps=0,
                minimum_improvement=0.0,
                auxiliary_tolerance=0.0,
                target_regressor_mae=0.1,
                output_adapter=object(),  # type: ignore[arg-type]
            )
        self.assertEqual(captured, ["stem", "feature_block_00"])
        self.assertTrue(report["metadata"]["greedy_include_all_groups"])

    def test_path_report_detects_first_target_step(self) -> None:
        first, second, third = self._patch_graph()
        with first, second, third:
            groups = build_weight_sensitivity_groups(object())
        baseline = {
            "regressors": {"mae": 0.5},
            "classifiers": {"mae": 0.2},
        }
        steps = (
            SensitivityPathResult(
                step=1,
                group="stem",
                selected_groups=("stem",),
                outputs={
                    "regressors": {"mae": 0.2},
                    "classifiers": {"mae": 0.18},
                },
                score=0.2,
                cumulative_sensitivity=0.3,
                incremental_sensitivity=0.3,
                matched_sites=groups[0].site_paths,
                selected_sites=groups[0].site_paths,
            ),
            SensitivityPathResult(
                step=2,
                group="feature_block_00",
                selected_groups=("stem", "feature_block_00"),
                outputs={
                    "regressors": {"mae": 0.08},
                    "classifiers": {"mae": 0.17},
                },
                score=0.08,
                cumulative_sensitivity=0.42,
                incremental_sensitivity=0.12,
                matched_sites=groups[1].site_paths,
                selected_sites=tuple((*groups[0].site_paths, *groups[1].site_paths)),
            ),
        )
        report, summary = build_path_report(
            baseline=baseline,
            results=steps,
            groups=groups,
            target_regressor_mae=0.1,
        )
        self.assertTrue(summary["target_reached"])
        self.assertEqual(summary["first_target_step"], 2)
        self.assertEqual(
            report[-1]["selected_parameter_element_count"],
            sum(group.parameter_element_count for group in groups),
        )


if __name__ == "__main__":
    unittest.main()
