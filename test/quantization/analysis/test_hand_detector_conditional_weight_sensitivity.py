# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0

"""Tests for conditional regular/depthwise weight sensitivity helpers."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

from examples.hand_detector._support import conditional_weight_sensitivity as module
from examples.hand_detector._support.conditional_weight_sensitivity import (
    _build_independent_report,
    _ConditionalWeightEvaluator,
    _run_constrained_greedy,
    build_conditional_weight_groups,
    ConditionalWeightGroup,
    select_conditional_weight_groups,
)


class ConditionalWeightSensitivityTest(unittest.TestCase):
    def _site_group(
        self,
        name: str,
        semantic: str,
        kind: str,
        path: str,
        elements: int,
        position: int,
    ):
        return SimpleNamespace(
            name=name,
            semantic_group=semantic,
            block_kind="feature",
            operation_positions=(position,),
            operation_indices=(position + 100,),
            operation_names=("CONV_2D",),
            site_paths=(path,),
            site_count=1,
            parameter_element_count=elements,
            parameter_breakdown=(SimpleNamespace(kind=kind),),
        )

    def _conditional_group(
        self,
        name: str,
        path: str,
        elements: int,
        position: int,
    ) -> ConditionalWeightGroup:
        return ConditionalWeightGroup(
            name=name,
            family="depthwise",
            granularity="semantic",
            semantic_group=name,
            block_kind="feature",
            operation_positions=(position,),
            operation_indices=(position + 100,),
            operation_names=("DEPTHWISE_CONV_2D",),
            site_paths=(path,),
            parameter_element_count=elements,
        )

    def test_semantic_groups_cover_only_requested_family(self) -> None:
        site_groups = (
            self._site_group(
                "layer_001_regular",
                "feature_block_00",
                "conv2d_weight",
                "regular.weight",
                16,
                1,
            ),
            self._site_group(
                "layer_002_depthwise",
                "feature_block_00",
                "depthwise_conv2d_weight",
                "depthwise.weight",
                8,
                2,
            ),
            self._site_group(
                "layer_003_prelu",
                "feature_block_00",
                "prelu_slope",
                "prelu.weight",
                4,
                3,
            ),
            self._site_group(
                "layer_004_depthwise",
                "feature_block_01",
                "depthwise_conv2d_weight",
                "depthwise2.weight",
                12,
                4,
            ),
        )
        with mock.patch.object(
            module,
            "build_weight_sensitivity_groups",
            return_value=site_groups,
        ):
            definitions, groups = build_conditional_weight_groups(
                object(),
                target_family="depthwise",
                granularity="semantic",
            )
        self.assertEqual(set(definitions), {"regular", "depthwise", "prelu"})
        self.assertEqual(
            [group.name for group in groups],
            ["feature_block_00", "feature_block_01"],
        )
        self.assertEqual(groups[0].site_paths, ("depthwise.weight",))
        self.assertEqual(groups[0].parameter_element_count, 8)
        self.assertEqual(definitions["depthwise"].parameter_element_count, 20)

    def test_site_granularity_keeps_stable_site_names(self) -> None:
        site_groups = (
            self._site_group(
                "layer_002_depthwise_conv2d_weight",
                "feature_block_00",
                "depthwise_conv2d_weight",
                "depthwise.weight",
                8,
                2,
            ),
            self._site_group(
                "layer_003_regular",
                "feature_block_00",
                "conv2d_weight",
                "regular.weight",
                16,
                3,
            ),
            self._site_group(
                "layer_004_prelu",
                "feature_block_00",
                "prelu_slope",
                "prelu.weight",
                4,
                4,
            ),
        )
        with mock.patch.object(
            module,
            "build_weight_sensitivity_groups",
            return_value=site_groups,
        ):
            _, groups = build_conditional_weight_groups(
                object(),
                target_family="depthwise",
                granularity="site",
            )
        self.assertEqual(
            [group.name for group in groups],
            ["layer_002_depthwise_conv2d_weight"],
        )

    def test_select_groups_preserves_requested_order(self) -> None:
        groups = (
            self._conditional_group("a", "a.weight", 10, 1),
            self._conditional_group("b", "b.weight", 20, 2),
        )
        selected = select_conditional_weight_groups(groups, ("b", "a"))
        self.assertEqual([group.name for group in selected], ["b", "a"])
        with self.assertRaises(KeyError):
            select_conditional_weight_groups(groups, ("missing",))

    def test_evaluator_disables_baseline_and_selected_target_paths(self) -> None:
        class DummyState:
            def __init__(self):
                self.enabled = None
                self.calls = []

            def set_all(self, enabled):
                self.enabled = enabled

            def set_where(self, selector, enabled):
                self.calls.append((selector, enabled))

        group = self._conditional_group("g", "depthwise.weight", 8, 2)
        evaluator = object.__new__(_ConditionalWeightEvaluator)
        evaluator.reference_model = object()
        evaluator.candidate_model = object()
        evaluator.samples = (object(),)
        evaluator.baseline_float_paths = frozenset({"regular.weight"})
        evaluator.groups = (group,)
        evaluator.output_adapter = object()
        evaluator._group_paths = {"g": frozenset({"depthwise.weight"})}
        evaluator._cache = {}
        evaluator._state = DummyState()
        evaluator.evaluation_count = 0
        outputs = {
            "regressors": {"mae": 0.1},
            "classifiers": {"mae": 0.05},
        }
        with mock.patch.object(module, "evaluate_models", return_value=outputs):
            result = evaluator.evaluate(frozenset({"g"}))
        self.assertEqual(result, outputs)
        self.assertTrue(evaluator._state.enabled)
        self.assertEqual(len(evaluator._state.calls), 1)
        selector, enabled = evaluator._state.calls[0]
        self.assertFalse(enabled)
        self.assertTrue(selector(SimpleNamespace(path="regular.weight")))
        self.assertTrue(selector(SimpleNamespace(path="depthwise.weight")))
        self.assertFalse(selector(SimpleNamespace(path="prelu.weight")))

    def test_independent_report_uses_conditional_baseline(self) -> None:
        group = self._conditional_group("g", "g.weight", 10, 1)

        class Evaluator:
            def evaluate(self, names):
                self.names = names
                return {
                    "regressors": {"mae": 0.08},
                    "classifiers": {"mae": 0.05},
                }

        evaluator = Evaluator()
        rows = _build_independent_report(
            (group,),
            {
                "regressors": {"mae": 0.18},
                "classifiers": {"mae": 0.06},
            },
            evaluator,
            auxiliary_tolerance=0.0,
            target_regressor_mae=0.1,
            target_classifier_mae=0.1,
            full_parameter_count=100,
        )
        self.assertEqual(evaluator.names, frozenset({"g"}))
        self.assertAlmostEqual(rows[0]["regressor_mae_improvement"], 0.1)
        self.assertTrue(rows[0]["target_reached"])
        self.assertTrue(rows[0]["eligible"])

    def test_greedy_rejects_incremental_classifier_regression(self) -> None:
        group_a = self._conditional_group("a", "a.weight", 10, 1)
        group_b = self._conditional_group("b", "b.weight", 30, 2)
        outputs = {
            frozenset({"a"}): {
                "regressors": {"mae": 0.10},
                "classifiers": {"mae": 0.07},
            },
            frozenset({"b"}): {
                "regressors": {"mae": 0.13},
                "classifiers": {"mae": 0.05},
            },
            frozenset({"a", "b"}): {
                "regressors": {"mae": 0.08},
                "classifiers": {"mae": 0.04},
            },
        }

        class Evaluator:
            def evaluate(self, names):
                return outputs[names]

        steps, summary = _run_constrained_greedy(
            (group_a, group_b),
            {
                "regressors": {"mae": 0.18},
                "classifiers": {"mae": 0.06},
            },
            Evaluator(),
            max_steps=0,
            minimum_improvement=0.0,
            auxiliary_tolerance=0.0,
            target_regressor_mae=0.1,
            target_classifier_mae=0.1,
            full_parameter_count=40,
        )
        self.assertEqual([step["added_group"] for step in steps], ["b", "a"])
        self.assertTrue(summary["target_reached"])
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(summary["selected_parameter_element_count"], 40)


if __name__ == "__main__":
    unittest.main()
