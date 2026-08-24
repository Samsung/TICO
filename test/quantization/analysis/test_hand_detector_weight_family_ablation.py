# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0

"""Tests for hand-detector W8/A16 parameter-family ablation helpers."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

from examples.hand_detector._support import weight_family_ablation as module
from examples.hand_detector._support.weight_family_ablation import (
    build_weight_families,
    build_weight_family_summary,
    WEIGHT_FAMILY_PROFILES,
)


class HandDetectorWeightFamilyAblationTest(unittest.TestCase):
    def _site_group(self, name, kind, elements, path):
        return SimpleNamespace(
            name=name,
            site_count=1,
            site_paths=(path,),
            parameter_element_count=elements,
            parameter_breakdown=(SimpleNamespace(kind=kind),),
        )

    def test_profiles_match_f0_to_f4_contract(self) -> None:
        self.assertEqual(
            [profile.name for profile in WEIGHT_FAMILY_PROFILES],
            ["F0", "F1", "F2", "F3", "F4"],
        )
        self.assertEqual(WEIGHT_FAMILY_PROFILES[0].float_families, ())
        self.assertEqual(
            WEIGHT_FAMILY_PROFILES[4].float_families,
            ("regular_conv", "depthwise_conv"),
        )

    def test_build_weight_families_covers_supported_parameter_kinds(self) -> None:
        groups = (
            self._site_group("a", "conv2d_weight", 100, "a.weight"),
            self._site_group(
                "b",
                "depthwise_conv2d_weight",
                20,
                "b.weight",
            ),
            self._site_group("c", "prelu_slope", 4, "c.weight"),
            self._site_group("d", "conv2d_weight", 10, "d.weight"),
        )
        with mock.patch.object(
            module,
            "build_weight_sensitivity_groups",
            return_value=groups,
        ):
            families = build_weight_families(object())
        self.assertEqual(
            [family.name for family in families],
            ["regular_conv", "depthwise_conv", "prelu_slope"],
        )
        self.assertEqual(families[0].parameter_element_count, 110)
        self.assertEqual(families[0].site_count, 2)
        self.assertEqual(families[1].site_paths, ("b.weight",))

    def test_unknown_parameter_kind_fails_loudly(self) -> None:
        groups = (self._site_group("a", "linear_weight", 10, "a.weight"),)
        with (
            mock.patch.object(
                module,
                "build_weight_sensitivity_groups",
                return_value=groups,
            ),
            self.assertRaises(ValueError),
        ):
            build_weight_families(object())

    def test_summary_selects_smallest_target_feasible_oracle(self) -> None:
        def profile(name, reg, cls, float_count, families):
            return {
                "name": name,
                "outputs": {
                    "regressors": {"mae": reg},
                    "classifiers": {"mae": cls},
                },
                "float_parameter_element_count": float_count,
                "float_parameter_ratio": float_count / 1000.0,
                "float_families": list(families),
                "target_feasible": reg < 0.1 and cls < 0.1,
            }

        profiles = (
            profile("F0", 0.5, 0.2, 0, ()),
            profile("F1", 0.08, 0.05, 700, ("regular_conv",)),
            profile("F2", 0.2, 0.08, 200, ("depthwise_conv",)),
            profile("F3", 0.49, 0.19, 10, ("prelu_slope",)),
            profile(
                "F4",
                0.04,
                0.03,
                900,
                ("regular_conv", "depthwise_conv"),
            ),
        )
        summary = build_weight_family_summary(
            profiles,
            p3_outputs={
                "regressors": {"mae": 0.01},
                "classifiers": {"mae": 0.02},
            },
            target_regressor_mae=0.1,
            target_classifier_mae=0.1,
        )
        self.assertEqual(summary["best_target_feasible_profile"], "F1")
        self.assertEqual(
            summary["best_target_feasible_float_families"],
            ["regular_conv"],
        )
        self.assertAlmostEqual(summary["regular_conv_regressor_gain"], 0.42)
        self.assertAlmostEqual(summary["depthwise_conv_regressor_gain"], 0.3)

    def test_recommendation_requires_broad_joint_optimization(self) -> None:
        summary = {
            "best_target_feasible_profile": None,
            "p3_target_feasible": True,
        }
        self.assertIn("broad joint", module._recommendation(summary))

    def test_evaluator_disables_only_requested_family_paths(self) -> None:
        class DummyState:
            def __init__(self):
                self.disabled = []

            def set_all(self, enabled):
                self.enabled = enabled

            def set_where(self, selector, enabled):
                self.disabled.append((selector, enabled))

        evaluator = object.__new__(module._WeightFamilyEvaluator)
        evaluator.reference_model = object()
        evaluator.candidate_model = object()
        evaluator.samples = (object(),)
        evaluator.output_adapter = object()  # type: ignore[assignment]
        evaluator._family_paths = {
            "regular_conv": frozenset({"a.weight"}),
            "depthwise_conv": frozenset({"b.weight"}),
            "prelu_slope": frozenset({"c.weight"}),
        }
        evaluator._cache = {}
        evaluator._state = DummyState()  # type: ignore[assignment]
        outputs = {
            "regressors": {"mae": 0.1},
            "classifiers": {"mae": 0.05},
        }
        with mock.patch.object(module, "evaluate_models", return_value=outputs):
            result = evaluator.evaluate(frozenset({"regular_conv"}))
        self.assertEqual(result, outputs)
        self.assertTrue(evaluator._state.enabled)  # type: ignore[union-attr]
        self.assertEqual(len(evaluator._state.disabled), 1)  # type: ignore[union-attr]
        self.assertFalse(evaluator._state.disabled[0][1])  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
