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

"""Tests for the hand-detector precision-floor matrix."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

from examples.hand_detector._support import precision_ablation as precision_module
from examples.hand_detector._support.precision_ablation import (
    _build_floor_summary,
    _policy_path,
    _recommendation,
    PRECISION_FLOOR_PROFILES,
    PrecisionProfileResult,
)
from tico.quantization.wrapq.control import SiteRole


class HandDetectorPrecisionAblationTest(unittest.TestCase):
    def test_precision_floor_profile_contract(self) -> None:
        self.assertEqual(
            [profile.name for profile in PRECISION_FLOOR_PROFILES],
            ["P0", "P1", "P2", "P3", "P4"],
        )
        self.assertEqual(PRECISION_FLOOR_PROFILES[1].regressor_output, "int16")
        self.assertEqual(PRECISION_FLOOR_PROFILES[2].internal_activation, "int16")
        self.assertEqual(PRECISION_FLOOR_PROFILES[3].weight, "float")
        self.assertEqual(PRECISION_FLOOR_PROFILES[4].internal_activation, "float")
        self.assertTrue(
            all(
                profile.classifier_output == "uint8"
                for profile in PRECISION_FLOOR_PROFILES
            )
        )

    def test_policy_path_uses_original_fp_name(self) -> None:
        site = SimpleNamespace(
            module=SimpleNamespace(fp_name="detector.layers.7.conv"),
            module_path="detector.layers.7.conv.wrapped",
            observer_name="act_out",
        )
        self.assertEqual(
            _policy_path(site),  # type: ignore[arg-type]
            "detector.layers.7.conv.act_out",
        )

    def test_discovers_output_paths_without_layer_prefix_collision(self) -> None:
        detector = SimpleNamespace(
            output_tensors=(10, 20),
            operations=(
                {"outputs": (10,)},
                {"outputs": (99,)},
                {"outputs": (20,)},
            ),
        )
        sites = (
            SimpleNamespace(
                path="runtime.layers.0.act_out",
                module_path="runtime.layers.0",
                observer_name="act_out",
                role=SiteRole.ACTIVATION_OUTPUT,
                module=SimpleNamespace(fp_name="detector.layers.0"),
            ),
            # This path begins with the same characters as layer 0 only in a
            # naïve string-prefix implementation and must not be selected.
            SimpleNamespace(
                path="runtime.layers.20.act_out",
                module_path="runtime.layers.20",
                observer_name="act_out",
                role=SiteRole.ACTIVATION_OUTPUT,
                module=SimpleNamespace(fp_name="detector.layers.20"),
            ),
            SimpleNamespace(
                path="runtime.layers.2.conv.act_out",
                module_path="runtime.layers.2.conv",
                observer_name="act_out",
                role=SiteRole.ACTIVATION_OUTPUT,
                module=SimpleNamespace(fp_name="detector.layers.2.conv"),
            ),
        )
        with (
            mock.patch.object(
                precision_module,
                "_prepare_candidate",
                return_value=object(),
            ),
            mock.patch.object(
                precision_module,
                "_find_detector",
                return_value=(detector, "detector."),
            ),
            mock.patch.object(
                precision_module,
                "iter_quantization_sites",
                return_value=sites,
            ),
        ):
            paths = precision_module.discover_output_observer_paths(
                object(),
                object(),  # type: ignore[arg-type]
            )
        self.assertEqual(paths.regressors, "detector.layers.0.act_out")
        self.assertEqual(paths.classifiers, "detector.layers.2.conv.act_out")

    def test_floor_summary_and_recommendation(self) -> None:
        def result(name: str, reg: float) -> PrecisionProfileResult:
            profile = next(
                value for value in PRECISION_FLOOR_PROFILES if value.name == name
            )
            return PrecisionProfileResult(
                profile=profile,
                outputs={
                    "regressors": {"mae": reg},
                    "classifiers": {"mae": 0.01},
                },
                inventory={},
            )

        results = {
            "P0": result("P0", 1.0),
            "P1": result("P1", 0.8),
            "P2": result("P2", 0.3),
            "P3": result("P3", 0.02),
            "P4": result("P4", 0.25),
        }
        floors = _build_floor_summary(results, 0.1)
        self.assertAlmostEqual(floors["uint8_output_penalty"], 0.2)
        self.assertAlmostEqual(floors["a8_internal_penalty_at_w8"], 0.5)
        self.assertAlmostEqual(floors["w8_penalty_at_a16"], 0.28)
        self.assertIn("weight optimization is mandatory", _recommendation(floors))

    def test_parameter_role_name_remains_stable(self) -> None:
        self.assertEqual(SiteRole.PARAMETER.value, "parameter")


if __name__ == "__main__":
    unittest.main()
