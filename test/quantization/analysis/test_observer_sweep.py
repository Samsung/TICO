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

"""Tests for C/D/E activation observer sweep reporting."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from examples.hand_detector._support.observer_sweep import (
    build_observer_sweep_result,
    evaluate_observer_profiles,
    OBSERVER_SWEEP_PROFILES,
    print_observer_sweep,
    rank_observer_sweep_results,
    serialize_observer_profiles,
)
from tico.quantization.analysis import (
    QuantizationProfile,
    QuantizationProfileResult,
    QuantizationReport,
)


_SITE_COUNTS = {
    QuantizationProfile.ACTIVATION_ONLY: 197,
    QuantizationProfile.FULL: 293,
    QuantizationProfile.INTERNAL_FULL: 291,
}


def _profile_result(
    profile: QuantizationProfile,
    *,
    regressor_mae: float,
    classifier_mae: float,
) -> dict[str, object]:
    return {
        "profile": profile.value,
        "label": profile.label,
        "enabled_site_count": _SITE_COUNTS[profile],
        "outputs": {
            "regressors": {
                "mae": regressor_mae,
                "cosine_similarity": 1.0 - regressor_mae / 100.0,
            },
            "classifiers": {
                "mae": classifier_mae,
                "cosine_similarity": 1.0 - classifier_mae / 100.0,
            },
        },
    }


def _profiles(
    *,
    c_reg: float,
    d_reg: float,
    e_reg: float,
    c_cls: float = 0.3,
    d_cls: float = 0.2,
    e_cls: float = 0.1,
) -> dict[str, dict[str, object]]:
    values = {
        QuantizationProfile.ACTIVATION_ONLY: (c_reg, c_cls),
        QuantizationProfile.FULL: (d_reg, d_cls),
        QuantizationProfile.INTERNAL_FULL: (e_reg, e_cls),
    }
    return {
        profile.value: _profile_result(
            profile,
            regressor_mae=regressor_mae,
            classifier_mae=classifier_mae,
        )
        for profile, (regressor_mae, classifier_mae) in values.items()
    }


def _report() -> QuantizationReport:
    profiles = _profiles(c_reg=3.0, d_reg=2.0, e_reg=1.0)
    return QuantizationReport(
        float_parity={},
        profiles={
            profile: QuantizationProfileResult(
                profile=profile,
                enabled_sites=tuple(
                    f"{profile.value}.site.{index}"
                    for index in range(_SITE_COUNTS[profile])
                ),
                outputs=cast(dict, profiles[profile.value]["outputs"]),
            )
            for profile in OBSERVER_SWEEP_PROFILES
        },
    )


class ObserverSweepTest(unittest.TestCase):
    """Verify profile selection, compatibility, ranking, and terminal output."""

    @patch("examples.hand_detector._support.observer_sweep.QuantizationAblation")
    def test_evaluate_observer_profiles_requests_c_d_e(self, ablation_cls) -> None:
        """Use the standard C/D/E profiles for every observer candidate."""
        profile_results = {
            profile: SimpleNamespace(
                enabled_site_count=_SITE_COUNTS[profile],
                outputs=_profile_result(
                    profile,
                    regressor_mae=float(index + 1),
                    classifier_mae=float(index + 1) / 10.0,
                )["outputs"],
            )
            for index, profile in enumerate(OBSERVER_SWEEP_PROFILES)
        }
        report = SimpleNamespace(profiles=profile_results)
        runner = ablation_cls.return_value
        runner.run.return_value = report

        result = evaluate_observer_profiles(
            MagicMock(),
            MagicMock(),
            [MagicMock()],
            boundaries=MagicMock(),
            output_adapter=MagicMock(),
        )

        self.assertEqual(tuple(result), ("C", "D", "E"))
        self.assertEqual(
            runner.run.call_args.kwargs["profiles"],
            OBSERVER_SWEEP_PROFILES,
        )

    def test_serialization_drops_repeated_enabled_site_paths(self) -> None:
        """Retain site counts without multiplying large path lists per candidate."""
        profiles = serialize_observer_profiles(_report())

        self.assertEqual(tuple(profiles), ("C", "D", "E"))
        for profile in profiles.values():
            self.assertIn("enabled_site_count", profile)
            self.assertNotIn("enabled_sites", profile)

    def test_result_preserves_full_outputs_compatibility_alias(self) -> None:
        """Keep the former outputs field as an alias for D:full metrics."""
        profiles = _profiles(c_reg=3.0, d_reg=2.0, e_reg=1.0)
        result = build_observer_sweep_result(
            observer="PercentileObserver",
            percentile=99.99,
            profiles=profiles,
            observer_details=(),
        )

        self.assertEqual(result["outputs"], profiles["D"]["outputs"])
        self.assertEqual(tuple(result["profiles"]), ("C", "D", "E"))
        self.assertEqual(result["percentile"], 99.99)

    def test_result_rejects_a_missing_profile(self) -> None:
        """Reject incomplete reports instead of silently omitting C, D, or E."""
        profiles = _profiles(c_reg=3.0, d_reg=2.0, e_reg=1.0)
        del profiles["E"]

        with self.assertRaisesRegex(ValueError, "missing profiles"):
            build_observer_sweep_result(
                observer="MinMaxObserver",
                profiles=profiles,
                observer_details=(),
            )

    def test_ranking_uses_internal_full_regressor_mae(self) -> None:
        """Prefer the best E candidate even when a different observer wins D."""
        results = {
            "minmax": build_observer_sweep_result(
                observer="MinMaxObserver",
                profiles=_profiles(c_reg=0.5, d_reg=0.1, e_reg=0.4),
                observer_details=(),
            ),
            "percentile_99_99": build_observer_sweep_result(
                observer="PercentileObserver",
                percentile=99.99,
                profiles=_profiles(c_reg=0.6, d_reg=0.2, e_reg=0.3),
                observer_details=(),
            ),
        }

        ranked = rank_observer_sweep_results(results)
        self.assertEqual(ranked[0][0], "percentile_99_99")

    def test_prints_c_d_e_profiles_and_ranks_by_e(self) -> None:
        """Expose every requested profile and mark the E-regressor winner."""
        results = {
            "minmax": build_observer_sweep_result(
                observer="MinMaxObserver",
                profiles=_profiles(c_reg=0.5, d_reg=0.1, e_reg=0.4),
                observer_details=(),
            ),
            "percentile_99_99": build_observer_sweep_result(
                observer="PercentileObserver",
                percentile=99.99,
                profiles=_profiles(c_reg=0.6, d_reg=0.2, e_reg=0.3),
                observer_details=(),
            ),
        }

        output = io.StringIO()
        with redirect_stdout(output):
            print_observer_sweep(results, "uint8")
        text = output.getvalue()

        self.assertIn("*percentile_99_99", text)
        self.assertEqual(text.count("C:activation-only"), 2)
        self.assertEqual(text.count("D:full"), 2)
        self.assertGreaterEqual(text.count("E:internal-full"), 2)
        self.assertIn("     197", text)
        self.assertIn("     293", text)
        self.assertIn("     291", text)
        self.assertIn("lowest E:internal-full regressor MAE", text)


if __name__ == "__main__":
    unittest.main()
