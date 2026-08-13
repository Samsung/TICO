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

"""C/D/E profile evaluation and reporting for activation observer sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tico.quantization.analysis import (
    ModelInput,
    OutputAdapter,
    QuantizationAblation,
    QuantizationBoundaries,
    QuantizationProfile,
    QuantizationReport,
)

from torch import nn


OBSERVER_SWEEP_PROFILES = (
    QuantizationProfile.ACTIVATION_ONLY,
    QuantizationProfile.FULL,
    QuantizationProfile.INTERNAL_FULL,
)
OBSERVER_SWEEP_RANKING_PROFILE = QuantizationProfile.INTERNAL_FULL


def evaluate_observer_profiles(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    samples: Sequence[ModelInput],
    *,
    boundaries: QuantizationBoundaries,
    output_adapter: OutputAdapter | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate one calibrated observer candidate under C, D, and E."""
    report = QuantizationAblation(
        reference_model,
        candidate_model,
        boundaries=boundaries,
        output_adapter=output_adapter,
    ).run(
        samples,
        profiles=OBSERVER_SWEEP_PROFILES,
    )
    return serialize_observer_profiles(report)


def serialize_observer_profiles(
    report: QuantizationReport,
) -> dict[str, dict[str, Any]]:
    """Return compact C/D/E results without repeated enabled-site path lists."""
    profiles: dict[str, dict[str, Any]] = {}
    for profile in OBSERVER_SWEEP_PROFILES:
        result = report.profiles[profile]
        profiles[profile.value] = {
            "profile": profile.value,
            "label": profile.label,
            "enabled_site_count": result.enabled_site_count,
            "outputs": _copy_outputs(result.outputs),
        }
    return profiles


def build_observer_sweep_result(
    *,
    observer: str,
    profiles: Mapping[str, Mapping[str, Any]],
    observer_details: Sequence[Mapping[str, Any]],
    percentile: float | None = None,
) -> dict[str, Any]:
    """Build one JSON-ready observer result with a legacy full-output alias."""
    missing_profiles = tuple(
        profile.value
        for profile in OBSERVER_SWEEP_PROFILES
        if profile.value not in profiles
    )
    if missing_profiles:
        raise ValueError(
            "Observer sweep results are missing profiles: " f"{missing_profiles}."
        )

    normalized_profiles = {
        profile.value: dict(profiles[profile.value])
        for profile in OBSERVER_SWEEP_PROFILES
    }
    full_outputs = normalized_profiles[QuantizationProfile.FULL.value].get("outputs")
    if not isinstance(full_outputs, Mapping):
        raise ValueError("The full profile does not contain output metrics.")

    result: dict[str, Any] = {
        "observer": observer,
        "profiles": normalized_profiles,
        # Preserve the previous report field as a compatibility alias for D:full.
        "outputs": _copy_outputs(full_outputs),
        "observer_details": [dict(details) for details in observer_details],
    }
    if percentile is not None:
        result["percentile"] = float(percentile)
    return result


def rank_observer_sweep_results(
    results: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Rank observer candidates by E regressor MAE."""
    if not results:
        raise ValueError("Observer sweep results must not be empty.")
    return sorted(
        results.items(),
        key=lambda item: float(
            _profile_outputs(item[1], OBSERVER_SWEEP_RANKING_PROFILE,)[
                "regressors"
            ]["mae"]
        ),
    )


def print_observer_sweep(
    results: Mapping[str, Mapping[str, Any]],
    dtype_name: str,
) -> None:
    """Print C/D/E metrics and rank candidates by internal-full regressor MAE."""
    print(f"\n{dtype_name.upper()} activation observer sweep")
    print("Candidates are ranked by E:internal-full regressor MAE.")
    print(
        f"{'observer':24s} {'profile':18s} {'REG_MAE':>13s} "
        f"{'REG_COS':>13s} {'CLS_MAE':>13s} {'CLS_COS':>13s} {'SITES':>7s}"
    )
    ranked = rank_observer_sweep_results(results)
    for observer_index, (name, result) in enumerate(ranked):
        for profile_index, profile in enumerate(OBSERVER_SWEEP_PROFILES):
            marker = "*" if observer_index == 0 and profile_index == 0 else " "
            observer_name = name if profile_index == 0 else ""
            profile_result = _profile_result(result, profile)
            outputs = profile_result["outputs"]
            profile_label = f"{profile.value}:{profile.label}"
            print(
                f"{marker}{observer_name:23s} "
                f"{profile_label:18s} "
                f"{float(outputs['regressors']['mae']):13.6e} "
                f"{float(outputs['regressors']['cosine_similarity']):13.9f} "
                f"{float(outputs['classifiers']['mae']):13.6e} "
                f"{float(outputs['classifiers']['cosine_similarity']):13.9f} "
                f"{int(profile_result['enabled_site_count']):7d}"
            )
    print("* lowest E:internal-full regressor MAE.")


def _profile_result(
    result: Mapping[str, Any],
    profile: QuantizationProfile,
) -> Mapping[str, Any]:
    profiles = result.get("profiles")
    if not isinstance(profiles, Mapping):
        raise TypeError("Observer sweep result does not contain profile metrics.")
    profile_result = profiles.get(profile.value)
    if not isinstance(profile_result, Mapping):
        raise KeyError(
            "Observer sweep result does not contain profile " f"{profile.value}."
        )
    return profile_result


def _profile_outputs(
    result: Mapping[str, Any],
    profile: QuantizationProfile,
) -> Mapping[str, Mapping[str, float | int | None]]:
    outputs = _profile_result(result, profile).get("outputs")
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"Observer sweep profile {profile.value} does not contain output metrics."
        )
    return outputs  # type: ignore[return-value]


def _copy_outputs(
    outputs: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    """Copy nested output metrics into a JSON-ready mapping."""
    return {name: dict(metrics) for name, metrics in outputs.items()}
