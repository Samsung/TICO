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

"""Phase-1 W8/A8/A16 precision-floor analysis for the hand detector."""

from __future__ import annotations

import copy
import math

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector.hand_detector import HandDetector, NHWCInputAdapter
from tico.quantization import convert as freeze_quantization, prepare
from tico.quantization.analysis import evaluate_models, OutputAdapter
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, no_quant, QuantSpec
from tico.quantization.wrapq.control import (
    iter_quantization_sites,
    QuantizationSite,
    SiteRole,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


MetricSummary = Mapping[str, Mapping[str, float | int | None]]


@dataclass(frozen=True)
class PrecisionFloorProfile:
    """Describe one diagnostic precision configuration."""

    name: str
    weight: str
    internal_activation: str
    regressor_output: str
    classifier_output: str
    explanation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "weight": self.weight,
            "internal_activation": self.internal_activation,
            "regressor_output": self.regressor_output,
            "classifier_output": self.classifier_output,
            "explanation": self.explanation,
        }


PHASE1_PROFILES = (
    PrecisionFloorProfile(
        name="P0",
        weight="uint8",
        internal_activation="uint8",
        regressor_output="uint8",
        classifier_output="uint8",
        explanation="Current W8A8 full-integer numerical baseline.",
    ),
    PrecisionFloorProfile(
        name="P1",
        weight="uint8",
        internal_activation="uint8",
        regressor_output="int16",
        classifier_output="uint8",
        explanation="Remove the UINT8 regressor-output representation floor.",
    ),
    PrecisionFloorProfile(
        name="P2",
        weight="uint8",
        internal_activation="int16",
        regressor_output="int16",
        classifier_output="uint8",
        explanation="Measure the best all-A16 activation ceiling with W8 weights.",
    ),
    PrecisionFloorProfile(
        name="P3",
        weight="float",
        internal_activation="int16",
        regressor_output="int16",
        classifier_output="uint8",
        explanation="Measure the A16 floor after removing parameter quantization.",
    ),
    PrecisionFloorProfile(
        name="P4",
        weight="uint8",
        internal_activation="float",
        regressor_output="int16",
        classifier_output="uint8",
        explanation="Measure the W8 floor with floating-point internal activations.",
    ),
)


@dataclass(frozen=True)
class OutputObserverPaths:
    """Store PTQConfig override paths for the two detector outputs."""

    regressors: str
    classifiers: str

    def to_dict(self) -> dict[str, str]:
        return {
            "regressors": self.regressors,
            "classifiers": self.classifiers,
        }


@dataclass(frozen=True)
class ObserverPolicy:
    """Bundle one activation QuantSpec with stable report metadata."""

    spec: QuantSpec
    name: str


@dataclass(frozen=True)
class PrecisionProfileResult:
    """Store one evaluated phase-1 profile."""

    profile: PrecisionFloorProfile
    outputs: MetricSummary
    inventory: Mapping[str, Any]

    def to_dict(self, baseline: MetricSummary) -> dict[str, Any]:
        regressor_mae = _metric(self.outputs, "regressors", "mae")
        classifier_mae = _metric(self.outputs, "classifiers", "mae")
        return {
            **self.profile.to_dict(),
            "outputs": {name: dict(metrics) for name, metrics in self.outputs.items()},
            "inventory": dict(self.inventory),
            "regressor_mae_improvement_vs_p0": (
                _metric(baseline, "regressors", "mae") - regressor_mae
            ),
            "classifier_mae_improvement_vs_p0": (
                _metric(baseline, "classifiers", "mae") - classifier_mae
            ),
        }


def run_precision_floor_ablation(
    float_model: nn.Module,
    calibration_samples: Sequence[torch.Tensor],
    evaluation_samples: Sequence[torch.Tensor],
    *,
    uint8_percentile: float,
    int16_observer: str,
    int16_percentile: float,
    max_samples: int,
    samples_per_batch: int,
    sampling_seed: int,
    target_regressor_mae: float,
    output_adapter: OutputAdapter,
) -> dict[str, Any]:
    """Evaluate P0-P4 with independent candidates and one common FP32 teacher."""
    _validate_inputs(
        calibration_samples,
        evaluation_samples,
        uint8_percentile=uint8_percentile,
        int16_observer=int16_observer,
        int16_percentile=int16_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        target_regressor_mae=target_regressor_mae,
    )
    uint8_policy = _uint8_activation_policy(
        percentile=uint8_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        seed=sampling_seed,
    )
    int16_policy = _int16_activation_policy(
        observer=int16_observer,
        percentile=int16_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        seed=sampling_seed,
    )
    output_paths = discover_output_observer_paths(float_model, uint8_policy.spec)

    profile_results: list[PrecisionProfileResult] = []
    for profile in PHASE1_PROFILES:
        candidate = _build_profile_candidate(
            float_model,
            calibration_samples,
            profile=profile,
            output_paths=output_paths,
            uint8_policy=uint8_policy,
            int16_policy=int16_policy,
        )
        outputs = evaluate_models(
            float_model,
            candidate,
            evaluation_samples,
            output_adapter=output_adapter,
        )
        inventory = summarize_precision_inventory(candidate, output_paths)
        validate_profile_inventory(profile, inventory)
        profile_results.append(
            PrecisionProfileResult(
                profile=profile,
                outputs=outputs,
                inventory=inventory,
            )
        )

    by_name = {result.profile.name: result for result in profile_results}
    baseline = by_name["P0"].outputs
    serialized = [result.to_dict(baseline) for result in profile_results]
    floors = _build_floor_summary(by_name, target_regressor_mae)
    return {
        "analysis": "phase1_precision_floor_ablation",
        "metadata": {
            "weight_policy": "uint8 per-channel asymmetric unless profile=P3",
            "classifier_output_policy": "uint8 per-tensor asymmetric",
            "uint8_activation_observer": uint8_policy.name,
            "uint8_percentile": uint8_percentile,
            "int16_activation_observer": int16_policy.name,
            "int16_percentile": (
                int16_percentile if int16_observer == "percentile" else None
            ),
            "max_samples": max_samples,
            "samples_per_batch": samples_per_batch,
            "sampling_seed": sampling_seed,
            "target_regressor_mae": target_regressor_mae,
            "output_override_paths": output_paths.to_dict(),
            "profile_count": len(PHASE1_PROFILES),
        },
        "profiles": serialized,
        "floors": floors,
        "recommendation": _recommendation(floors),
    }


def discover_output_observer_paths(
    float_model: nn.Module,
    activation_spec: QuantSpec,
) -> OutputObserverPaths:
    """Resolve exact PTQConfig paths for regressor/classifier output observers."""
    probe = _prepare_candidate(
        float_model,
        activation_spec=activation_spec,
        weight_spec=_uint8_weight_spec(),
        overrides={},
    )
    detector, prefix = _find_detector(float_model)
    if len(detector.output_tensors) != len(OUTPUT_NAMES):
        raise RuntimeError(
            "The hand detector output count does not match OUTPUT_NAMES: "
            f"{len(detector.output_tensors)} != {len(OUTPUT_NAMES)}."
        )
    sites = tuple(iter_quantization_sites(probe))
    paths: dict[str, str] = {}
    for output_name, tensor_id in zip(OUTPUT_NAMES, detector.output_tensors):
        positions = tuple(
            position
            for position, operation in enumerate(detector.operations)
            if int(tensor_id) in {int(value) for value in operation["outputs"]}
        )
        if len(positions) != 1:
            raise RuntimeError(
                f"Expected one producer for {output_name!r} tensor {tensor_id}, "
                f"found {positions}."
            )
        layer_prefix = (
            f"{prefix}layers.{positions[0]}" if prefix else f"layers.{positions[0]}"
        )
        matches = tuple(
            site
            for site in sites
            if site.role is SiteRole.ACTIVATION_OUTPUT
            and (
                _fp_module_name(site) == layer_prefix
                or _fp_module_name(site).startswith(layer_prefix + ".")
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one output observer for {output_name!r} under "
                f"{layer_prefix!r}, found {[site.path for site in matches]}."
            )
        paths[output_name] = _policy_path(matches[0])
    return OutputObserverPaths(
        regressors=paths["regressors"],
        classifiers=paths["classifiers"],
    )


def summarize_precision_inventory(
    model: nn.Module,
    output_paths: OutputObserverPaths,
) -> dict[str, Any]:
    """Summarize effective fake-quant sites by role, dtype, and output domain."""
    role_dtype_counts: dict[str, dict[str, int]] = {}
    parameter_dtype_counts: dict[str, int] = {}
    internal_activation_dtype_counts: dict[str, int] = {}
    quantized_site_count = 0
    identity_site_count = 0
    outputs: dict[str, dict[str, str]] = {}
    output_by_path = {
        output_paths.regressors: "regressors",
        output_paths.classifiers: "classifiers",
    }
    for site in iter_quantization_sites(model):
        policy_path = _policy_path(site)
        if isinstance(site.observer, IdentityObserver):
            dtype_name = "float"
            identity_site_count += 1
        else:
            dtype = getattr(site.observer, "dtype", None)
            dtype_name = str(dtype) if dtype is not None else "unknown"
            quantized_site_count += 1
        role = site.role.value
        role_dtype_counts.setdefault(role, {})
        role_dtype_counts[role][dtype_name] = (
            role_dtype_counts[role].get(dtype_name, 0) + 1
        )
        output_name = output_by_path.get(policy_path)
        if output_name is not None:
            outputs[output_name] = {
                "site_path": site.path,
                "policy_path": policy_path,
                "dtype": dtype_name,
            }
        elif site.role is SiteRole.PARAMETER:
            parameter_dtype_counts[dtype_name] = (
                parameter_dtype_counts.get(dtype_name, 0) + 1
            )
        else:
            internal_activation_dtype_counts[dtype_name] = (
                internal_activation_dtype_counts.get(dtype_name, 0) + 1
            )
    if set(outputs) != set(OUTPUT_NAMES):
        raise RuntimeError(
            "Could not identify both output observers in the precision inventory: "
            f"{tuple(outputs)}."
        )
    return {
        "total_site_count": quantized_site_count + identity_site_count,
        "quantized_site_count": quantized_site_count,
        "identity_site_count": identity_site_count,
        "role_dtype_counts": role_dtype_counts,
        "parameter_dtype_counts": parameter_dtype_counts,
        "internal_activation_dtype_counts": internal_activation_dtype_counts,
        "outputs": outputs,
    }


def validate_profile_inventory(
    profile: PrecisionFloorProfile,
    inventory: Mapping[str, Any],
) -> None:
    """Fail fast when a prepared candidate does not implement its profile."""
    outputs = inventory["outputs"]
    expected_outputs = {
        "regressors": profile.regressor_output,
        "classifiers": profile.classifier_output,
    }
    for name, expected in expected_outputs.items():
        actual = outputs[name]["dtype"]
        if actual != expected:
            raise RuntimeError(
                f"Profile {profile.name} expected {name} dtype {expected}, "
                f"but found {actual}."
            )
    parameter_counts = inventory["parameter_dtype_counts"]
    if profile.weight == "float":
        unexpected = {
            dtype: count
            for dtype, count in parameter_counts.items()
            if dtype != "float" and count
        }
        if unexpected:
            raise RuntimeError(
                f"Profile {profile.name} still quantizes parameters: {unexpected}."
            )
        if parameter_counts.get("float", 0) == 0:
            raise RuntimeError(f"Profile {profile.name} has no parameter sites.")
    else:
        unexpected = {
            dtype: count
            for dtype, count in parameter_counts.items()
            if dtype != "uint8" and count
        }
        if unexpected or parameter_counts.get("uint8", 0) == 0:
            raise RuntimeError(
                f"Profile {profile.name} parameter dtypes are invalid: "
                f"{parameter_counts}."
            )

    internal_counts = inventory["internal_activation_dtype_counts"]
    expected_internal = profile.internal_activation
    unexpected = {
        dtype: count
        for dtype, count in internal_counts.items()
        if dtype != expected_internal and count
    }
    if unexpected or internal_counts.get(expected_internal, 0) == 0:
        raise RuntimeError(
            f"Profile {profile.name} internal activation dtypes are invalid: "
            f"{internal_counts}; expected only {expected_internal}."
        )


def print_precision_floor_report(report: Mapping[str, Any]) -> None:
    """Print the P0-P4 matrix and derived floors."""
    print("\nPhase-1 precision floor ablation")
    print(
        f"{'profile':8s} {'W':>6s} {'A_INTERNAL':>12s} {'REG_OUT':>9s} "
        f"{'CLS_OUT':>9s} {'REG_MAE':>13s} {'CLS_MAE':>13s} "
        f"{'Q_SITES':>8s}"
    )
    for profile in report["profiles"]:
        outputs = profile["outputs"]
        inventory = profile["inventory"]
        print(
            f"{profile['name']:8s} "
            f"{profile['weight']:>6s} "
            f"{profile['internal_activation']:>12s} "
            f"{profile['regressor_output']:>9s} "
            f"{profile['classifier_output']:>9s} "
            f"{_metric(outputs, 'regressors', 'mae'):13.6e} "
            f"{_metric(outputs, 'classifiers', 'mae'):13.6e} "
            f"{int(inventory['quantized_site_count']):8d}"
        )
    floors = report["floors"]
    print("\nDerived regressor floors")
    print(
        "UINT8 output penalty (P0-P1):       "
        f"{float(floors['uint8_output_penalty']):.6e}"
    )
    print(
        "A8 internal penalty at W8 (P1-P2): "
        f"{float(floors['a8_internal_penalty_at_w8']):.6e}"
    )
    print(
        "W8 penalty at A16 (P2-P3):         "
        f"{float(floors['w8_penalty_at_a16']):.6e}"
    )
    print(
        "W8 + FP activation floor (P4):     "
        f"{float(floors['w8_fp_activation_floor']):.6e}"
    )
    print("Recommendation: " + str(report["recommendation"]))


def _build_profile_candidate(
    float_model: nn.Module,
    calibration_samples: Sequence[torch.Tensor],
    *,
    profile: PrecisionFloorProfile,
    output_paths: OutputObserverPaths,
    uint8_policy: ObserverPolicy,
    int16_policy: ObserverPolicy,
) -> nn.Module:
    activation_spec = {
        "uint8": uint8_policy.spec,
        "int16": int16_policy.spec,
        "float": no_quant(),
    }[profile.internal_activation]
    weight_spec = _uint8_weight_spec() if profile.weight == "uint8" else no_quant()
    output_specs = {
        "regressors": (
            uint8_policy.spec
            if profile.regressor_output == "uint8"
            else int16_policy.spec
        ),
        "classifiers": uint8_policy.spec,
    }
    overrides = {
        output_paths.regressors: output_specs["regressors"],
        output_paths.classifiers: output_specs["classifiers"],
    }
    candidate = _prepare_candidate(
        float_model,
        activation_spec=activation_spec,
        weight_spec=weight_spec,
        overrides=overrides,
    )
    _calibrate(candidate, calibration_samples)
    return freeze_quantization(candidate, inplace=True).eval()


def _prepare_candidate(
    float_model: nn.Module,
    *,
    activation_spec: QuantSpec,
    weight_spec: QuantSpec,
    overrides: Mapping[str, QuantSpec],
) -> nn.Module:
    candidate = copy.deepcopy(float_model).eval()
    config = PTQConfig(
        activation=activation_spec,
        weight=weight_spec,
        overrides=dict(overrides),
        strict_wrap=False,
    )
    return prepare(candidate, config, inplace=True).eval()


def _calibrate(model: nn.Module, samples: Sequence[torch.Tensor]) -> None:
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            model(sample)


def _uint8_weight_spec() -> QuantSpec:
    return affine(
        DType.uint(8),
        qscheme=QScheme.PER_CHANNEL_ASYMM,
        observer=MinMaxObserver,
    )


def _uint8_activation_policy(
    *,
    percentile: float,
    max_samples: int,
    samples_per_batch: int,
    seed: int,
) -> ObserverPolicy:
    return ObserverPolicy(
        spec=affine(
            DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
            observer=PercentileObserver,
            percentile=percentile,
            max_samples=max_samples,
            samples_per_batch=samples_per_batch,
            seed=seed,
        ),
        name=f"PercentileObserver(P{percentile:g})",
    )


def _int16_activation_policy(
    *,
    observer: str,
    percentile: float,
    max_samples: int,
    samples_per_batch: int,
    seed: int,
) -> ObserverPolicy:
    if observer == "minmax":
        return ObserverPolicy(
            spec=affine(
                DType.int(16),
                qscheme=QScheme.PER_TENSOR_SYMM,
                observer=MinMaxObserver,
            ),
            name="MinMaxObserver",
        )
    if observer == "percentile":
        return ObserverPolicy(
            spec=affine(
                DType.int(16),
                qscheme=QScheme.PER_TENSOR_SYMM,
                observer=PercentileObserver,
                percentile=percentile,
                max_samples=max_samples,
                samples_per_batch=samples_per_batch,
                seed=seed,
            ),
            name=f"PercentileObserver(P{percentile:g})",
        )
    raise ValueError(f"Unsupported INT16 observer: {observer!r}.")


def _build_floor_summary(
    results: Mapping[str, PrecisionProfileResult],
    target_regressor_mae: float,
) -> dict[str, Any]:
    reg = {
        name: _metric(result.outputs, "regressors", "mae")
        for name, result in results.items()
    }
    return {
        "uint8_output_penalty": reg["P0"] - reg["P1"],
        "a8_internal_penalty_at_w8": reg["P1"] - reg["P2"],
        "w8_penalty_at_a16": reg["P2"] - reg["P3"],
        "w8_fp_activation_floor": reg["P4"],
        "a16_with_w8_floor": reg["P2"],
        "a16_with_float_weight_floor": reg["P3"],
        "target_regressor_mae": target_regressor_mae,
        "p1_meets_target": reg["P1"] < target_regressor_mae,
        "p2_meets_target": reg["P2"] < target_regressor_mae,
        "p3_meets_target": reg["P3"] < target_regressor_mae,
        "p4_meets_target": reg["P4"] < target_regressor_mae,
    }


def _recommendation(floors: Mapping[str, Any]) -> str:
    target = float(floors["target_regressor_mae"])
    if not bool(floors["p3_meets_target"]):
        return (
            f"Even FP weights with all-A16 exceed {target:g}; activation precision "
            "above INT16 or a different output representation is required."
        )
    if not bool(floors["p4_meets_target"]):
        return (
            f"W8 exceeds {target:g} even with floating internal activations; "
            "weight optimization is mandatory before mixed-A deployment."
        )
    if not bool(floors["p2_meets_target"]):
        return (
            f"All-A16 with W8 still exceeds {target:g}; combine stronger W8 "
            "optimization with activation mixed precision."
        )
    if not bool(floors["p1_meets_target"]):
        return (
            "Regressor INT16 output removes the output floor, but internal A8 "
            "still dominates; proceed to A8-to-A16 sensitivity."
        )
    return "The target is reachable without all-A16; begin precision-island search."


def _find_detector(model: nn.Module) -> tuple[HandDetector, str]:
    if isinstance(model, NHWCInputAdapter):
        return model.detector, "detector."
    if isinstance(model, HandDetector):
        return model, ""
    detector = getattr(model, "detector", None)
    if isinstance(detector, HandDetector):
        return detector, "detector."
    raise TypeError("Expected HandDetector or NHWCInputAdapter.")


def _fp_module_name(site: QuantizationSite) -> str:
    return getattr(site.module, "fp_name", None) or site.module_path


def _policy_path(site: QuantizationSite) -> str:
    return f"{_fp_module_name(site)}.{site.observer_name}"


def _metric(
    outputs: MetricSummary,
    output_name: str,
    metric_name: str,
) -> float:
    value = outputs[output_name][metric_name]
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Output {output_name!r} metric {metric_name!r} is not numeric."
        )
    return float(value)


def _validate_inputs(
    calibration_samples: Sequence[torch.Tensor],
    evaluation_samples: Sequence[torch.Tensor],
    *,
    uint8_percentile: float,
    int16_observer: str,
    int16_percentile: float,
    max_samples: int,
    samples_per_batch: int,
    target_regressor_mae: float,
) -> None:
    if not calibration_samples or not evaluation_samples:
        raise ValueError("Precision ablation requires calibration and evaluation data.")
    for name, value in (
        ("uint8_percentile", uint8_percentile),
        ("int16_percentile", int16_percentile),
    ):
        if not math.isfinite(value) or not 0.0 < value <= 100.0:
            raise ValueError(f"{name} must be finite and in (0, 100].")
    if int16_observer not in {"minmax", "percentile"}:
        raise ValueError("int16_observer must be 'minmax' or 'percentile'.")
    if max_samples <= 0 or samples_per_batch <= 0:
        raise ValueError("Observer sample limits must be positive.")
    if not math.isfinite(target_regressor_mae) or target_regressor_mae <= 0.0:
        raise ValueError("target_regressor_mae must be finite and positive.")
