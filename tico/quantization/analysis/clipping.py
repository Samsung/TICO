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

"""Model-independent output clipping and quantization-grid analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from tico.quantization.analysis.inputs import invoke_model, ModelInput
from tico.quantization.analysis.metrics import TensorErrorMetrics
from tico.quantization.analysis.output_quantization import (
    AffineQuantizationPolicy,
    OutputCodeStatistics,
    OutputTensorQuantizer,
)
from tico.quantization.analysis.outputs import make_output_adapter, OutputAdapter


@dataclass(frozen=True)
class OutputCalibrationData:
    """Store exact extrema and bounded representative values for one output."""

    name: str
    values: torch.Tensor
    observed_minimum: float
    observed_maximum: float
    total_value_count: int

    @property
    def sampled_value_count(self) -> int:
        return int(self.values.numel())


@dataclass(frozen=True)
class ClippingCandidate:
    """Describe one output clipping range and its calibration error."""

    name: str
    method: str
    minimum: float
    maximum: float
    lower_tail_percent: float
    upper_tail_percent: float
    calibration_error: Mapping[str, float | int | None]


@dataclass(frozen=True)
class EvaluatedClippingCandidate:
    """Store calibration and evaluation results for one clipping candidate."""

    candidate: ClippingCandidate
    evaluation_error: Mapping[str, float | int | None]
    quantizer: Mapping[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.candidate.name,
            "method": self.candidate.method,
            "minimum": self.candidate.minimum,
            "maximum": self.candidate.maximum,
            "lower_tail_percent": self.candidate.lower_tail_percent,
            "upper_tail_percent": self.candidate.upper_tail_percent,
            "calibration_error": dict(self.candidate.calibration_error),
            "evaluation_error": dict(self.evaluation_error),
            "quantizer": dict(self.quantizer),
        }


def collect_output_calibration_data(
    model: nn.Module,
    samples: Sequence[ModelInput],
    *,
    output_adapter: OutputAdapter | None = None,
    max_values_per_output: int = 1_000_000,
    seed: int = 0,
) -> tuple[OutputCalibrationData, ...]:
    """Run float inference and retain bounded values from every named output."""
    if not samples:
        raise ValueError("Output calibration requires at least one sample.")
    if max_values_per_output <= 0:
        raise ValueError("max_values_per_output must be positive.")
    adapter = output_adapter or make_output_adapter()
    per_input_limit = max(1, math.ceil(max_values_per_output / len(samples)))
    chunks: dict[str, list[torch.Tensor]] = {}
    minima: dict[str, float] = {}
    maxima: dict[str, float] = {}
    counts: dict[str, int] = {}
    generators: dict[str, torch.Generator] = {}

    expected_output_names: tuple[str, ...] | None = None
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            outputs = dict(adapter(invoke_model(model, sample)))
            if not outputs:
                raise ValueError("The output adapter returned no tensors.")
            output_names = tuple(outputs)
            if expected_output_names is None:
                expected_output_names = output_names
            elif output_names != expected_output_names:
                raise ValueError(
                    "Output names changed between calibration samples: "
                    f"{output_names} != {expected_output_names}."
                )
            for output_index, (name, output) in enumerate(outputs.items()):
                flattened = _finite_values(output)
                if flattened.numel() == 0:
                    continue
                chunks.setdefault(name, [])
                minima[name] = min(minima.get(name, math.inf), float(flattened.min()))
                maxima[name] = max(maxima.get(name, -math.inf), float(flattened.max()))
                counts[name] = counts.get(name, 0) + flattened.numel()
                if name not in generators:
                    generators[name] = torch.Generator(device="cpu").manual_seed(
                        seed + output_index
                    )
                chunks[name].append(
                    _stratified_sample(
                        flattened,
                        per_input_limit,
                        generators[name],
                    )
                )

    result: list[OutputCalibrationData] = []
    for output_index, name in enumerate(chunks):
        values = torch.cat(chunks[name])
        if values.numel() > max_values_per_output:
            values = _stratified_sample(
                values,
                max_values_per_output,
                torch.Generator(device="cpu").manual_seed(seed + 10_000 + output_index),
            )
        result.append(
            OutputCalibrationData(
                name=name,
                values=values,
                observed_minimum=minima[name],
                observed_maximum=maxima[name],
                total_value_count=counts[name],
            )
        )
    if not result:
        raise RuntimeError("No finite output values were collected.")
    return tuple(result)


def build_clipping_candidates(
    data: OutputCalibrationData,
    policy: AffineQuantizationPolicy,
    *,
    percentiles: Sequence[float],
    tail_percentages: Sequence[float],
    include_l1_search: bool = True,
) -> tuple[ClippingCandidate, ...]:
    """Build MinMax, fixed-percentile, and optional L1-grid candidates."""
    _validate_percentages(percentiles, tail_percentages)
    candidates = [
        _candidate(
            data,
            policy,
            name="minmax",
            method="minmax",
            minimum=data.observed_minimum,
            maximum=data.observed_maximum,
            lower_tail=0.0,
            upper_tail=0.0,
        )
    ]
    seen_ranges = {(data.observed_minimum, data.observed_maximum)}
    for percentile in percentiles:
        minimum, maximum, lower_tail, upper_tail = percentile_range(
            data,
            policy,
            float(percentile),
        )
        key = (minimum, maximum)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        label = f"p{percentile:g}".replace(".", "_")
        candidates.append(
            _candidate(
                data,
                policy,
                name=label,
                method="percentile",
                minimum=minimum,
                maximum=maximum,
                lower_tail=lower_tail,
                upper_tail=upper_tail,
            )
        )
    if include_l1_search:
        candidates.append(find_l1_optimal_candidate(data, policy, tail_percentages))
    return tuple(candidates)


def evaluate_clipping_candidates(
    model: nn.Module,
    samples: Sequence[ModelInput],
    calibration_data: Sequence[OutputCalibrationData],
    candidates: Mapping[str, Sequence[ClippingCandidate]],
    policy: AffineQuantizationPolicy,
    *,
    output_adapter: OutputAdapter | None = None,
) -> dict[str, list[EvaluatedClippingCandidate]]:
    """Evaluate precomputed clipping candidates on independent model samples."""
    if not samples:
        raise ValueError("Output clipping evaluation requires at least one sample.")
    adapter = output_adapter or make_output_adapter()
    data_by_name = {data.name: data for data in calibration_data}
    quantizers: dict[str, list[OutputTensorQuantizer]] = {}
    errors: dict[str, list[TensorErrorMetrics]] = {}
    statistics: dict[str, list[OutputCodeStatistics]] = {}
    for name, output_candidates in candidates.items():
        if name not in data_by_name:
            raise KeyError(f"No calibration data exists for output {name!r}.")
        quantizers[name] = [
            OutputTensorQuantizer.from_range(
                name,
                policy,
                candidate.minimum,
                candidate.maximum,
            )
            for candidate in output_candidates
        ]
        errors[name] = [TensorErrorMetrics() for _ in output_candidates]
        statistics[name] = [OutputCodeStatistics() for _ in output_candidates]

    model.eval()
    with torch.inference_mode():
        for sample in samples:
            outputs = dict(adapter(invoke_model(model, sample)))
            for name, output_quantizers in quantizers.items():
                if name not in outputs:
                    raise KeyError(f"Evaluation output {name!r} is missing.")
                reference = outputs[name]
                for index, quantizer in enumerate(output_quantizers):
                    candidate = quantizer.fake_quant(reference)
                    errors[name][index].update(reference, candidate)
                    statistics[name][index].update(reference, quantizer)

    result: dict[str, list[EvaluatedClippingCandidate]] = {}
    for name, output_candidates in candidates.items():
        result[name] = [
            EvaluatedClippingCandidate(
                candidate=candidate,
                evaluation_error=errors[name][index].summary(),
                quantizer=statistics[name][index].summary(quantizers[name][index]),
            )
            for index, candidate in enumerate(output_candidates)
        ]
    return result


def percentile_range(
    data: OutputCalibrationData,
    policy: AffineQuantizationPolicy,
    percentile: float,
) -> tuple[float, float, float, float]:
    """Return bounds and effective tail percentages for one percentile."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in the interval (0, 100].")
    if percentile == 100.0:
        return data.observed_minimum, data.observed_maximum, 0.0, 0.0
    retained = percentile / 100.0
    if policy.qscheme.is_symmetric():
        threshold = _quantile(data.values.abs(), retained)
        tail = 100.0 - percentile
        return -threshold, threshold, tail, tail
    if data.observed_minimum >= 0.0:
        return 0.0, _quantile(data.values, retained), 0.0, 100.0 - percentile
    if data.observed_maximum <= 0.0:
        return _quantile(data.values, 1.0 - retained), 0.0, 100.0 - percentile, 0.0
    tail = (1.0 - retained) / 2.0
    return (
        _quantile(data.values, tail),
        _quantile(data.values, 1.0 - tail),
        tail * 100.0,
        tail * 100.0,
    )


def find_l1_optimal_candidate(
    data: OutputCalibrationData,
    policy: AffineQuantizationPolicy,
    tail_percentages: Sequence[float],
) -> ClippingCandidate:
    """Search independent tail clipping values that minimize calibration MAE."""
    tails = tuple(sorted(set(float(value) for value in tail_percentages)))
    if policy.qscheme.is_symmetric():
        combinations: Iterable[tuple[float, float]] = ((tail, tail) for tail in tails)
    elif data.observed_minimum >= 0.0:
        combinations = ((0.0, tail) for tail in tails)
    elif data.observed_maximum <= 0.0:
        combinations = ((tail, 0.0) for tail in tails)
    else:
        combinations = (
            (lower_tail, upper_tail) for lower_tail in tails for upper_tail in tails
        )

    best: ClippingCandidate | None = None
    for lower_tail, upper_tail in combinations:
        minimum, maximum = _tail_range(
            data,
            policy,
            lower_tail,
            upper_tail,
        )
        candidate = _candidate(
            data,
            policy,
            name="l1_optimal",
            method="l1_grid_search",
            minimum=minimum,
            maximum=maximum,
            lower_tail=lower_tail,
            upper_tail=upper_tail,
        )
        if best is None:
            best = candidate
            continue
        key = (
            float(candidate.calibration_error["mae"]),
            lower_tail + upper_tail,
        )
        best_key = (
            float(best.calibration_error["mae"]),
            best.lower_tail_percent + best.upper_tail_percent,
        )
        if key < best_key:
            best = candidate
    if best is None:
        raise RuntimeError("No L1 clipping candidate was generated.")
    return best


def _candidate(
    data: OutputCalibrationData,
    policy: AffineQuantizationPolicy,
    *,
    name: str,
    method: str,
    minimum: float,
    maximum: float,
    lower_tail: float,
    upper_tail: float,
) -> ClippingCandidate:
    quantizer = OutputTensorQuantizer.from_range(
        data.name,
        policy,
        minimum,
        maximum,
    )
    metrics = TensorErrorMetrics()
    metrics.update(data.values, quantizer.fake_quant(data.values))
    return ClippingCandidate(
        name=name,
        method=method,
        minimum=minimum,
        maximum=maximum,
        lower_tail_percent=lower_tail,
        upper_tail_percent=upper_tail,
        calibration_error=metrics.summary(),
    )


def _tail_range(
    data: OutputCalibrationData,
    policy: AffineQuantizationPolicy,
    lower_tail_percent: float,
    upper_tail_percent: float,
) -> tuple[float, float]:
    lower_fraction = lower_tail_percent / 100.0
    upper_fraction = upper_tail_percent / 100.0
    if policy.qscheme.is_symmetric():
        if lower_tail_percent != upper_tail_percent:
            raise ValueError("Symmetric quantization requires one common tail value.")
        threshold = (
            max(abs(data.observed_minimum), abs(data.observed_maximum))
            if lower_tail_percent == 0.0
            else _quantile(data.values.abs(), 1.0 - lower_fraction)
        )
        return -threshold, threshold
    if data.observed_minimum >= 0.0:
        maximum = (
            data.observed_maximum
            if upper_tail_percent == 0.0
            else _quantile(data.values, 1.0 - upper_fraction)
        )
        return 0.0, maximum
    if data.observed_maximum <= 0.0:
        minimum = (
            data.observed_minimum
            if lower_tail_percent == 0.0
            else _quantile(data.values, lower_fraction)
        )
        return minimum, 0.0
    minimum = (
        data.observed_minimum
        if lower_tail_percent == 0.0
        else _quantile(data.values, lower_fraction)
    )
    maximum = (
        data.observed_maximum
        if upper_tail_percent == 0.0
        else _quantile(data.values, 1.0 - upper_fraction)
    )
    return min(minimum, 0.0), max(maximum, 0.0)


def _quantile(values: torch.Tensor, fraction: float) -> float:
    if fraction <= 0.0:
        return float(values.min())
    if fraction >= 1.0:
        return float(values.max())
    return float(torch.quantile(values.to(torch.float64), fraction))


def _finite_values(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.detach().reshape(-1)
    finite = torch.isfinite(values)
    if not bool(finite.all()):
        values = values[finite]
    return values


def _stratified_sample(
    values: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    flattened = _finite_values(values)
    if flattened.numel() <= count:
        return flattened.to(device="cpu", dtype=torch.float32)
    offsets = torch.rand(count, generator=generator, dtype=torch.float64)
    positions = (torch.arange(count, dtype=torch.float64) + offsets) * (
        flattened.numel() / count
    )
    indices = positions.floor().clamp_max(flattened.numel() - 1).to(torch.long)
    return flattened.index_select(0, indices.to(flattened.device)).to(
        device="cpu",
        dtype=torch.float32,
    )


def _validate_percentages(
    percentiles: Sequence[float],
    tail_percentages: Sequence[float],
) -> None:
    if not percentiles:
        raise ValueError("At least one percentile candidate is required.")
    if any(not 0.0 < value <= 100.0 for value in percentiles):
        raise ValueError("Percentiles must be in the interval (0, 100].")
    if not tail_percentages:
        raise ValueError("At least one tail percentage is required.")
    if any(not 0.0 <= value < 50.0 for value in tail_percentages):
        raise ValueError("Tail percentages must be in the interval [0, 50).")
