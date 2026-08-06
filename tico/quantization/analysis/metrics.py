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

"""Streaming numerical-error metrics for model-output comparisons."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from tico.quantization.analysis.inputs import invoke_model, ModelInput
from tico.quantization.analysis.outputs import make_output_adapter, OutputAdapter


@dataclass
class TensorErrorMetrics:
    """Accumulate elementwise error statistics without retaining outputs."""

    count: int = 0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    max_absolute_error: float = 0.0
    dot_sum: float = 0.0
    reference_norm_sum: float = 0.0
    candidate_norm_sum: float = 0.0
    reference_absolute_sum: float = 0.0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        """Accumulate one pair of same-shaped tensors."""
        if reference.shape != candidate.shape:
            raise ValueError(
                f"Tensor shapes must match, got {tuple(reference.shape)} and "
                f"{tuple(candidate.shape)}."
            )
        reference64 = reference.detach().to(torch.float64).reshape(-1)
        candidate64 = candidate.detach().to(torch.float64).reshape(-1)
        difference = candidate64 - reference64
        self.count += difference.numel()
        self.absolute_error_sum += float(difference.abs().sum())
        self.squared_error_sum += float(torch.dot(difference, difference))
        self.max_absolute_error = max(
            self.max_absolute_error,
            float(difference.abs().max()) if difference.numel() else 0.0,
        )
        self.dot_sum += float(torch.dot(reference64, candidate64))
        self.reference_norm_sum += float(torch.dot(reference64, reference64))
        self.candidate_norm_sum += float(torch.dot(candidate64, candidate64))
        self.reference_absolute_sum += float(reference64.abs().sum())

    def summary(self) -> dict[str, float | int | None]:
        """Return stable JSON-compatible derived metrics."""
        if self.count == 0:
            raise RuntimeError("No tensor values were accumulated.")
        mae = self.absolute_error_sum / self.count
        mse = self.squared_error_sum / self.count
        cosine_denominator = math.sqrt(
            self.reference_norm_sum * self.candidate_norm_sum
        )
        cosine = self.dot_sum / cosine_denominator if cosine_denominator > 0 else 1.0
        signal = self.reference_norm_sum
        noise = self.squared_error_sum
        sqnr_db = (
            None if noise == 0.0 else 10.0 * math.log10(max(signal, 1e-300) / noise)
        )
        mean_reference_absolute = self.reference_absolute_sum / self.count
        relative_mae = mae / max(mean_reference_absolute, 1e-12)
        result: dict[str, float | int | None] = asdict(self)
        result.update(
            {
                "mae": mae,
                "mse": mse,
                "rmse": math.sqrt(mse),
                "cosine_similarity": cosine,
                "relative_mae": relative_mae,
                "sqnr_db": sqnr_db,
            }
        )
        return result


def metric_float(summary: Mapping[str, float | int | None], key: str) -> float:
    """Return one numeric metric value, rejecting missing or ``None`` entries."""
    value = summary.get(key)
    if not isinstance(value, (int, float)):
        raise KeyError(f"Metric {key!r} is missing or not numeric.")
    return float(value)


def evaluate_models(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    samples: Iterable[ModelInput],
    *,
    output_adapter: OutputAdapter | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Compare model outputs over an iterable of invocations."""
    adapter = output_adapter or make_output_adapter()
    accumulators: dict[str, TensorErrorMetrics] = {}
    sample_count = 0
    reference_model.eval()
    candidate_model.eval()

    with torch.inference_mode():
        for sample in samples:
            reference_outputs = dict(adapter(invoke_model(reference_model, sample)))
            candidate_outputs = dict(adapter(invoke_model(candidate_model, sample)))
            if tuple(reference_outputs) != tuple(candidate_outputs):
                raise ValueError(
                    "Reference and candidate output keys differ: "
                    f"{tuple(reference_outputs)} != {tuple(candidate_outputs)}."
                )
            if not accumulators:
                accumulators = {
                    name: TensorErrorMetrics() for name in reference_outputs
                }
            for name, reference in reference_outputs.items():
                accumulators[name].update(reference, candidate_outputs[name])
            sample_count += 1

    if sample_count == 0:
        raise ValueError("Evaluation requires at least one model sample.")
    return {name: metrics.summary() for name, metrics in accumulators.items()}
