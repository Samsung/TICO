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

"""Hand-detector-specific adapters for reusable quantization analysis."""

from __future__ import annotations

from typing import Any

from examples.hand_detector.hand_detector import HandDetector, NHWCInputAdapter
from tico.quantization.analysis import QuantizationBoundaries, SiteSelector
from tico.quantization.wrapq.observers.percentile import PercentileObserver

from torch import nn


OUTPUT_NAMES = ("regressors", "classifiers")


def output_boundaries(model: nn.Module) -> QuantizationBoundaries:
    """Select the final output-domain observers from a prepared detector."""
    detector, prefix = _find_detector(model)
    output_paths: list[str] = []
    output_tensors = set(detector.output_tensors)
    for layer_index, operation in enumerate(detector.operations):
        produced = {int(value) for value in operation["outputs"]}
        if produced & output_tensors:
            module_path = (
                f"{prefix}layers.{layer_index}" if prefix else f"layers.{layer_index}"
            )
            output_paths.append(module_path)
    if not output_paths:
        raise RuntimeError("No detector layer produces a configured model output.")
    selector = SiteSelector.fp_module_paths(
        *output_paths
    ) & SiteSelector.observer_names("act_out")
    return QuantizationBoundaries(outputs=selector)


def summarize_percentile_observers(model: nn.Module) -> list[dict[str, Any]]:
    """Return ranges and qparams for every percentile activation observer."""
    summaries: list[dict[str, Any]] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, PercentileObserver):
            continue
        scale, zero_point = module.compute_qparams()
        summaries.append(
            {
                "module": module_name,
                "observer_name": module.name,
                "observed_minimum": float(module.min_val.detach().cpu()),
                "observed_maximum": float(module.max_val.detach().cpu()),
                "clip_minimum": float(module.clip_min_val.detach().cpu()),
                "clip_maximum": float(module.clip_max_val.detach().cpu()),
                "scale": float(scale.detach().cpu()),
                "zero_point": int(zero_point.detach().cpu()),
                "sampled_value_count": module.sampled_value_count,
                "percentile": module.percentile,
            }
        )
    return summaries


def _find_detector(model: nn.Module) -> tuple[HandDetector, str]:
    if isinstance(model, NHWCInputAdapter):
        return model.detector, "detector."
    if isinstance(model, HandDetector):
        return model, ""
    detector = getattr(model, "detector", None)
    if isinstance(detector, HandDetector):
        return detector, "detector."
    raise TypeError("Expected HandDetector or NHWCInputAdapter.")
