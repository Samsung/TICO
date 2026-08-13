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

"""Reusable A/B/C/D/E fake-quantization ablation runner."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from torch import nn

from tico.quantization.analysis.inputs import ModelInput
from tico.quantization.analysis.metrics import evaluate_models
from tico.quantization.analysis.outputs import OutputAdapter
from tico.quantization.analysis.profile import QuantizationProfile
from tico.quantization.analysis.report import (
    QuantizationProfileResult,
    QuantizationReport,
)
from tico.quantization.analysis.selector import QuantizationBoundaries
from tico.quantization.wrapq.control import FakeQuantState, iter_quantization_sites


_DEFAULT_PROFILES = (
    QuantizationProfile.OUTPUT_ONLY,
    QuantizationProfile.WEIGHT_ONLY,
    QuantizationProfile.ACTIVATION_ONLY,
    QuantizationProfile.FULL,
    QuantizationProfile.INTERNAL_FULL,
)


class QuantizationAblation:
    """Compare a float model with selectable fake-quantization profiles."""

    def __init__(
        self,
        reference_model: nn.Module,
        quantized_model: nn.Module,
        *,
        boundaries: QuantizationBoundaries,
        output_adapter: OutputAdapter | None = None,
    ) -> None:
        """Store models and quantization-domain definitions.

        The quantized model must already be prepared, calibrated, and converted
        to WrapQ quantization mode. The runner only changes the runtime
        fake-quantization switches owned by its observers.
        """
        self.reference_model = reference_model
        self.quantized_model = quantized_model
        self.boundaries = boundaries
        self.output_adapter = output_adapter

    def run(
        self,
        samples: Sequence[ModelInput],
        *,
        profiles: Iterable[QuantizationProfile] = _DEFAULT_PROFILES,
        metadata: dict[str, Any] | None = None,
    ) -> QuantizationReport:
        """Evaluate float parity and each requested quantization profile."""
        if not samples:
            raise ValueError("Quantization ablation requires at least one sample.")
        selected_profiles = tuple(profiles)
        if not selected_profiles:
            raise ValueError("At least one quantization profile is required.")
        all_sites = tuple(iter_quantization_sites(self.quantized_model))
        if not all_sites:
            raise ValueError("The candidate model does not contain WrapQ observers.")
        output_boundary_required = (
            QuantizationProfile.OUTPUT_ONLY in selected_profiles
            or QuantizationProfile.INTERNAL_FULL in selected_profiles
            or (
                QuantizationProfile.ACTIVATION_ONLY in selected_profiles
                and self.boundaries.activations is None
            )
        )
        output_boundary_matches = any(
            self.boundaries.outputs(site) for site in all_sites
        )
        if output_boundary_required and not output_boundary_matches:
            raise ValueError(
                "The output selector did not match any quantization sites."
            )

        with FakeQuantState(self.quantized_model) as state:
            state.set_all(False)
            float_parity = evaluate_models(
                self.reference_model,
                self.quantized_model,
                samples,
                output_adapter=self.output_adapter,
            )

            results: dict[QuantizationProfile, QuantizationProfileResult] = {}
            for profile in selected_profiles:
                selector = self.boundaries.selector_for(profile)
                state.set_all(False)
                enabled_sites = tuple(site.path for site in all_sites if selector(site))
                state.set_where(selector, True)
                outputs = evaluate_models(
                    self.reference_model,
                    self.quantized_model,
                    samples,
                    output_adapter=self.output_adapter,
                )
                results[profile] = QuantizationProfileResult(
                    profile=profile,
                    enabled_sites=enabled_sites,
                    outputs=outputs,
                )

        report_metadata = dict(metadata or {})
        report_metadata.setdefault("site_count", len(all_sites))
        report_metadata.setdefault("sample_count", len(samples))
        return QuantizationReport(
            float_parity=float_parity,
            profiles=results,
            metadata=report_metadata,
        )
