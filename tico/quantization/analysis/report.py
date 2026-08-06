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

"""Serializable reports for quantization analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tico.quantization.analysis.profile import QuantizationProfile


MetricSummary = Mapping[str, float | int | None]


@dataclass(frozen=True)
class QuantizationProfileResult:
    """Store metrics and site counts for one ablation profile."""

    profile: QuantizationProfile
    enabled_sites: tuple[str, ...]
    outputs: Mapping[str, MetricSummary]

    @property
    def enabled_site_count(self) -> int:
        """Return the number of enabled fake-quantization sites."""
        return len(self.enabled_sites)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible profile representation."""
        return {
            "profile": self.profile.value,
            "label": self.profile.label,
            "enabled_site_count": self.enabled_site_count,
            "enabled_sites": list(self.enabled_sites),
            "outputs": {name: dict(metrics) for name, metrics in self.outputs.items()},
        }


@dataclass
class QuantizationReport:
    """Store float parity and standard quantization-ablation results."""

    float_parity: Mapping[str, MetricSummary]
    profiles: dict[QuantizationProfile, QuantizationProfileResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible report mapping."""
        return {
            "analysis": "quantization_ablation",
            "metadata": dict(self.metadata),
            "float_parity": {
                name: dict(metrics) for name, metrics in self.float_parity.items()
            },
            "profiles": {
                profile.value: result.to_dict()
                for profile, result in self.profiles.items()
            },
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the report as indented UTF-8 JSON and return its path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return output
