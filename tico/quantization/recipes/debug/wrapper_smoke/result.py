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

"""Result objects for wrapper smoke checks."""

from dataclasses import dataclass, field
from typing import Any

from tico.quantization.recipes.debug.wrapper_smoke.utils import format_metric


@dataclass
class WrapperSmokeResult:
    """Structured result for one wrapper smoke run."""

    case: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a JSON-compatible dictionary."""
        return {
            "case": self.case,
            "passed": self.passed,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "messages": self.messages,
        }

    def raise_if_failed(self) -> None:
        """Raise a RuntimeError if the smoke run failed."""
        if not self.passed:
            details = "; ".join(self.messages) if self.messages else "no details"
            raise RuntimeError(f"Wrapper smoke case '{self.case}' failed: {details}")

    def format_text(self) -> str:
        """Return a human-readable summary for CLI output."""
        lines = [
            "┌───────────── Wrapper Smoke Summary ─────────────",
            f"│ Case             : {self.case}",
            f"│ Status           : {'PASS' if self.passed else 'FAIL'}",
            f"│ Mean |diff|      : {format_metric(self.metrics.get('mean_abs_diff'))}",
            f"│ Max |diff|       : {format_metric(self.metrics.get('max_abs_diff'))}",
            f"│ PEIR             : {format_metric(self.metrics.get('peir'))}",
            f"│ Shape match      : {self.metrics.get('shape_match')}",
            f"│ Quant finite     : {self.metrics.get('quant_finite')}",
            "└─────────────────────────────────────────────────",
        ]
        if self.artifacts:
            lines.append("Artifacts:")
            for name, path in sorted(self.artifacts.items()):
                lines.append(f"  - {name}: {path}")
        if self.messages:
            lines.append("Messages:")
            for message in self.messages:
                lines.append(f"  - {message}")
        return "\n".join(lines)
