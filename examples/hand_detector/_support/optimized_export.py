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

"""Full-integer evaluation, Circle export, and manifest helpers."""

from __future__ import annotations

import hashlib
import json

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from examples.hand_detector._support.quantization import export_quantized_circle
from examples.hand_detector._support.verify_circle_layout import verify_circle_layout
from examples.hand_detector._support.verify_quantized_circle import (
    verify_quantized_circle,
)
from tico.quantization.analysis import evaluate_models, OutputAdapter
from tico.quantization.wrapq.control import FakeQuantState
from torch import nn


OutputMetrics = Mapping[str, Mapping[str, float | int | None]]


def evaluate_full_quantized_model(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    samples: Sequence[torch.Tensor],
    *,
    output_adapter: OutputAdapter,
) -> dict[str, dict[str, float | int | None]]:
    """Evaluate the deployment profile with every fake-quant site enabled."""
    if not samples:
        raise ValueError("Full-quantized evaluation requires at least one sample.")
    with FakeQuantState(candidate_model) as state:
        state.set_all(True)
        return evaluate_models(
            reference_model,
            candidate_model,
            samples,
            output_adapter=output_adapter,
        )


def export_full_integer_circle(
    candidate_model: nn.Module,
    output_path: str | Path,
    *,
    bit_width: int,
    verify: bool = True,
) -> dict[str, object]:
    """Export with every fake-quant site enabled and optionally verify the graph."""
    output = Path(output_path)
    if output.suffix != ".circle":
        raise ValueError(
            "The optimized hand-detector export must use a .circle suffix."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    candidate_model.to(device="cpu").eval()
    with FakeQuantState(candidate_model) as state:
        state.set_all(True)
        exported = export_quantized_circle(candidate_model, output)

    summary: dict[str, object] = {
        "path": str(exported),
        "size_bytes": exported.stat().st_size,
        "sha256": _sha256(exported),
        "profile": "D:full",
        "all_fake_quant_sites_enabled": True,
        "verification_skipped": not verify,
    }
    if verify:
        summary["quantization_verification"] = _json_value(
            verify_quantized_circle(exported, bit_width)
        )
        summary["layout_verification"] = _json_value(
            verify_circle_layout(exported)
        )
    return summary


def default_manifest_path(circle_path: str | Path) -> Path:
    """Return a sidecar manifest path for one Circle artifact."""
    path = Path(circle_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def build_export_manifest(
    *,
    bit_width: int,
    circle_summary: Mapping[str, object],
    optimization_report_path: str | Path,
    activation_report_path: str | Path | None,
    optimization_metadata: Mapping[str, object],
    baseline_internal_full: OutputMetrics,
    final_internal_full: OutputMetrics,
    final_full: OutputMetrics,
    steps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a reproducible manifest for one optimized Circle artifact."""
    accepted_windows: list[dict[str, object]] = []
    for step in steps:
        result = step.get("adaround")
        window = step.get("window")
        if not isinstance(result, Mapping) or not isinstance(window, Mapping):
            raise TypeError(
                "AdaRound report steps must contain window and result maps."
            )
        if bool(result.get("accepted")):
            accepted_windows.append(
                {
                    "name": str(window.get("name")),
                    "best_step": int(result.get("best_step", 0)),
                    "weight_groups": list(result.get("weight_groups", ())),
                    "weight_statistics": _json_value(
                        result.get("weight_statistics", ())
                    ),
                }
            )

    return {
        "schema_version": 1,
        "model": "mediapipe_palm_detector",
        "dtype": "uint8" if bit_width == 8 else "int16",
        "bit_width": bit_width,
        "circle_export_profile": "D:full",
        "evaluation_profiles": {
            "internal_full": (
                "E:internal-full; graph outputs remain float for numerical analysis"
            ),
            "full": (
                "D:full; every fake-quant site is enabled and matches Circle export"
            ),
        },
        "source": {
            "optimization_report": str(optimization_report_path),
            "activation_report": (
                str(activation_report_path)
                if activation_report_path is not None
                else None
            ),
        },
        "recipe": {
            "metadata": _json_value(optimization_metadata),
            "accepted_adaround_windows": accepted_windows,
        },
        "evaluation": {
            "baseline_internal_full": _json_value(baseline_internal_full),
            "final_internal_full": _json_value(final_internal_full),
            "final_full": _json_value(final_full),
        },
        "circle": _json_value(circle_summary),
    }


def write_export_manifest(
    path: str | Path,
    manifest: Mapping[str, object],
) -> Path:
    """Write one deterministic JSON sidecar and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_value(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value
