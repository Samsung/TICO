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

"""Hand-detector weight groups, activation replay, and AdaRound flow."""

from __future__ import annotations

import json
import re

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from tico.quantization.algorithm.adaround import (
    AdaRoundConfig,
    AdaRoundRunner,
    AdaRoundWeightGroup,
)
from tico.quantization.algorithm.block_reconstruction import ValidationObjective
from tico.quantization.analysis import OutputAdapter, QuantizationProfile
from tico.quantization.wrapq.control import iter_quantization_sites, SiteRole
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from torch import nn

from examples.hand_detector._support.analysis import output_boundaries
from examples.hand_detector._support.multistart_reconstruction import (
    ReconstructionDataSplit,
)
from examples.hand_detector._support.reconstruction import (
    _find_detector,
    collect_reconstruction_cache,
    DetectorWindow,
    evaluate_internal_full,
    ReconstructionWindow,
)


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def build_window_weight_groups(
    model: nn.Module,
    window: ReconstructionWindow,
) -> tuple[AdaRoundWeightGroup, ...]:
    """Return Conv2d weight sites owned by operations inside one window."""
    positions = frozenset(window.operation_positions)
    groups: list[AdaRoundWeightGroup] = []
    for site in iter_quantization_sites(model):
        if site.role is not SiteRole.PARAMETER or site.observer_name != "weight":
            continue
        match = _LAYER_PATTERN.search(site.module_path)
        if match is None or int(match.group(1)) not in positions:
            continue
        weight_module = getattr(site.module, "module", None)
        if not isinstance(weight_module, nn.Conv2d):
            continue
        groups.append(
            AdaRoundWeightGroup(
                name=site.module_path + ".weight",
                site_path=site.path,
            )
        )
    groups.sort(key=lambda group: group.site_path)
    if not groups:
        raise ValueError(
            f"Reconstruction window {window.name!r} contains no Conv2d weight sites."
        )
    return tuple(groups)


def apply_activation_reconstruction_report(
    model: nn.Module,
    report_path: Path,
    *,
    expected_percentile: float | None = None,
    expected_max_samples: int | None = None,
) -> dict[str, object]:
    """Replay accepted activation qparams from a reconstruction JSON report."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("Activation report metadata must be a mapping.")
    percentile = metadata.get("percentile", metadata.get("global_percentile"))
    if (
        expected_percentile is not None
        and percentile is not None
        and not math_isclose(float(percentile), expected_percentile)
    ):
        raise ValueError(
            f"Activation report percentile {percentile} does not match "
            f"{expected_percentile}."
        )
    max_samples = metadata.get("max_samples", metadata.get("observer_max_samples"))
    if (
        expected_max_samples is not None
        and max_samples is not None
        and int(max_samples) != expected_max_samples
    ):
        raise ValueError(
            f"Activation report max_samples {max_samples} does not match "
            f"{expected_max_samples}."
        )

    sites = {site.path: site for site in iter_quantization_sites(model)}
    committed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    accepted_steps: list[str] = []
    applied_site_paths: list[str] = []
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise TypeError("Activation report must contain a steps list.")
    for step in steps:
        if not isinstance(step, Mapping):
            raise TypeError("Every activation report step must be a mapping.")
        reconstruction = step.get("reconstruction")
        if not isinstance(reconstruction, Mapping):
            raise TypeError("Activation report step lacks reconstruction data.")
        if not bool(reconstruction.get("accepted", False)):
            continue
        qparams = reconstruction.get("qparams")
        observer_groups = step.get("observer_groups")
        if not isinstance(qparams, Mapping) or not isinstance(observer_groups, list):
            raise TypeError(
                "Accepted activation report steps need qparams and observer_groups."
            )
        window = step.get("window", {})
        window_name = (
            str(window.get("name", step.get("step", "unknown")))
            if isinstance(window, Mapping)
            else str(step.get("step", "unknown"))
        )
        for group in observer_groups:
            if not isinstance(group, Mapping):
                raise TypeError("Activation observer-group entries must be mappings.")
            name = str(group.get("name", ""))
            paths = group.get("site_paths")
            values = qparams.get(name)
            if (
                not name
                or not isinstance(paths, list)
                or not isinstance(values, Mapping)
            ):
                raise ValueError(
                    f"Malformed activation qparam group {name!r} in {window_name!r}."
                )
            scale = torch.as_tensor(values["scale"], dtype=torch.float32)
            zero_point = torch.as_tensor(values["zero_point"], dtype=torch.int)
            for site_path in paths:
                path = str(site_path)
                site = sites.get(path)
                if site is None:
                    raise KeyError(f"Unknown activation report site {path!r}.")
                if site.role is SiteRole.PARAMETER:
                    raise ValueError(
                        "Activation report unexpectedly targets parameter site "
                        f"{path!r}."
                    )
                observer = site.observer
                if not isinstance(observer, AffineObserverBase):
                    raise TypeError(f"Activation report site {path!r} is not affine.")
                identity = id(observer)
                previous = committed.get(identity)
                current = (scale, zero_point)
                if previous is not None:
                    if not torch.equal(previous[0], scale) or not torch.equal(
                        previous[1], zero_point
                    ):
                        raise RuntimeError(
                            "One activation observer received inconsistent replay "
                            "qparams."
                        )
                    continue
                device = observer.min_val.device
                observer.load_qparams(
                    scale.to(device=device),
                    zero_point.to(device=device),
                    lock=True,
                )
                committed[identity] = current
                applied_site_paths.append(path)
        accepted_steps.append(window_name)

    if not accepted_steps:
        raise ValueError("Activation report contains no accepted reconstruction steps.")
    return {
        "path": str(report_path),
        "accepted_steps": accepted_steps,
        "accepted_step_count": len(accepted_steps),
        "applied_site_count": len(applied_site_paths),
        "applied_site_paths": applied_site_paths,
        "source_analysis": payload.get("analysis"),
    }


def run_hand_detector_adaround(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *,
    data_split: ReconstructionDataSplit,
    evaluation_samples: Sequence[torch.Tensor],
    windows: Sequence[ReconstructionWindow],
    config: AdaRoundConfig,
    selection_objective: ValidationObjective,
    acceptance_objective: ValidationObjective,
    output_adapter: OutputAdapter,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Optimize Conv2d rounding window by window with held-out rollback."""
    boundaries = output_boundaries(candidate_model)
    selector = boundaries.selector_for(QuantizationProfile.INTERNAL_FULL)
    site_count = sum(
        selector(site) for site in iter_quantization_sites(candidate_model)
    )
    baseline_selection = evaluate_internal_full(
        reference_model,
        candidate_model,
        data_split.selection,
        boundaries=boundaries,
        output_adapter=output_adapter,
    )
    baseline_acceptance = evaluate_internal_full(
        reference_model,
        candidate_model,
        data_split.acceptance,
        boundaries=boundaries,
        output_adapter=output_adapter,
    )
    baseline_evaluation = evaluate_internal_full(
        reference_model,
        candidate_model,
        evaluation_samples,
        boundaries=boundaries,
        output_adapter=output_adapter,
    )
    steps: list[dict[str, object]] = []

    for step_index, window in enumerate(windows, start=1):
        weight_groups = build_window_weight_groups(candidate_model, window)
        train_cache = collect_reconstruction_cache(
            reference_model,
            candidate_model,
            data_split.train,
            window,
            boundaries=boundaries,
        )
        selection_cache = collect_reconstruction_cache(
            reference_model,
            candidate_model,
            data_split.selection,
            window,
            boundaries=boundaries,
        )
        block = DetectorWindow(_find_detector(candidate_model), window)

        def selection_evaluator():
            return evaluate_internal_full(
                reference_model,
                candidate_model,
                data_split.selection,
                boundaries=boundaries,
                output_adapter=output_adapter,
            )

        def acceptance_evaluator():
            return evaluate_internal_full(
                reference_model,
                candidate_model,
                data_split.acceptance,
                boundaries=boundaries,
                output_adapter=output_adapter,
            )

        result = AdaRoundRunner(config).reconstruct(
            block_name=window.name,
            observer_model=candidate_model,
            block=block,
            cache=train_cache,
            selection_cache=selection_cache,
            weight_groups=weight_groups,
            selection_evaluator=selection_evaluator,
            selection_objective=selection_objective,
            acceptance_evaluator=acceptance_evaluator,
            acceptance_objective=acceptance_objective,
            device=device,
        )
        after_selection = selection_evaluator()
        after_acceptance = acceptance_evaluator()
        after_evaluation = evaluate_internal_full(
            reference_model,
            candidate_model,
            evaluation_samples,
            boundaries=boundaries,
            output_adapter=output_adapter,
        )
        steps.append(
            {
                "step": step_index,
                "window": window.to_dict(),
                "weight_groups": [
                    {"name": group.name, "site_path": group.site_path}
                    for group in weight_groups
                ],
                "adaround": result.to_dict(),
                "selection_after": after_selection,
                "acceptance_after": after_acceptance,
                "evaluation_after": after_evaluation,
            }
        )

    return {
        "profile": "E:internal-full",
        "enabled_site_count": site_count,
        "data_split": data_split.to_dict(),
        "baseline_selection": baseline_selection,
        "baseline_acceptance": baseline_acceptance,
        "baseline_evaluation": baseline_evaluation,
        "steps": steps,
        "final_selection": (
            steps[-1]["selection_after"] if steps else baseline_selection
        ),
        "final_acceptance": (
            steps[-1]["acceptance_after"] if steps else baseline_acceptance
        ),
        "final_evaluation": (
            steps[-1]["evaluation_after"] if steps else baseline_evaluation
        ),
    }


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-9 * max(1.0, abs(left), abs(right))
