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

"""Hand-detector joint DW/PW learnable-scale AdaRound integration."""

from __future__ import annotations

import re

from collections.abc import Sequence

import torch

from examples.hand_detector._support.multistart_reconstruction import (
    ReconstructionDataSplit,
)
from examples.hand_detector._support.reconstruction import (
    _find_detector,
    collect_reconstruction_cache,
    DetectorWindow,
    ReconstructionWindow,
)
from tico.quantization.algorithm.adaround import (
    JointAdaRoundConfig,
    JointAdaRoundObjective,
    JointAdaRoundRunner,
    JointAdaRoundWeightGroup,
)
from tico.quantization.analysis import evaluate_models, OutputAdapter
from tico.quantization.wrapq.control import (
    FakeQuantState,
    iter_quantization_sites,
    SiteRole,
)
from torch import nn


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


ALL_CONV_JOINT_GROUPS = (
    "stem",
    *(f"feature_block_{index:02d}" for index in range(30)),
    "regressors_low_resolution_head",
    "regressors_high_resolution_head",
)


PRIORITY_JOINT_GROUPS = (
    "stem",
    "feature_block_00",
    "feature_block_01",
    "feature_block_02",
    "feature_block_05",
    "feature_block_06",
    "feature_block_07",
    "feature_block_09",
    "feature_block_10",
    "feature_block_13",
    "feature_block_14",
    "feature_block_15",
    "feature_block_17",
    "feature_block_18",
    "feature_block_21",
    "feature_block_25",
    "regressors_low_resolution_head",
    "regressors_high_resolution_head",
)


def build_joint_window_weight_groups(
    model: nn.Module,
    window: ReconstructionWindow,
) -> tuple[JointAdaRoundWeightGroup, ...]:
    """Return regular and depthwise Conv weights owned by one window."""
    positions = frozenset(window.operation_positions)
    values: list[tuple[int, JointAdaRoundWeightGroup]] = []
    for site in iter_quantization_sites(model):
        if site.role is not SiteRole.PARAMETER or site.observer_name != "weight":
            continue
        match = _LAYER_PATTERN.search(site.module_path)
        if match is None:
            continue
        position = int(match.group(1))
        if position not in positions:
            continue
        wrapped = getattr(site.module, "module", None)
        if not isinstance(wrapped, nn.Conv2d):
            continue
        family = _conv_family(wrapped)
        values.append(
            (
                position,
                JointAdaRoundWeightGroup(
                    name=f"layer_{position:03d}_{family}",
                    site_path=site.path,
                    family=family,
                ),
            )
        )
    values.sort(key=lambda value: (value[0], value[1].site_path))
    groups = tuple(value[1] for value in values)
    if not groups:
        raise ValueError(
            f"Reconstruction window {window.name!r} contains no Conv2d weights."
        )
    return groups


def evaluate_full_quantized(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    samples: Sequence[torch.Tensor],
    *,
    output_adapter: OutputAdapter,
) -> dict[str, dict[str, float | int | None]]:
    """Evaluate the complete W8/A16 deployment fake-quant profile."""
    with FakeQuantState(candidate_model) as state:
        state.set_all(True)
        return evaluate_models(
            reference_model,
            candidate_model,
            samples,
            output_adapter=output_adapter,
        )


def run_hand_detector_joint_adaround(
    reference_model: nn.Module,
    candidate_model: nn.Module,
    *,
    data_split: ReconstructionDataSplit,
    evaluation_samples: Sequence[torch.Tensor],
    windows: Sequence[ReconstructionWindow],
    config: JointAdaRoundConfig,
    selection_objective: JointAdaRoundObjective,
    acceptance_objective: JointAdaRoundObjective,
    output_adapter: OutputAdapter,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Optimize joint Conv windows sequentially with held-out rollback."""
    baseline_selection = evaluate_full_quantized(
        reference_model,
        candidate_model,
        data_split.selection,
        output_adapter=output_adapter,
    )
    baseline_acceptance = evaluate_full_quantized(
        reference_model,
        candidate_model,
        data_split.acceptance,
        output_adapter=output_adapter,
    )
    baseline_evaluation = evaluate_full_quantized(
        reference_model,
        candidate_model,
        evaluation_samples,
        output_adapter=output_adapter,
    )
    steps: list[dict[str, object]] = []

    from examples.hand_detector._support.analysis import output_boundaries

    boundaries = output_boundaries(candidate_model)
    for step_index, window in enumerate(windows, start=1):
        weight_groups = build_joint_window_weight_groups(candidate_model, window)
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
            return evaluate_full_quantized(
                reference_model,
                candidate_model,
                data_split.selection,
                output_adapter=output_adapter,
            )

        def acceptance_evaluator():
            return evaluate_full_quantized(
                reference_model,
                candidate_model,
                data_split.acceptance,
                output_adapter=output_adapter,
            )

        result = JointAdaRoundRunner(config).reconstruct(
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
        after_evaluation = evaluate_full_quantized(
            reference_model,
            candidate_model,
            evaluation_samples,
            output_adapter=output_adapter,
        )
        steps.append(
            {
                "step": step_index,
                "window": window.to_dict(),
                "weight_groups": [
                    {
                        "name": group.name,
                        "site_path": group.site_path,
                        "family": group.family,
                    }
                    for group in weight_groups
                ],
                "joint_adaround": result.to_dict(),
                "selection_after": after_selection,
                "acceptance_after": after_acceptance,
                "evaluation_after": after_evaluation,
            }
        )

    return {
        "profile": "P2:W8/A16/reg-I16/cls-U8",
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


def _conv_family(module: nn.Conv2d) -> str:
    if (
        module.groups == module.in_channels
        and module.out_channels % module.in_channels == 0
    ):
        return "depthwise_conv"
    return "regular_conv"


JOINT_ADAROUND_CHECKPOINT_FORMAT = "tico_joint_dwpw_adaround_v1"


def save_joint_adaround_checkpoint(
    model: nn.Module,
    path,
    *,
    metadata: dict[str, object] | None = None,
) -> str:
    """Save finalized weights and non-persistent affine parameter qparams."""
    from pathlib import Path

    from tico.quantization.wrapq.observers.affine_base import AffineObserverBase

    output = Path(path)
    affine_qparams: dict[str, dict[str, torch.Tensor]] = {}
    for site in iter_quantization_sites(model):
        if not isinstance(site.observer, AffineObserverBase):
            continue
        scale, zero_point = site.observer.compute_qparams()
        affine_qparams[site.path] = {
            "scale": scale.detach().cpu().clone(),
            "zero_point": zero_point.detach().cpu().clone(),
        }
    payload = {
        "format": JOINT_ADAROUND_CHECKPOINT_FORMAT,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "affine_qparams": affine_qparams,
        "metadata": dict(metadata or {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return str(output)


def _materialize_inference_buffers(model: nn.Module) -> int:
    """Replace inference-mode buffers with ordinary mutable tensor clones.

    Calibration observers may rebind ``min_val`` or ``max_val`` while running
    under ``torch.inference_mode``. PyTorch then marks those registered buffers
    as inference tensors, and ``load_state_dict`` cannot copy into them outside
    inference mode. Replacing only such buffers preserves registration, device,
    dtype, persistence metadata, and values while restoring normal mutability.
    """
    count = 0
    for module in model.modules():
        for name, value in tuple(module._buffers.items()):
            if isinstance(value, torch.Tensor) and value.is_inference():
                module._buffers[name] = value.detach().clone()
                count += 1
    return count


def apply_joint_adaround_checkpoint(model: nn.Module, path) -> dict[str, object]:
    """Restore one finalized joint-AdaRound checkpoint onto a fresh P2 model."""
    from pathlib import Path

    from tico.quantization.wrapq.observers.affine_base import AffineObserverBase

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Joint AdaRound checkpoint must contain a mapping.")
    if payload.get("format") != JOINT_ADAROUND_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported joint AdaRound checkpoint format {payload.get('format')!r}."
        )
    state_dict = payload.get("model_state_dict")
    qparams = payload.get("affine_qparams")
    if not isinstance(state_dict, dict) or not isinstance(qparams, dict):
        raise TypeError("Joint AdaRound checkpoint is missing model or qparam data.")
    normalized_inference_buffer_count = _materialize_inference_buffers(model)
    model.load_state_dict(state_dict, strict=True)

    sites = {
        site.path: site
        for site in iter_quantization_sites(model)
        if isinstance(site.observer, AffineObserverBase)
    }
    missing = tuple(sorted(set(sites).difference(qparams)))
    extra = tuple(sorted(set(qparams).difference(sites)))
    if missing or extra:
        raise ValueError(
            "Joint AdaRound checkpoint affine-site mismatch: "
            f"missing={missing}, extra={extra}."
        )
    for path_value, site in sites.items():
        values = qparams[path_value]
        if not isinstance(values, dict):
            raise TypeError(f"Checkpoint qparams for {path_value!r} are invalid.")
        scale = values.get("scale")
        zero_point = values.get("zero_point")
        if not isinstance(scale, torch.Tensor) or not isinstance(
            zero_point,
            torch.Tensor,
        ):
            raise TypeError(f"Checkpoint qparams for {path_value!r} need tensors.")
        device = site.observer.min_val.device
        site.observer.load_qparams(
            scale.to(device=device),
            zero_point.to(device=device),
            lock=True,
        )
        site.observer.fake_quant_enabled = True
    metadata = payload.get("metadata")
    return {
        "path": str(source),
        "affine_site_count": len(sites),
        "normalized_inference_buffer_count": (normalized_inference_buffer_count),
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
    }
