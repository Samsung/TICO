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

"""Shared FP/Quant eager quality evaluation setup and protocol reporting."""

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import torch

from tico.quantization.recipes.config import to_plain_data
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.sequence_length import (
    model_context_length,
    positive_sequence_length,
    resolve_evaluation_max_seq_len,
)
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


def validate_eager_evaluation_config(eval_cfg: Mapping[str, Any]) -> None:
    """Validate context, cache, and reporting settings for eager evaluation."""
    if eval_cfg.get("max_seq_len") is not None:
        positive_sequence_length(
            eval_cfg["max_seq_len"], field="evaluation.max_seq_len"
        )
    if not isinstance(eval_cfg.get("use_cache", True), bool):
        raise TypeError("evaluation.use_cache must be a boolean.")
    output_dir = eval_cfg.get("output_dir")
    if output_dir is not None and (
        not isinstance(output_dir, str) or not output_dir.strip()
    ):
        raise ValueError("evaluation.output_dir must be a non-empty string or null.")


def _package_versions() -> dict[str, str | None]:
    """Record installed evaluator versions without importing optional packages."""
    result: dict[str, str | None] = {}
    for package in (
        "tico",
        "torch",
        "transformers",
        "datasets",
        "lm-eval",
        "lmms-eval",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _component_identity(component: Any) -> dict[str, Any] | None:
    """Record processor/tokenizer identity and a digest of its chat template."""
    if component is None:
        return None
    template = getattr(component, "chat_template", None)
    template_hash = None
    if template is not None:
        template_hash = hashlib.sha256(
            json.dumps(to_plain_data(template), sort_keys=True).encode("utf-8")
        ).hexdigest()
    return {
        "class": type(component).__name__,
        "name_or_path": getattr(component, "name_or_path", None),
        "chat_template_sha256": template_hash,
    }


def prepare_eager_evaluation(ctx: RecipeContext) -> int:
    """Set a common inference policy and record the resolved quality protocol.

    This function never changes positional configuration, quantization modes,
    observer statistics, or frozen qparams. A model still collecting calibration
    statistics is rejected instead of silently calibrating on benchmark data.
    """
    eval_cfg = ctx.cfg.get("evaluation", {})
    validate_eager_evaluation_config(eval_cfg)
    max_seq_len = resolve_evaluation_max_seq_len(ctx)
    model = ctx.require_model()
    quant_modes: set[str] = set()
    if isinstance(model, torch.nn.Module):
        for module in model.modules():
            if isinstance(module, QuantModuleBase):
                if module._mode is Mode.CALIB:
                    raise ValueError(
                        "Eager evaluation requires a converted checkpoint; "
                        "a quantization module is still in calibration mode."
                    )
                quant_modes.add(str(module._mode))
            if getattr(module, "force_export", False) is True:
                raise ValueError(
                    "Eager evaluation cannot use a wrapper with force_export=True."
                )
        model.eval()

    use_cache = eval_cfg.get("use_cache", True)
    execution_model = getattr(model, "wrapped", model)
    config = getattr(execution_model, "config", None)
    for cache_config in (
        config,
        getattr(config, "text_config", None),
        getattr(execution_model, "generation_config", None),
    ):
        if cache_config is not None and hasattr(cache_config, "use_cache"):
            cache_config.use_cache = use_cache

    model_cfg = ctx.cfg.get("model", {})
    tokenizer = ctx.tokenizer
    if tokenizer is None:
        tokenizer = getattr(ctx.processor, "tokenizer", None)
    protocol = {
        "schema_version": 1,
        "purpose": "quantization_quality",
        "model": {
            "family": model_cfg.get("family", getattr(ctx.adapter, "family", None)),
            "name_or_path": model_cfg.get("name_or_path"),
            "loaded_name_or_path": getattr(config, "_name_or_path", None),
            "revision": model_cfg.get("revision"),
            "resolved_revision": getattr(config, "_commit_hash", None),
        },
        "tokenizer": _component_identity(tokenizer),
        "processor": _component_identity(ctx.processor),
        "context": {
            "requested": eval_cfg.get("max_seq_len"),
            "resolved": max_seq_len,
            "model_capacity": model_context_length(model),
            "pretrained_capacity": ctx.artifacts.get("pretrained_context_length"),
            "calibration_seq_len": ctx.cfg.get("calibration", {}).get("seq_len"),
            "export_max_seq_len": ctx.cfg.get("export", {}).get("max_seq_len"),
        },
        "use_cache": use_cache,
        "seed": ctx.cfg.get("runtime", {}).get("seed", 42),
        "device": str(ctx.device),
        "configured_dtype": str(ctx.dtype),
        "parameter_dtypes": (
            sorted({str(p.dtype) for p in model.parameters()})
            if isinstance(model, torch.nn.Module)
            else []
        ),
        "quantization_modes": sorted(quant_modes),
        "model_class": type(execution_model).__name__,
        "checkpoint": ctx.artifacts.get("evaluation_checkpoint"),
        "evaluation": to_plain_data(
            {key: value for key, value in eval_cfg.items() if key != "output_dir"}
        ),
        "versions": _package_versions(),
    }
    ctx.artifacts["evaluation_protocol"] = protocol
    output_dir = eval_cfg.get("output_dir")
    if output_dir:
        path = Path(output_dir) / "evaluation_protocol.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        ctx.artifacts["evaluation_protocol_path"] = str(path)
    return max_seq_len
