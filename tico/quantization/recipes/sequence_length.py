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

"""Independent context policies for calibration, eager evaluation, and export."""

from typing import Any, Mapping

from tico.quantization.recipes.context import RecipeContext


def positive_sequence_length(value: Any, *, field: str) -> int:
    """Validate a sequence length without silently coercing booleans or floats."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    return value


def model_context_length(model: Any) -> int | None:
    """Read the recorded text context capacity, never an export template size."""
    config = getattr(model, "config", None)
    if config is None:
        config = getattr(getattr(model, "wrapped", None), "config", None)
    text_config = getattr(config, "text_config", config)
    capacity = getattr(text_config, "max_position_embeddings", None)
    if capacity is None:
        return None
    return positive_sequence_length(capacity, field="model.max_position_embeddings")


def resolve_evaluation_max_seq_len(ctx: RecipeContext) -> int:
    """Resolve eager context independently of calibration and static export.

    Checkpoint evaluation keeps the freshly loaded source model's context as
    the default. A legacy checkpoint with a smaller recorded capacity must be
    evaluated with an explicit common cap or regenerated; changing its config
    cannot recover the original positional configuration.
    """
    requested = ctx.cfg.get("evaluation", {}).get("max_seq_len")
    capacity = model_context_length(ctx.model)
    if requested is None:
        requested = ctx.artifacts.get("pretrained_context_length") or capacity
    if requested is None:
        raise ValueError(
            "Cannot infer the model context length. Set evaluation.max_seq_len."
        )
    resolved = positive_sequence_length(requested, field="evaluation.max_seq_len")
    if capacity is not None and resolved > capacity:
        raise ValueError(
            f"evaluation.max_seq_len={resolved} exceeds the loaded model/checkpoint "
            f"context capacity ({capacity}). Use the same explicit evaluation cap "
            "for FP and Quant, or regenerate a checkpoint with the original "
            "model configuration. Do not increase checkpoint config values to "
            "bypass this check."
        )
    return resolved


def resolve_export_max_seq_len(ctx: RecipeContext) -> int:
    """Resolve static export capacity without reading calibration or evaluation."""
    capacity = model_context_length(ctx.model)
    requested = ctx.cfg.get("export", {}).get("max_seq_len")
    if requested is None:
        requested = min(2048, capacity) if capacity is not None else 2048
    resolved = positive_sequence_length(requested, field="export.max_seq_len")
    if capacity is not None and resolved > capacity:
        raise ValueError(
            f"export.max_seq_len={resolved} exceeds the model context capacity "
            f"({capacity})."
        )
    return resolved


def resolve_ppl_max_seq_len(
    ppl_cfg: Mapping[str, Any], *, evaluation_max_seq_len: int
) -> int:
    """Resolve an optional PPL window within the common evaluation context."""
    requested = ppl_cfg.get("max_seq_len")
    if requested is None:
        return evaluation_max_seq_len
    resolved = positive_sequence_length(requested, field="perplexity.max_seq_len")
    if resolved > evaluation_max_seq_len:
        raise ValueError("The PPL window must not exceed evaluation.max_seq_len.")
    return resolved
