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

"""PTQConfig builders for the Gemma4 assistant (MTP draft) family."""

import copy
from typing import Any, Dict, Mapping, Optional

from tico.quantization.config.gemma4_builders import (
    _gemma4_text_layer_override,
    _weight_override,
)
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, QuantSpec
from tico.quantization.wrapq.dtypes import DType


def _default_activation() -> QuantSpec:
    """Return the default assistant activation spec (A16)."""
    return affine(DType.int(16))


def _default_weight() -> QuantSpec:
    """Return the default assistant weight spec (W8, safe_w8a16 profile)."""
    return affine(DType.uint(8))


def build_gemma4_assistant_ptq_config(
    *,
    num_hidden_layers: int,
    model_args: Optional[Mapping[str, Any]] = None,
    activation: Optional[QuantSpec] = None,
    weight: Optional[QuantSpec] = None,
    linear_weight: Optional[QuantSpec] = None,
    projection_weight: Optional[QuantSpec] = None,
    centroid_weight: Optional[QuantSpec] = None,
    lm_head_weight: Optional[QuantSpec] = None,
    norm_weight: Optional[QuantSpec] = None,
    strict_wrap: bool = True,
) -> PTQConfig:
    """Build a PTQConfig for ``QuantGemma4AssistantForCausalLM``.

    The default profile is acceptance-first ``safe_w8a16`` (int16 activations,
    uint8 weights). A compact ``w4a16`` experiment lowers ``linear_weight``
    while keeping the sensitive boundaries (``projection_weight``,
    ``centroid_weight``, ``lm_head_weight``) at W8 through their dedicated
    override arguments.

    Wrapper scopes mirror the HF assistant structure:
      pre_projection / post_projection / lm_head /
      masked_embedding.centroids / model.layers.{i} / model.norm

    ``model.embed_tokens`` is intentionally absent: the assistant core never
    uses it and its weight is tied to ``lm_head.weight``, which is the single
    quantized source of truth for the ordered sparse head.
    """
    if num_hidden_layers <= 0:
        raise ValueError(
            f"num_hidden_layers must be positive, got {num_hidden_layers}."
        )

    activation = activation or _default_activation()
    weight = weight or _default_weight()
    linear_weight = linear_weight or weight
    projection_weight = projection_weight or linear_weight
    centroid_weight = centroid_weight or linear_weight
    lm_head_weight = lm_head_weight or linear_weight

    projection_override = _weight_override(projection_weight)
    overrides: Dict[str, Any] = {
        "pre_projection": copy.deepcopy(projection_override),
        "post_projection": copy.deepcopy(projection_override),
        "lm_head": _weight_override(lm_head_weight),
        "masked_embedding": {"centroids": _weight_override(centroid_weight)},
        "model": {
            "layers": {
                str(idx): _gemma4_text_layer_override(linear_weight, norm_weight)
                for idx in range(num_hidden_layers)
            },
            "norm": _weight_override(norm_weight),
        },
    }

    normalized_model_args = dict(model_args or {})
    assistant_args = normalized_model_args.get("assistant")
    if isinstance(assistant_args, Mapping):
        kv_capacities = [
            int(assistant_args[key])
            for key in ("full_kv_length", "sliding_kv_length")
            if assistant_args.get(key) is not None
        ]
        if kv_capacities and "max_seq" not in normalized_model_args:
            # The Gemma4 attention wrapper sizes its bounded static templates
            # from model_args["max_seq"]; the capacity must cover the longest
            # per-layer-type KV span so explicit masks always fit.
            normalized_model_args["max_seq"] = max(kv_capacities)

    return PTQConfig(
        activation=activation,
        weight=weight,
        overrides=overrides,
        model_args=normalized_model_args,
        strict_wrap=strict_wrap,
    )
