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

import copy
from typing import Any, Dict, Mapping, Optional

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, QuantSpec
from tico.quantization.wrapq.dtypes import DType


def _default_activation() -> QuantSpec:
    """Return the default Gemma4 activation quantization spec."""
    return affine(DType.int(16))


def _default_weight() -> QuantSpec:
    """Return the default Gemma4 weight quantization spec."""
    return affine(DType.int(16))


def _weight_override(spec: Optional[QuantSpec]) -> Dict[str, Any]:
    """Build a weight observer override for a module."""
    if spec is None:
        return {}
    return {"weight": spec.to_kwargs("weight", context="weight", mark_replace=True)}


def _set_nested(
    root: Dict[str, Any], path: tuple[str, ...], value: Dict[str, Any]
) -> None:
    """Set a nested override dictionary."""
    curr = root
    for key in path[:-1]:
        curr = curr.setdefault(key, {})
    curr[path[-1]] = copy.deepcopy(value)


def _linear_tree_override(
    linear_weight: Optional[QuantSpec], names: tuple[str, ...]
) -> Dict[str, Any]:
    """Build a tree of linear weight overrides."""
    root: Dict[str, Any] = {}
    override = _weight_override(linear_weight)
    if not override:
        return root
    for name in names:
        _set_nested(root, (name,), override)
    return root


def build_gemma4_e2b_ptq_config(
    *,
    num_text_layers: int,
    num_vision_layers: int,
    model_args: Optional[Mapping[str, Any]] = None,
    activation: Optional[QuantSpec] = None,
    weight: Optional[QuantSpec] = None,
    linear_weight: Optional[QuantSpec] = None,
    embedding_weight: Optional[QuantSpec] = None,
    lm_head_weight: Optional[QuantSpec] = None,
    vision_patch_embed_weight: Optional[QuantSpec] = None,
    norm_weight: Optional[QuantSpec] = None,
    strict_wrap: bool = True,
) -> PTQConfig:
    """Build a PTQConfig for Gemma4 E2B static image-text runtime.

    The returned config is intentionally conservative. It assigns default
    activation and weight specs globally, then adds explicit weight overrides for
    large module families so GPTQ/PTQ handoff can target predictable names.
    """

    activation = activation or _default_activation()
    weight = weight or _default_weight()
    linear_weight = linear_weight or weight

    overrides: Dict[str, Any] = {"model": {"model": {}}}

    model_overrides = overrides["model"]["model"]
    model_overrides["language_model"] = {
        "embed_tokens": _weight_override(embedding_weight),
        "layers": {},
        "norm": _weight_override(norm_weight),
    }
    for idx in range(num_text_layers):
        model_overrides["language_model"]["layers"][str(idx)] = {
            "self_attn": _linear_tree_override(
                linear_weight,
                ("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
            "mlp": _linear_tree_override(
                linear_weight,
                ("gate_proj", "up_proj", "down_proj"),
            ),
        }

    model_overrides["vision_tower"] = {
        "patch_embedder": {
            "input_proj": _weight_override(vision_patch_embed_weight or linear_weight)
        },
        "encoder": {"layers": {}},
    }
    for idx in range(num_vision_layers):
        model_overrides["vision_tower"]["encoder"]["layers"][str(idx)] = {
            "self_attn": _linear_tree_override(
                linear_weight,
                ("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
            "mlp": _linear_tree_override(
                linear_weight,
                ("gate_proj", "up_proj", "down_proj"),
            ),
        }

    model_overrides["embed_vision"] = {
        "embedding_projection": _weight_override(linear_weight),
    }
    overrides["model"]["lm_head"] = _weight_override(lm_head_weight or linear_weight)

    return PTQConfig(
        activation=activation,
        weight=weight,
        overrides=overrides,
        model_args=dict(model_args or {}),
        strict_wrap=strict_wrap,
    )
