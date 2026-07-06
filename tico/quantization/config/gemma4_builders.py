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


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``base`` recursively merged with ``overlay``."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _gemma4_text_attention_override(
    linear_weight: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for Gemma4TextAttention.

    HF structure:
      q_proj, q_norm, k_proj, k_norm, v_proj, v_norm, o_proj

    Some modules are absent for shared-KV or attention_k_eq_v variants. Extra
    override keys are harmless because wrappers only consume matching children.
    """
    result = _linear_tree_override(
        linear_weight,
        ("q_proj", "k_proj", "v_proj", "o_proj"),
    )

    norm_override = _weight_override(norm_weight)
    if norm_override:
        for name in ("q_norm", "k_norm", "v_norm"):
            result[name] = copy.deepcopy(norm_override)

    return result


def _gemma4_text_layer_override(
    linear_weight: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for one Gemma4TextDecoderLayer.

    Mirrors the dense/no-MoE E2B text layer structure in HF:
      self_attn
      mlp
      input_layernorm
      post_attention_layernorm
      pre_feedforward_layernorm
      post_feedforward_layernorm
      optional PLE modules
    """
    layer: Dict[str, Any] = {
        "self_attn": _gemma4_text_attention_override(linear_weight, norm_weight),
        "mlp": _linear_tree_override(
            linear_weight,
            ("gate_proj", "up_proj", "down_proj"),
        ),
    }

    norm_override = _weight_override(norm_weight)
    if norm_override:
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            # Optional PLE norm. Ignored when PLE is disabled.
            "post_per_layer_input_norm",
        ):
            layer[name] = copy.deepcopy(norm_override)

    # Optional PLE linears. Ignored when hidden_size_per_layer_input == 0.
    layer.update(
        _linear_tree_override(
            linear_weight,
            ("per_layer_input_gate", "per_layer_projection"),
        )
    )

    return layer


def _gemma4_text_model_override(
    *,
    num_text_layers: int,
    linear_weight: Optional[QuantSpec],
    embedding_weight: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for Gemma4TextModel.

    This tree is used directly by:
      - Gemma4TextModel
      - Gemma4ForCausalLM.model

    It is also nested below ``language_model`` for Gemma4Model and
    Gemma4ForConditionalGeneration.model.
    """
    text: Dict[str, Any] = {
        "embed_tokens": _weight_override(embedding_weight),
        "layers": {},
        "norm": _weight_override(norm_weight),
    }

    for idx in range(num_text_layers):
        text["layers"][str(idx)] = _gemma4_text_layer_override(
            linear_weight,
            norm_weight,
        )

    # Optional top-level PLE modules in Gemma4TextModel. These are ignored by
    # wrappers when PLE is disabled, but let one builder follow the HF structure.
    embedding_override = _weight_override(embedding_weight)
    if embedding_override:
        text["embed_tokens_per_layer"] = copy.deepcopy(embedding_override)

    linear_override = _weight_override(linear_weight)
    if linear_override:
        text["per_layer_model_projection"] = copy.deepcopy(linear_override)

    norm_override = _weight_override(norm_weight)
    if norm_override:
        text["per_layer_projection_norm"] = copy.deepcopy(norm_override)

    return text


def _gemma4_vision_attention_override(
    linear_weight: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for Gemma4VisionAttention."""
    result = _linear_tree_override(
        linear_weight,
        ("q_proj", "k_proj", "v_proj", "o_proj"),
    )

    norm_override = _weight_override(norm_weight)
    if norm_override:
        for name in ("q_norm", "k_norm", "v_norm"):
            result[name] = copy.deepcopy(norm_override)

    return result


def _gemma4_multimodal_embedder_override(
    *,
    linear_weight: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for Gemma4MultimodalEmbedder."""
    result: Dict[str, Any] = {
        "embedding_projection": _weight_override(linear_weight),
    }

    norm_override = _weight_override(norm_weight)
    if norm_override:
        result["embedding_pre_projection_norm"] = copy.deepcopy(norm_override)

    return result


def _gemma4_vision_model_override(
    *,
    num_vision_layers: int,
    linear_weight: Optional[QuantSpec],
    vision_patch_embed_weight: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for Gemma4VisionModel."""
    vision: Dict[str, Any] = {
        "patch_embedder": {
            "input_proj": _weight_override(vision_patch_embed_weight or linear_weight)
        },
        "encoder": {"layers": {}},
    }

    norm_override = _weight_override(norm_weight)
    for idx in range(num_vision_layers):
        layer = {
            "self_attn": _gemma4_vision_attention_override(linear_weight, norm_weight),
            "mlp": _linear_tree_override(
                linear_weight,
                ("gate_proj", "up_proj", "down_proj"),
            ),
        }

        if norm_override:
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                layer[name] = copy.deepcopy(norm_override)

        vision["encoder"]["layers"][str(idx)] = layer
    return vision


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

    text_model_overrides = _gemma4_text_model_override(
        num_text_layers=num_text_layers,
        linear_weight=linear_weight,
        embedding_weight=embedding_weight,
        norm_weight=norm_weight,
    )
    vision_model_overrides = _gemma4_vision_model_override(
        num_vision_layers=num_vision_layers,
        linear_weight=linear_weight,
        vision_patch_embed_weight=vision_patch_embed_weight,
        norm_weight=norm_weight,
    )

    multimodal_embedder_overrides = _gemma4_multimodal_embedder_override(
        linear_weight=linear_weight,
        norm_weight=norm_weight,
    )

    # Follow the original Hugging Face structures while keeping one public API:
    #
    # Gemma4TextModel:
    #   embed_tokens / layers / norm / ...
    #
    # Gemma4ForCausalLM:
    #   model.(embed_tokens / layers / norm / ...)
    #
    # Gemma4Model:
    #   language_model.(embed_tokens / layers / norm / ...)
    #   vision_tower / embed_vision
    #
    # Gemma4ForConditionalGeneration:
    #   model.language_model.(embed_tokens / layers / norm / ...)
    #   model.vision_tower / model.embed_vision
    #   lm_head
    gemma4_model_overrides: Dict[str, Any] = {
        "language_model": copy.deepcopy(text_model_overrides),
        "vision_tower": copy.deepcopy(vision_model_overrides),
        "embed_vision": copy.deepcopy(multimodal_embedder_overrides),
    }

    model_like_overrides = _deep_merge(
        text_model_overrides,
        gemma4_model_overrides,
    )
    overrides: Dict[str, Any] = _deep_merge(
        model_like_overrides,
        {"model": model_like_overrides},
    )
    overrides["lm_head"] = _weight_override(lm_head_weight or linear_weight)

    return PTQConfig(
        activation=activation,
        weight=weight,
        overrides=overrides,
        model_args=dict(model_args or {}),
        strict_wrap=strict_wrap,
    )
