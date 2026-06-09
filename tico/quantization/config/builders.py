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
from typing import Any, Dict, Mapping, Optional, Tuple

from tico.quantization.config.llama_attention import (
    DEFAULT_EXECUTION_PROFILE,
    ExecutionProfile,
    normalize_execution_profile,
)
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, QuantSpec
from tico.quantization.wrapq.dtypes import DType


_RMSNORM_ACTIVATION_OBSERVERS = ("act_in", "act_out")
_LAYERNORM_ACTIVATION_OBSERVERS = (
    "act_in",
    "mean",
    "centered",
    "square",
    "var",
    "eps",
    "add_eps",
    "inv_std",
    "norm",
    "affine_mul",
    "affine_add",
    "act_out",
)


def _default_builder_activation() -> QuantSpec:
    """Return the default builder activation spec."""
    return affine(DType.int(16))


def _default_builder_weight() -> QuantSpec:
    """Return the default builder parameter spec."""
    return affine(DType.int(16))


def _set_nested_override(
    root: Dict[str, Any],
    path: Tuple[str, ...],
    value: Dict[str, Any],
) -> None:
    """Set an override value at a nested path."""
    current = root
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = copy.deepcopy(value)


def _build_weight_override(weight: Optional[QuantSpec]) -> Dict[str, Any]:
    """Build a parameter override dictionary for a module weight."""
    if weight is None:
        return {}
    return {
        "weight": weight.to_kwargs(
            "weight",
            context="weight",
            mark_replace=True,
        )
    }


def _build_bias_override(weight: Optional[QuantSpec]) -> Dict[str, Any]:
    """Build a parameter override dictionary for a module bias."""
    if weight is None:
        return {}
    return {
        "bias": weight.to_kwargs(
            "bias",
            context="bias",
            mark_replace=True,
        )
    }


def _build_activation_overrides(
    activation: Optional[QuantSpec],
    observer_names: tuple[str, ...],
) -> Dict[str, Any]:
    """Build per-observer activation overrides from a role spec."""
    if activation is None:
        return {}
    return {
        obs_name: activation.to_kwargs(
            obs_name,
            context=obs_name,
            mark_replace=True,
        )
        for obs_name in observer_names
    }


def _build_norm_override(
    *,
    norm: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for an RMSNorm-style module."""
    override: Dict[str, Any] = {}
    override.update(_build_activation_overrides(norm, _RMSNORM_ACTIVATION_OBSERVERS))
    override.update(_build_weight_override(norm_weight))
    override.update(_build_bias_override(norm_weight))
    return override


def _build_llama_layer_overrides(
    *,
    linear_weight: Optional[QuantSpec],
    norm: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build per-layer overrides for a Llama decoder block."""
    layer_overrides: Dict[str, Any] = {}

    linear_override = _build_weight_override(linear_weight)
    if linear_override:
        _set_nested_override(layer_overrides, ("self_attn", "q_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "k_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "v_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "o_proj"), linear_override)

        _set_nested_override(layer_overrides, ("mlp", "gate_proj"), linear_override)
        _set_nested_override(layer_overrides, ("mlp", "up_proj"), linear_override)
        _set_nested_override(layer_overrides, ("mlp", "down_proj"), linear_override)

    norm_override = _build_norm_override(norm=norm, norm_weight=norm_weight)
    if norm_override:
        _set_nested_override(layer_overrides, ("input_layernorm",), norm_override)
        _set_nested_override(
            layer_overrides, ("post_attention_layernorm",), norm_override
        )

    return layer_overrides


def _build_llama_overrides(
    *,
    num_hidden_layers: int,
    linear_weight: Optional[QuantSpec],
    embedding_weight: Optional[QuantSpec],
    lm_head_weight: Optional[QuantSpec],
    spin_rotation_weight: Optional[QuantSpec],
    norm: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build PTQ overrides for a Llama-style causal LM."""
    overrides: Dict[str, Any] = {"model": {"layers": {}}}

    embedding_override = _build_weight_override(embedding_weight)
    if embedding_override:
        _set_nested_override(overrides, ("model", "embed_tokens"), embedding_override)

    lm_head_override = _build_weight_override(lm_head_weight)
    if lm_head_override:
        overrides["lm_head"] = lm_head_override

    spin_rotation_override = _build_weight_override(spin_rotation_weight)
    if spin_rotation_override:
        _set_nested_override(
            overrides,
            ("model", "rotate_embedding"),
            spin_rotation_override,
        )
        _set_nested_override(overrides, ("rotate_lm_head",), spin_rotation_override)

    final_norm_override = _build_norm_override(norm=norm, norm_weight=norm_weight)
    if final_norm_override:
        _set_nested_override(overrides, ("model", "norm"), final_norm_override)

    for layer_idx in range(num_hidden_layers):
        overrides["model"]["layers"][str(layer_idx)] = _build_llama_layer_overrides(
            linear_weight=linear_weight,
            norm=norm,
            norm_weight=norm_weight,
        )

    return overrides


def build_llm_ptq_config(
    *,
    model_type: str,
    num_hidden_layers: int,
    activation: Optional[QuantSpec] = None,
    weight: Optional[QuantSpec] = None,
    linear_weight: Optional[QuantSpec] = None,
    embedding_weight: Optional[QuantSpec] = None,
    lm_head_weight: Optional[QuantSpec] = None,
    spin_rotation_weight: Optional[QuantSpec] = None,
    norm: Optional[QuantSpec] = None,
    norm_weight: Optional[QuantSpec] = None,
    strict_wrap: bool = True,
    profile: ExecutionProfile = DEFAULT_EXECUTION_PROFILE,
) -> PTQConfig:
    """
    Build a PTQConfig for an LLM using QuantSpec-based policies.

    Args:
        model_type: Model family used to select override generation logic.
        num_hidden_layers: Number of decoder layers in the model.
        activation: Default spec for activation-like observers.
        weight: Default spec for parameter-like observers.
        linear_weight: Weight spec for decoder-layer linear projections.
        embedding_weight: Weight spec for input embedding weights.
        lm_head_weight: Weight spec for the output projection.
        spin_rotation_weight: Weight spec for SpinLlama rotation matrices.
        norm: Activation spec for norm module internals.
        norm_weight: Weight spec for norm affine parameters.
        strict_wrap: If True, unsupported modules raise during wrapping.
        profile: Llama execution profile stored in model_args.

    Returns:
        A PTQ configuration object ready to pass into prepare().
    """
    profile = normalize_execution_profile(
        profile,
        context="build_llm_ptq_config.profile",
    )

    activation = activation or _default_builder_activation()
    weight = weight or _default_builder_weight()

    if model_type == "llama":
        overrides = _build_llama_overrides(
            num_hidden_layers=num_hidden_layers,
            linear_weight=linear_weight,
            embedding_weight=embedding_weight,
            lm_head_weight=lm_head_weight,
            spin_rotation_weight=spin_rotation_weight,
            norm=norm,
            norm_weight=norm_weight,
        )
    else:
        raise NotImplementedError(
            f"Unsupported model_type: {model_type!r}. Currently supported: ['llama']"
        )

    return PTQConfig(
        activation=activation,
        weight=weight,
        overrides=overrides,
        model_args={"profile": profile},
        strict_wrap=strict_wrap,
    )


def _build_qwen3_vl_norm_override(
    *,
    norm: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for Qwen3-VL norm modules."""
    override: Dict[str, Any] = {}
    override.update(_build_activation_overrides(norm, _LAYERNORM_ACTIVATION_OBSERVERS))
    override.update(_build_weight_override(norm_weight))
    override.update(_build_bias_override(norm_weight))
    return override


def _build_qwen3_vl_vision_block_overrides(
    *,
    linear_weight: Optional[QuantSpec],
    norm: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build per-block overrides for a Qwen3-VL vision transformer block."""
    block_overrides: Dict[str, Any] = {}

    linear_override = _build_weight_override(linear_weight)
    if linear_override:
        _set_nested_override(block_overrides, ("attn", "qkv"), linear_override)
        _set_nested_override(block_overrides, ("attn", "proj"), linear_override)
        _set_nested_override(block_overrides, ("mlp", "linear_fc1"), linear_override)
        _set_nested_override(block_overrides, ("mlp", "linear_fc2"), linear_override)

    norm_override = _build_qwen3_vl_norm_override(norm=norm, norm_weight=norm_weight)
    if norm_override:
        _set_nested_override(block_overrides, ("norm1",), norm_override)
        _set_nested_override(block_overrides, ("norm2",), norm_override)

    return block_overrides


def _build_qwen3_vl_vision_merger_overrides(
    *,
    linear_weight: Optional[QuantSpec],
    norm: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build overrides for a Qwen3-VL vision patch merger."""
    merger_overrides: Dict[str, Any] = {}

    linear_override = _build_weight_override(linear_weight)
    if linear_override:
        _set_nested_override(merger_overrides, ("linear_fc1",), linear_override)
        _set_nested_override(merger_overrides, ("linear_fc2",), linear_override)

    norm_override = _build_qwen3_vl_norm_override(norm=norm, norm_weight=norm_weight)
    if norm_override:
        _set_nested_override(merger_overrides, ("norm",), norm_override)

    return merger_overrides


def _build_qwen3_vl_text_layer_overrides(
    *,
    linear_weight: Optional[QuantSpec],
    norm: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build per-layer overrides for a Qwen3-VL text decoder block."""
    layer_overrides: Dict[str, Any] = {}

    linear_override = _build_weight_override(linear_weight)
    if linear_override:
        _set_nested_override(layer_overrides, ("self_attn", "q_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "k_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "v_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "o_proj"), linear_override)

        _set_nested_override(layer_overrides, ("mlp", "gate_proj"), linear_override)
        _set_nested_override(layer_overrides, ("mlp", "up_proj"), linear_override)
        _set_nested_override(layer_overrides, ("mlp", "down_proj"), linear_override)

    norm_override = _build_qwen3_vl_norm_override(norm=norm, norm_weight=norm_weight)
    if norm_override:
        _set_nested_override(layer_overrides, ("input_layernorm",), norm_override)
        _set_nested_override(
            layer_overrides, ("post_attention_layernorm",), norm_override
        )
        _set_nested_override(layer_overrides, ("self_attn", "q_norm"), norm_override)
        _set_nested_override(layer_overrides, ("self_attn", "k_norm"), norm_override)

    return layer_overrides


def _build_qwen3_vl_overrides(
    *,
    num_vision_blocks: int,
    num_text_layers: int,
    num_deepstack_mergers: int,
    linear_weight: Optional[QuantSpec],
    vision_patch_embed_weight: Optional[QuantSpec],
    embedding_weight: Optional[QuantSpec],
    lm_head_weight: Optional[QuantSpec],
    spin_rotation_weight: Optional[QuantSpec],
    norm: Optional[QuantSpec],
    norm_weight: Optional[QuantSpec],
) -> Dict[str, Any]:
    """Build PTQ overrides for a full Qwen3-VL model."""
    overrides: Dict[str, Any] = {"model": {}}

    vision_overrides: Dict[str, Any] = {}

    patch_embed_override = _build_weight_override(vision_patch_embed_weight)
    if patch_embed_override:
        _set_nested_override(
            vision_overrides,
            ("patch_embed", "proj"),
            patch_embed_override,
        )

    vision_overrides["blocks"] = {}
    for block_idx in range(num_vision_blocks):
        vision_overrides["blocks"][
            str(block_idx)
        ] = _build_qwen3_vl_vision_block_overrides(
            linear_weight=linear_weight,
            norm=norm,
            norm_weight=norm_weight,
        )

    merger_override = _build_qwen3_vl_vision_merger_overrides(
        linear_weight=linear_weight,
        norm=norm,
        norm_weight=norm_weight,
    )
    if merger_override:
        vision_overrides["merger"] = merger_override

    if num_deepstack_mergers > 0:
        vision_overrides["deepstack_merger_list"] = {}
        deepstack_override = _build_qwen3_vl_vision_merger_overrides(
            linear_weight=linear_weight,
            norm=norm,
            norm_weight=norm_weight,
        )
        for merger_idx in range(num_deepstack_mergers):
            vision_overrides["deepstack_merger_list"][str(merger_idx)] = copy.deepcopy(
                deepstack_override
            )

    overrides["model"]["visual"] = vision_overrides

    text_overrides: Dict[str, Any] = {}

    embedding_override = _build_weight_override(embedding_weight)
    if embedding_override:
        _set_nested_override(text_overrides, ("embed_tokens",), embedding_override)

    spin_rotation_override = _build_weight_override(spin_rotation_weight)
    if spin_rotation_override:
        _set_nested_override(
            text_overrides,
            ("rotate_embedding",),
            spin_rotation_override,
        )
        _set_nested_override(overrides, ("rotate_lm_head",), spin_rotation_override)

    text_overrides["layers"] = {}
    for layer_idx in range(num_text_layers):
        text_overrides["layers"][str(layer_idx)] = _build_qwen3_vl_text_layer_overrides(
            linear_weight=linear_weight,
            norm=norm,
            norm_weight=norm_weight,
        )

    final_norm_override = _build_qwen3_vl_norm_override(
        norm=norm,
        norm_weight=norm_weight,
    )
    if final_norm_override:
        _set_nested_override(text_overrides, ("norm",), final_norm_override)

    overrides["model"]["language_model"] = text_overrides

    lm_head_override = _build_weight_override(lm_head_weight)
    if lm_head_override:
        overrides["lm_head"] = lm_head_override

    return overrides


def build_qwen3_vl_ptq_config(
    *,
    num_vision_blocks: int,
    num_text_layers: int,
    num_deepstack_mergers: int,
    model_args: Mapping[str, Any],
    activation: Optional[QuantSpec] = None,
    weight: Optional[QuantSpec] = None,
    linear_weight: Optional[QuantSpec] = None,
    vision_patch_embed_weight: Optional[QuantSpec] = None,
    embedding_weight: Optional[QuantSpec] = None,
    lm_head_weight: Optional[QuantSpec] = None,
    spin_rotation_weight: Optional[QuantSpec] = None,
    norm: Optional[QuantSpec] = None,
    norm_weight: Optional[QuantSpec] = None,
    strict_wrap: bool = True,
) -> PTQConfig:
    """Build a PTQConfig for the full Qwen3-VL model.

    Args:
        num_vision_blocks: Number of vision transformer blocks.
        num_text_layers: Number of text decoder layers.
        num_deepstack_mergers: Number of deepstack merger modules.
        model_args: Model-specific arguments forwarded to PTQConfig.
        activation: Default spec for activation-like observers.
        weight: Default spec for parameter-like observers.
        linear_weight: Weight spec for linear projections.
        vision_patch_embed_weight: Weight spec for vision patch embedding.
        embedding_weight: Weight spec for token embeddings.
        lm_head_weight: Weight spec for the language modeling head.
        spin_rotation_weight: Weight spec for Qwen3-VL SpinQuant runtime rotations.
        norm: Activation spec for norm module internals.
        norm_weight: Weight spec for norm affine parameters.
        strict_wrap: If True, unsupported modules raise during wrapping.

    Returns:
        A PTQ configuration object ready to pass into prepare().
    """
    activation = activation or _default_builder_activation()
    weight = weight or _default_builder_weight()

    overrides = _build_qwen3_vl_overrides(
        num_vision_blocks=num_vision_blocks,
        num_text_layers=num_text_layers,
        num_deepstack_mergers=num_deepstack_mergers,
        linear_weight=linear_weight,
        vision_patch_embed_weight=vision_patch_embed_weight,
        embedding_weight=embedding_weight,
        lm_head_weight=lm_head_weight,
        spin_rotation_weight=spin_rotation_weight,
        norm=norm,
        norm_weight=norm_weight,
    )

    return PTQConfig(
        activation=activation,
        weight=weight,
        overrides=overrides,
        strict_wrap=strict_wrap,
        model_args=model_args,
    )
