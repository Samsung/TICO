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

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from tico.quantization.config.qwen3_vl_spinquant import Qwen3VLSpinQuantConfig


@dataclass
class Qwen3VLSpinQuantComponents:
    """
    Resolved Qwen3-VL module references required by SpinQuant.

    Attributes:
        language_model: Qwen3-VL text model.
        text_layers: Text decoder layers.
        lm_head: Final language modeling head.
        visual_deepstack_mergers: DeepStack visual merger modules.
    """

    language_model: nn.Module
    text_layers: nn.ModuleList
    lm_head: nn.Linear
    visual_deepstack_mergers: nn.ModuleList


def get_module_by_path(root: nn.Module, path: str) -> Any:
    """
    Resolve a dotted attribute path from a root module.

    Parameters:
        root: Root module.
        path: Dotted path such as ``"model.language_model.layers"``.

    Returns:
        The resolved object.

    Raises:
        AttributeError: If the path cannot be fully resolved.
        ValueError: If the path is empty.
    """
    if not path:
        raise ValueError("path must be a non-empty string.")

    current: Any = root
    for part in path.split("."):
        if isinstance(current, (nn.ModuleList, list, tuple)) and part.isdigit():
            index = int(part)
            try:
                current = current[index]
            except IndexError as exc:
                raise AttributeError(
                    f"Failed to resolve path {path!r}. Index {index} is out of range."
                ) from exc
            continue

        if not hasattr(current, part):
            raise AttributeError(
                f"Failed to resolve attribute path {path!r}. "
                f"Missing attribute {part!r} on object of type {type(current).__name__}."
            )
        current = getattr(current, part)

    return current


def require_linear_attr(module: nn.Module, attr_name: str) -> nn.Linear:
    """
    Return a required Linear attribute from a module.

    Parameters:
        module: Parent module.
        attr_name: Attribute name.

    Returns:
        The resolved Linear module.

    Raises:
        AttributeError: If the attribute is missing.
        TypeError: If the attribute is not an nn.Linear.
    """
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Expected attribute {attr_name!r} on module {type(module).__name__}."
        )

    value = getattr(module, attr_name)
    if not isinstance(value, nn.Linear):
        raise TypeError(
            f"Expected {attr_name!r} to be nn.Linear, got {type(value).__name__}."
        )

    return value


def resolve_qwen3_vl_spinquant_components(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> Qwen3VLSpinQuantComponents:
    """
    Resolve Qwen3-VL modules required by SpinQuant.

    Parameters:
        model: Target model.
        config: Qwen3-VL SpinQuant configuration.

    Returns:
        Resolved component references.

    Raises:
        TypeError: If a resolved module has an unexpected type.
    """
    language_model = get_module_by_path(model, config.language_model_attr)
    text_layers = get_module_by_path(model, config.text_layers_attr)
    lm_head = get_module_by_path(model, config.lm_head_attr)

    if config.fuse_deepstack_visual_outputs:
        visual_deepstack_mergers = get_module_by_path(
            model,
            config.visual_deepstack_mergers_attr,
        )
    else:
        visual_deepstack_mergers = nn.ModuleList()

    if not isinstance(language_model, nn.Module):
        raise TypeError(
            f"{config.language_model_attr!r} must resolve to nn.Module, "
            f"got {type(language_model).__name__}."
        )

    if not isinstance(text_layers, nn.ModuleList):
        raise TypeError(
            f"{config.text_layers_attr!r} must resolve to nn.ModuleList, "
            f"got {type(text_layers).__name__}."
        )

    if not isinstance(lm_head, nn.Linear):
        raise TypeError(
            f"{config.lm_head_attr!r} must resolve to nn.Linear, "
            f"got {type(lm_head).__name__}."
        )

    if not isinstance(visual_deepstack_mergers, nn.ModuleList):
        raise TypeError(
            f"{config.visual_deepstack_mergers_attr!r} must resolve to nn.ModuleList, "
            f"got {type(visual_deepstack_mergers).__name__}."
        )

    return Qwen3VLSpinQuantComponents(
        language_model=language_model,
        text_layers=text_layers,
        lm_head=lm_head,
        visual_deepstack_mergers=visual_deepstack_mergers,
    )


def is_tied_word_embedding(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> bool:
    """
    Return whether the Qwen3-VL input embedding and LM head share storage.

    Parameters:
        model: Target model.
        config: Qwen3-VL SpinQuant configuration.

    Returns:
        True if the two weights share the same data pointer.
    """
    components = resolve_qwen3_vl_spinquant_components(model, config)

    if not hasattr(components.language_model, "embed_tokens"):
        return False

    embed_tokens = components.language_model.embed_tokens
    if not isinstance(embed_tokens, nn.Embedding):
        return False

    return embed_tokens.weight.data_ptr() == components.lm_head.weight.data_ptr()


def assert_tied_word_embedding(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> None:
    """
    Validate that Qwen3-VL input embedding and LM head are tied.

    Parameters:
        model: Target model.
        config: Qwen3-VL SpinQuant configuration.

    Raises:
        ValueError: If the weights are not tied.
    """
    if not is_tied_word_embedding(model, config):
        raise ValueError(
            "Qwen3-VL SpinQuant assumes tied word embeddings, but "
            "`model.language_model.embed_tokens.weight` and `lm_head.weight` "
            "do not share storage."
        )


def validate_qwen3_vl_for_spinquant(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
    *,
    require_spin_runtime: bool = False,
) -> None:
    """
    Validate that a model exposes the modules required by Qwen3-VL SpinQuant.

    Parameters:
        model: Target model.
        config: Qwen3-VL SpinQuant configuration.
        require_spin_runtime: Whether to require added SpinQuant runtime layers.

    Raises:
        TypeError: If the input is not a module or a submodule has an invalid type.
        ValueError: If the model type or tied embedding assumption is invalid.
        AttributeError: If a required module is missing.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected an nn.Module, got {type(model).__name__}.")

    model_config = getattr(model, "config", None)
    model_type = getattr(model_config, "model_type", None)
    if model_type != "qwen3_vl":
        raise ValueError(
            "Qwen3-VL SpinQuant supports only Qwen3-VL dense models, "
            f"but got model_type={model_type!r}."
        )

    if not hasattr(model_config, "text_config"):
        raise ValueError("Qwen3-VL SpinQuant requires `model.config.text_config`.")

    components = resolve_qwen3_vl_spinquant_components(model, config)

    if not hasattr(components.language_model, "embed_tokens"):
        raise AttributeError("Expected language model to expose `embed_tokens`.")
    if not isinstance(components.language_model.embed_tokens, nn.Embedding):
        raise TypeError(
            "Expected language_model.embed_tokens to be nn.Embedding, "
            f"got {type(components.language_model.embed_tokens).__name__}."
        )

    if not hasattr(components.language_model, "norm"):
        raise AttributeError("Expected language model to expose final `norm`.")

    for layer_idx, layer in enumerate(components.text_layers):
        if not hasattr(layer, "self_attn"):
            raise AttributeError(f"Text layer {layer_idx} is missing `self_attn`.")
        if not hasattr(layer, "mlp"):
            raise AttributeError(f"Text layer {layer_idx} is missing `mlp`.")
        if not hasattr(layer, "input_layernorm"):
            raise AttributeError(
                f"Text layer {layer_idx} is missing `input_layernorm`."
            )
        if not hasattr(layer, "post_attention_layernorm"):
            raise AttributeError(
                f"Text layer {layer_idx} is missing `post_attention_layernorm`."
            )

        for attr_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            require_linear_attr(layer.self_attn, attr_name)

        for attr_name in ("gate_proj", "up_proj", "down_proj"):
            require_linear_attr(layer.mlp, attr_name)

    if config.fuse_deepstack_visual_outputs:
        for merger_idx, merger in enumerate(components.visual_deepstack_mergers):
            try:
                require_linear_attr(merger, "linear_fc2")
            except Exception as exc:
                raise type(exc)(
                    f"Invalid DeepStack merger {merger_idx}: {exc}"
                ) from exc

    assert_tied_word_embedding(model, config)

    if require_spin_runtime:
        require_linear_attr(components.language_model, "rotate_embedding")
        require_linear_attr(model, "rotate_lm_head")
