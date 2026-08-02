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
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from tico.quantization.config.gemma4_gptq import Gemma4GPTQConfig

# Re-export shared utility functions that are identical between Qwen3-VL and
# Gemma4.  These cover tensor-tree manipulation, cache management, and
# quantizable-layer discovery.
from tico.quantization.algorithm.qwen3_vl_gptq.utils import (  # noqa: F401
    _QUANTIZABLE_LAYER_TYPES,
    find_layers,
    get_quantizable_layers,
    build_module_name_map,
    extract_primary_output,
    is_tensor_collection,
    tree_map_tensors,
    detach_clone_tree,
    move_tensor_tree,
    maybe_move_cache_to_cpu,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
    clone_batch_kwargs,
    prepare_model_kwargs,
    split_model_inputs,
    _num_cached_batches,
    append_batch_to_cache,
    iter_cached_batches,
    get_first_parameter_device,
    get_first_parameter_dtype,
    get_module_by_path,
)


@dataclass
class Gemma4Components:
    """
    Resolved Gemma4 module references.

    Attributes:
        vision_tower: Vision tower root module (Gemma4VisionModel).
        vision_patch_embed: Vision patch embedder (contains ``input_proj`` Linear).
        vision_encoder: Vision encoder root module.
        vision_encoder_layers: Vision encoder layers (ModuleList).
        vision_pooler: Vision pooler module (spatial pooling).
        multimodal_embedder: Multimodal embedder (``embed_vision``) that projects
            vision features into text hidden space.
        language_model: Text model root module.
        text_layers: Text decoder layers.
        lm_head: Final language modeling head.
    """

    vision_tower: nn.Module
    vision_patch_embed: nn.Module
    vision_encoder: nn.Module
    vision_encoder_layers: nn.ModuleList
    vision_pooler: nn.Module
    multimodal_embedder: nn.Module
    language_model: nn.Module
    text_layers: nn.ModuleList
    lm_head: nn.Module


def resolve_gemma4_components(
    model: nn.Module,
    config: Gemma4GPTQConfig,
) -> Gemma4Components:
    """
    Resolve key Gemma4 submodules from a model using config-defined paths.

    Args:
        model: Target Gemma4 model.
        config: Gemma4 GPTQ configuration.

    Returns:
        A dataclass containing resolved module references.

    Raises:
        TypeError: If a resolved object has an unexpected type.
    """
    vision_tower = get_module_by_path(model, config.vision_tower_attr)
    vision_patch_embed = get_module_by_path(model, config.vision_patch_embed_attr)
    vision_encoder = get_module_by_path(model, config.vision_encoder_attr)
    vision_encoder_layers = get_module_by_path(
        model, config.vision_encoder_layers_attr
    )
    vision_pooler = get_module_by_path(model, config.vision_pooler_attr)
    multimodal_embedder = get_module_by_path(
        model, config.multimodal_embedder_attr
    )
    language_model = get_module_by_path(model, config.language_model_attr)
    text_layers = get_module_by_path(model, config.text_layers_attr)
    lm_head = get_module_by_path(model, config.lm_head_attr)

    if not isinstance(vision_encoder_layers, nn.ModuleList):
        raise TypeError(
            f"{config.vision_encoder_layers_attr!r} must resolve to nn.ModuleList. "
            f"Got {type(vision_encoder_layers).__name__}."
        )
    if not isinstance(text_layers, nn.ModuleList):
        raise TypeError(
            f"{config.text_layers_attr!r} must resolve to nn.ModuleList. "
            f"Got {type(text_layers).__name__}."
        )
    if not isinstance(vision_tower, nn.Module):
        raise TypeError(
            f"vision_tower must be nn.Module. Got {type(vision_tower).__name__}."
        )
    if not isinstance(vision_patch_embed, nn.Module):
        raise TypeError(
            f"vision_patch_embed must be nn.Module. "
            f"Got {type(vision_patch_embed).__name__}."
        )
    if not isinstance(vision_encoder, nn.Module):
        raise TypeError(
            f"vision_encoder must be nn.Module. "
            f"Got {type(vision_encoder).__name__}."
        )
    if not isinstance(vision_pooler, nn.Module):
        raise TypeError(
            f"vision_pooler must be nn.Module. "
            f"Got {type(vision_pooler).__name__}."
        )
    if not isinstance(multimodal_embedder, nn.Module):
        raise TypeError(
            f"multimodal_embedder must be nn.Module. "
            f"Got {type(multimodal_embedder).__name__}."
        )
    if not isinstance(language_model, nn.Module):
        raise TypeError(
            f"language_model must be nn.Module. "
            f"Got {type(language_model).__name__}."
        )
    if not isinstance(lm_head, nn.Module):
        raise TypeError(
            f"lm_head must be nn.Module. Got {type(lm_head).__name__}."
        )

    return Gemma4Components(
        vision_tower=vision_tower,
        vision_patch_embed=vision_patch_embed,
        vision_encoder=vision_encoder,
        vision_encoder_layers=vision_encoder_layers,
        vision_pooler=vision_pooler,
        multimodal_embedder=multimodal_embedder,
        language_model=language_model,
        text_layers=text_layers,
        lm_head=lm_head,
    )


def should_quantize_vision_stage(
    config: Gemma4GPTQConfig,
    *,
    stage: str,
) -> bool:
    """
    Check whether a specific vision-side stage is enabled.

    Args:
        config: Gemma4 GPTQ configuration.
        stage: One of
            ``"patch_embed"``, ``"blocks"``, ``"pooler"``,
            ``"multimodal_embedder"``.

    Returns:
        ``True`` if the stage should be quantized.

    Raises:
        ValueError: If ``stage`` is unknown.
    """
    if not config.quantize_vision:
        return False

    if stage == "patch_embed":
        return config.quantize_vision_patch_embed
    if stage == "blocks":
        return config.quantize_vision_blocks
    if stage == "pooler":
        return config.quantize_vision_pooler
    if stage == "multimodal_embedder":
        return config.quantize_multimodal_embedder

    raise ValueError(f"Unknown vision stage: {stage!r}")


def should_quantize_text_stage(
    config: Gemma4GPTQConfig,
    *,
    stage: str,
) -> bool:
    """
    Check whether a specific text-side stage is enabled.

    Args:
        config: Gemma4 GPTQ configuration.
        stage: Currently supports ``"layers"`` and ``"lm_head"``.

    Returns:
        ``True`` if the stage should be quantized.

    Raises:
        ValueError: If ``stage`` is unknown.
    """
    if stage == "layers":
        return config.quantize_text and config.quantize_text_layers
    if stage == "lm_head":
        return config.quantize_lm_head

    raise ValueError(f"Unknown text stage: {stage!r}")
