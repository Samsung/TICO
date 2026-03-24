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

from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig


_QUANTIZABLE_LAYER_TYPES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose2d,
)


@dataclass
class Qwen3VLComponents:
    """
    Resolved Qwen3-VL module references.

    Attributes:
        visual: Vision tower root module.
        visual_blocks: Vision transformer blocks.
        visual_patch_embed: Vision patch embedding projection.
        visual_merger: Final vision patch merger.
        visual_deepstack_mergers: Auxiliary deepstack merger modules.
        language_model: Text model root module.
        text_layers: Text decoder layers.
        lm_head: Final language modeling head.
    """

    visual: nn.Module
    visual_blocks: nn.ModuleList
    visual_patch_embed: nn.Module
    visual_merger: nn.Module
    visual_deepstack_mergers: nn.ModuleList
    language_model: nn.Module
    text_layers: nn.ModuleList
    lm_head: nn.Module


@dataclass
class VisionBlockCache:
    """
    Cached per-sample inputs for a Qwen3-VL vision block.

    Attributes:
        hidden_states: Vision hidden states to feed into the next block.
        cu_seqlens: Cumulative sequence lengths used by vision attention.
        position_embeddings: Rotary/position embeddings for the vision path.
    """

    hidden_states: torch.Tensor
    cu_seqlens: torch.Tensor
    position_embeddings: Any


@dataclass
class TextLayerCache:
    """
    Cached per-sample inputs for a Qwen3-VL text decoder layer.
    """

    hidden_states: torch.Tensor
    position_embeddings: Any
    attention_mask: Any = None
    position_ids: Any = None
    past_key_values: Any = None
    use_cache: Any = None
    visual_pos_masks: Any = None
    deepstack_visual_embeds: Any = None


def get_module_by_path(root: nn.Module, path: str) -> Any:
    """
    Resolve a dotted attribute path from a root module.

    Args:
        root: Root module.
        path: Dotted path such as "model.visual.blocks".

    Returns:
        The resolved object.

    Raises:
        AttributeError: If the path cannot be fully resolved.
        ValueError: If `path` is empty.
    """
    if not path:
        raise ValueError("path must be a non-empty string.")

    current: Any = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(
                f"Failed to resolve attribute path {path!r}. "
                f"Missing attribute {part!r} on object of type {type(current).__name__}."
            )
        current = getattr(current, part)
    return current


def resolve_qwen3_vl_components(
    model: nn.Module,
    config: Qwen3VLGPTQConfig,
) -> Qwen3VLComponents:
    """
    Resolve key Qwen3-VL submodules from a model using config-defined paths.

    Args:
        model: Target Qwen3-VL model.
        config: Qwen3-VL GPTQ configuration.

    Returns:
        A dataclass containing resolved module references.

    Raises:
        TypeError: If a resolved object has an unexpected type.
    """
    visual = get_module_by_path(model, config.visual_attr)
    visual_blocks = get_module_by_path(model, config.visual_blocks_attr)
    visual_patch_embed = get_module_by_path(model, config.visual_patch_embed_attr)
    visual_merger = get_module_by_path(model, config.visual_merger_attr)
    visual_deepstack_mergers = get_module_by_path(
        model, config.visual_deepstack_mergers_attr
    )
    language_model = get_module_by_path(model, config.language_model_attr)
    text_layers = get_module_by_path(model, config.text_layers_attr)
    lm_head = get_module_by_path(model, config.lm_head_attr)

    if not isinstance(visual_blocks, nn.ModuleList):
        raise TypeError(
            f"{config.visual_blocks_attr!r} must resolve to nn.ModuleList. "
            f"Got {type(visual_blocks).__name__}."
        )
    if not isinstance(visual_deepstack_mergers, nn.ModuleList):
        raise TypeError(
            f"{config.visual_deepstack_mergers_attr!r} must resolve to nn.ModuleList. "
            f"Got {type(visual_deepstack_mergers).__name__}."
        )
    if not isinstance(text_layers, nn.ModuleList):
        raise TypeError(
            f"{config.text_layers_attr!r} must resolve to nn.ModuleList. "
            f"Got {type(text_layers).__name__}."
        )
    if not isinstance(visual, nn.Module):
        raise TypeError(f"visual must be nn.Module. Got {type(visual).__name__}.")
    if not isinstance(visual_patch_embed, nn.Module):
        raise TypeError(
            f"visual_patch_embed must be nn.Module. Got {type(visual_patch_embed).__name__}."
        )
    if not isinstance(visual_merger, nn.Module):
        raise TypeError(
            f"visual_merger must be nn.Module. Got {type(visual_merger).__name__}."
        )
    if not isinstance(language_model, nn.Module):
        raise TypeError(
            f"language_model must be nn.Module. Got {type(language_model).__name__}."
        )
    if not isinstance(lm_head, nn.Module):
        raise TypeError(f"lm_head must be nn.Module. Got {type(lm_head).__name__}.")

    return Qwen3VLComponents(
        visual=visual,
        visual_blocks=visual_blocks,
        visual_patch_embed=visual_patch_embed,
        visual_merger=visual_merger,
        visual_deepstack_mergers=visual_deepstack_mergers,
        language_model=language_model,
        text_layers=text_layers,
        lm_head=lm_head,
    )


def find_layers(
    module: nn.Module,
    layers: Sequence[type[nn.Module]] | None = None,
    name: str = "",
) -> dict[str, nn.Module]:
    """
    Recursively find submodules whose exact type is included in ``layers``.

    Args:
        module: Root module to search.
        layers: Candidate module types. If ``None``, common quantizable layer
            types are used.
        name: Prefix used during recursion.

    Returns:
        A mapping from qualified local name to module instance.
    """
    if layers is None:
        layers = _QUANTIZABLE_LAYER_TYPES

    if type(module) in layers:
        return {name: module}

    result: dict[str, nn.Module] = {}
    for child_name, child in module.named_children():
        qualified_name = f"{name}.{child_name}" if name else child_name
        result.update(find_layers(child, layers=layers, name=qualified_name))
    return result


def get_quantizable_layers(module: nn.Module) -> dict[str, nn.Module]:
    """
    Return quantizable submodules under ``module``.

    Args:
        module: Root module.

    Returns:
        Mapping from local qualified name to quantizable module.
    """
    return find_layers(module, layers=_QUANTIZABLE_LAYER_TYPES)


def build_module_name_map(model: nn.Module) -> dict[nn.Module, str]:
    """
    Build a reverse lookup from module object to its fully qualified model name.

    Args:
        model: Root model.

    Returns:
        A dictionary mapping module objects to names from `model.named_modules()`.
    """
    return {module: name for name, module in model.named_modules()}


def extract_primary_output(output: Any) -> Any:
    """
    Extract the primary tensor-like payload from a module output.

    This helper handles common Hugging Face style outputs where the first item
    of a tuple is the updated hidden states.

    Args:
        output: Arbitrary module output.

    Returns:
        The primary output object.
    """
    if isinstance(output, tuple):
        return output[0]
    return output


def is_tensor_collection(value: Any) -> bool:
    """
    Check whether a value is a supported tensor container.

    Args:
        value: Any Python object.

    Returns:
        ``True`` if the value is a tensor or a nested container that may hold
        tensors. Otherwise ``False``.
    """
    return isinstance(value, (torch.Tensor, dict, list, tuple))


def tree_map_tensors(
    value: Any,
    fn,
) -> Any:
    """
    Recursively apply a function to all tensors in a nested structure.

    Supported containers are:
        - torch.Tensor
        - dict / Mapping
        - list
        - tuple
        - dataclasses defined in this file

    Args:
        value: Arbitrary nested structure.
        fn: Function applied to each tensor.

    Returns:
        Structure with the same layout and transformed tensors.
    """
    if isinstance(value, torch.Tensor):
        return fn(value)

    if isinstance(value, VisionBlockCache):
        return VisionBlockCache(
            hidden_states=tree_map_tensors(value.hidden_states, fn),
            cu_seqlens=tree_map_tensors(value.cu_seqlens, fn),
            position_embeddings=tree_map_tensors(value.position_embeddings, fn),
        )

    if isinstance(value, TextLayerCache):
        return TextLayerCache(
            hidden_states=tree_map_tensors(value.hidden_states, fn),
            position_embeddings=tree_map_tensors(value.position_embeddings, fn),
            attention_mask=tree_map_tensors(value.attention_mask, fn),
            position_ids=tree_map_tensors(value.position_ids, fn),
            past_key_values=tree_map_tensors(value.past_key_values, fn),
            use_cache=tree_map_tensors(value.use_cache, fn),
            visual_pos_masks=tree_map_tensors(value.visual_pos_masks, fn),
            deepstack_visual_embeds=tree_map_tensors(value.deepstack_visual_embeds, fn),
        )

    if isinstance(value, Mapping):
        return {k: tree_map_tensors(v, fn) for k, v in value.items()}

    if isinstance(value, list):
        return [tree_map_tensors(v, fn) for v in value]

    if isinstance(value, tuple):
        return tuple(tree_map_tensors(v, fn) for v in value)

    return value


def detach_clone_tree(value: Any) -> Any:
    """
    Deep-copy a nested tensor structure using `detach().clone()`.

    Non-tensor leaf values are returned as-is.

    Args:
        value: Arbitrary nested structure.

    Returns:
        Detached and cloned copy of the input structure.
    """
    return tree_map_tensors(value, lambda x: x.detach().clone())


def move_tensor_tree(
    value: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> Any:
    """
    Move all tensors in a nested structure to a target device and/or dtype.

    Args:
        value: Arbitrary nested structure.
        device: Target device.
        dtype: Target dtype. If ``None``, dtype is preserved.
        non_blocking: Passed to ``Tensor.to``.

    Returns:
        Structure with moved tensors.
    """

    def _move(x: torch.Tensor) -> torch.Tensor:
        kwargs: dict[str, Any] = {"non_blocking": non_blocking}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None and x.is_floating_point():
            kwargs["dtype"] = dtype
        return x.to(**kwargs)

    return tree_map_tensors(value, _move)


def maybe_move_cache_to_cpu(
    value: Any,
    *,
    enabled: bool,
    dtype: torch.dtype | None = None,
) -> Any:
    """
    Optionally move a cached structure to CPU.

    Args:
        value: Cached structure.
        enabled: Whether to move the cache to CPU.
        dtype: Optional floating-point dtype conversion.

    Returns:
        Original value if disabled, otherwise a CPU copy.
    """
    if not enabled:
        return value
    return move_tensor_tree(value, device="cpu", dtype=dtype)


def gather_single_batch_from_dict(
    data_dict: Mapping[str, Sequence[Any]],
    idx: int,
) -> dict[str, Any]:
    """
    Gather one logical batch item from a dict of per-batch lists.

    Args:
        data_dict: Mapping from key to per-batch sequence.
        idx: Batch index.

    Returns:
        A single-batch dictionary.
    """
    return {k: v[idx] for k, v in data_dict.items()}


def gather_single_batch_from_list(
    data_list: Sequence[Sequence[Any]],
    idx: int,
) -> list[Any]:
    """
    Gather one logical batch item from a list of per-argument lists.

    Args:
        data_list: List where each element stores one positional argument across
            batches.
        idx: Batch index.

    Returns:
        Positional arguments for a single batch.
    """
    return [arg_list[idx] for arg_list in data_list]


def clone_batch_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """
    Clone and detach a kwargs-like mapping.

    Args:
        kwargs: Input kwargs mapping.

    Returns:
        Detached clone of the mapping.
    """
    return detach_clone_tree(dict(kwargs))


def prepare_model_kwargs(
    kwargs: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    """
    Move model kwargs to a target device.

    Args:
        kwargs: Input keyword arguments.
        device: Target device.

    Returns:
        A device-moved kwargs dictionary.
    """
    return move_tensor_tree(dict(kwargs), device=device)


def split_model_inputs(
    batch: Any,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Normalize an arbitrary calibration batch into ``(args, kwargs)``.

    Supported input forms:
        - tensor -> ``([tensor], {})``
        - tuple/list -> ``(list(batch), {})``
        - dict/mapping -> ``([], dict(batch))``

    Args:
        batch: Arbitrary user-provided batch.

    Returns:
        Normalized positional and keyword arguments.

    Raises:
        TypeError: If the batch type is unsupported.
    """
    if isinstance(batch, torch.Tensor):
        return [batch], {}

    if isinstance(batch, Mapping):
        return [], dict(batch)

    if isinstance(batch, (list, tuple)):
        return list(batch), {}

    raise TypeError(
        "Unsupported calibration batch type. "
        f"Got {type(batch).__name__}. Expected Tensor, Mapping, list, or tuple."
    )


def append_batch_to_cache(
    cache_args: list[list[Any]],
    cache_kwargs: dict[str, list[Any]],
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Append one batch of positional and keyword inputs to cache storage.

    Args:
        cache_args: Positional cache storage.
        cache_kwargs: Keyword cache storage.
        *args: Positional batch values.
        **kwargs: Keyword batch values.
    """
    for idx, item in enumerate(args):
        if idx >= len(cache_args):
            cache_args.append([])
        cache_args[idx].append(detach_clone_tree(item))

    for key, value in kwargs.items():
        if key not in cache_kwargs:
            cache_kwargs[key] = []
        cache_kwargs[key].append(detach_clone_tree(value))


def iter_cached_batches(
    cache_args: Sequence[Sequence[Any]],
    cache_kwargs: Mapping[str, Sequence[Any]],
    num_batches: int,
) -> Iterable[tuple[list[Any], dict[str, Any]]]:
    """
    Iterate over cached batches in normalized ``(args, kwargs)`` form.

    Args:
        cache_args: Positional cache storage.
        cache_kwargs: Keyword cache storage.
        num_batches: Number of cached batches.

    Yields:
        Tuples of ``(args, kwargs)`` for each batch index.
    """
    for batch_idx in range(num_batches):
        yield (
            gather_single_batch_from_list(cache_args, batch_idx),
            gather_single_batch_from_dict(cache_kwargs, batch_idx),
        )


def get_first_parameter_device(module: nn.Module) -> torch.device:
    """
    Return the device of the first parameter or buffer found in a module.

    Args:
        module: Target module.

    Returns:
        Device of the first parameter or buffer.

    Raises:
        RuntimeError: If the module has neither parameters nor buffers.
    """
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    raise RuntimeError(
        f"Could not infer device for module of type {type(module).__name__}."
    )


def get_first_parameter_dtype(module: nn.Module) -> torch.dtype:
    """
    Return the dtype of the first parameter or buffer found in a module.

    Args:
        module: Target module.

    Returns:
        Dtype of the first parameter or buffer.

    Raises:
        RuntimeError: If the module has neither parameters nor buffers.
    """
    for param in module.parameters(recurse=True):
        return param.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.dtype
    raise RuntimeError(
        f"Could not infer dtype for module of type {type(module).__name__}."
    )


def should_quantize_vision_stage(
    config: Qwen3VLGPTQConfig,
    *,
    stage: str,
) -> bool:
    """
    Check whether a specific vision-side stage is enabled.

    Args:
        config: Qwen3-VL GPTQ configuration.
        stage: One of
            ``"patch_embed"``, ``"blocks"``, ``"merger"``,
            ``"deepstack_mergers"``.

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
    if stage == "merger":
        return config.quantize_vision_merger
    if stage == "deepstack_mergers":
        return config.quantize_vision_deepstack_mergers

    raise ValueError(f"Unknown vision stage: {stage!r}")


def should_quantize_text_stage(
    config: Qwen3VLGPTQConfig,
    *,
    stage: str,
) -> bool:
    """
    Check whether a specific text-side stage is enabled.

    Args:
        config: Qwen3-VL GPTQ configuration.
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


def get_deepstack_entry(
    deepstack_visual_embeds: Any,
    layer_idx: int,
) -> Any:
    """
    Safely fetch the deepstack visual embedding associated with a text layer.

    Args:
        deepstack_visual_embeds: Deepstack embedding container. Usually a list,
            tuple, or ``None``.
        layer_idx: Decoder layer index.

    Returns:
        The layer-specific deepstack feature if available, otherwise ``None``.
    """
    if deepstack_visual_embeds is None:
        return None
    if isinstance(deepstack_visual_embeds, (list, tuple)):
        if 0 <= layer_idx < len(deepstack_visual_embeds):
            return deepstack_visual_embeds[layer_idx]
        return None
    return None
