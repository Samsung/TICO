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

import re
from typing import cast, Dict, Sequence, Tuple, TypeAlias

import torch
import torch.nn as nn
from tqdm import tqdm

from tico.quantization.config.cle import CLEConfig


SupportedLayer: TypeAlias = nn.Conv2d | nn.Linear


def _glob_to_regex(pattern: str) -> re.Pattern:
    """
    Convert a module-name glob pattern into a regex.

    The `*` wildcard captures one module-name segment. For example,
    `model.layers.*.mlp.up_proj` matches `model.layers.0.mlp.up_proj`
    and captures `0`.

    Args:
        pattern: Module-name pattern.

    Returns:
        Compiled regex pattern.
    """
    parts = pattern.split(".")
    regex_parts = []

    for part in parts:
        if part == "*":
            regex_parts.append(r"([^.]+)")
        else:
            regex_parts.append(re.escape(part))

    return re.compile("^" + r"\.".join(regex_parts) + "$")


def _has_wildcard(pattern: str) -> bool:
    """
    Return whether the pattern contains a wildcard segment.

    Args:
        pattern: Module-name pattern.

    Returns:
        True if the pattern contains `*`, otherwise False.
    """
    return "*" in pattern.split(".")


def _expand_pair_pattern(
    module_names: Sequence[str],
    first_pattern: str,
    second_pattern: str,
) -> list[tuple[str, str]]:
    """
    Expand one exact or wildcard CLE pair pattern into concrete module-name pairs.

    If both patterns contain wildcards, captured wildcard values must match.
    This prevents pairing `model.layers.0...` with `model.layers.1...`.

    Args:
        module_names: All module names from the target model.
        first_pattern: First layer name or glob-style pattern.
        second_pattern: Second layer name or glob-style pattern.

    Returns:
        Concrete module-name pairs.

    Raises:
        ValueError: If only one side uses wildcards, wildcard counts differ,
            or no concrete pair is found.
    """
    first_has_wildcard = _has_wildcard(first_pattern)
    second_has_wildcard = _has_wildcard(second_pattern)

    if first_has_wildcard != second_has_wildcard:
        raise ValueError(
            "CLE pair patterns must either both use wildcards or both be exact names. "
            f"Got: {first_pattern!r}, {second_pattern!r}"
        )

    if not first_has_wildcard:
        return [(first_pattern, second_pattern)]

    if first_pattern.count("*") != second_pattern.count("*"):
        raise ValueError(
            "CLE pair patterns must have the same number of wildcard segments. "
            f"Got: {first_pattern!r}, {second_pattern!r}"
        )

    first_regex = _glob_to_regex(first_pattern)
    second_regex = _glob_to_regex(second_pattern)

    first_matches: dict[tuple[str, ...], str] = {}
    second_matches: dict[tuple[str, ...], str] = {}

    for name in module_names:
        first_match = first_regex.match(name)
        if first_match is not None:
            first_matches[first_match.groups()] = name

        second_match = second_regex.match(name)
        if second_match is not None:
            second_matches[second_match.groups()] = name

    common_keys = sorted(first_matches.keys())
    concrete_pairs = [
        (first_matches[key], second_matches[key])
        for key in common_keys
        if key in second_matches
    ]

    if not concrete_pairs:
        raise ValueError(
            "No concrete CLE layer pairs were found for patterns: "
            f"{first_pattern!r}, {second_pattern!r}"
        )

    return concrete_pairs


def _expand_pairs(
    model: nn.Module,
    pairs: Sequence[Tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Expand all CLE pair specs into concrete module-name pairs.

    Args:
        model: Target PyTorch model.
        pairs: Exact or glob-style CLE pair specs.

    Returns:
        Concrete module-name pairs.
    """
    module_names = list(dict(model.named_modules()).keys())

    expanded_pairs: list[tuple[str, str]] = []
    for first_pattern, second_pattern in pairs:
        expanded_pairs.extend(
            _expand_pair_pattern(module_names, first_pattern, second_pattern)
        )

    # Remove duplicate pairs while preserving order
    expanded_pairs = list(dict.fromkeys(expanded_pairs))
    return expanded_pairs


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """
    Return a submodule by its dotted module name.

    Args:
        model: Root PyTorch module.
        name: Dotted module path such as ``"layer1.0.conv1"``.

    Returns:
        The requested submodule.

    Raises:
        ValueError: If the module name does not exist.
    """
    modules = dict(model.named_modules())
    if name not in modules:
        raise ValueError(f"Module '{name}' does not exist in the model.")
    return modules[name]


def _channel_range(weight: torch.Tensor, dim: int, method: str) -> torch.Tensor:
    """
    Compute per-channel weight ranges.

    Args:
        weight: Weight tensor.
        dim: Channel dimension to preserve.
        method: Range method. Either ``"absmax"`` or ``"range"``.

    Returns:
        A 1D tensor containing one range value per channel.
    """
    reduce_dims = tuple(i for i in range(weight.ndim) if i != dim)

    if method == "absmax":
        return weight.abs().amax(dim=reduce_dims)

    if method == "range":
        return weight.amax(dim=reduce_dims) - weight.amin(dim=reduce_dims)

    raise ValueError(f"Unsupported CLE range method: {method}")


def _validate_pair(first: nn.Module, second: nn.Module) -> None:
    """
    Validate that a layer pair is supported by Cross-Layer Equalization.

    Args:
        first: The first layer in the pair.
        second: The second layer in the pair.

    Raises:
        TypeError: If either layer type is unsupported.
        ValueError: If the two layers do not have compatible channels.
    """
    supported = (nn.Conv2d, nn.Linear)

    if not isinstance(first, supported):
        raise TypeError(
            "The first CLE layer must be nn.Conv2d or nn.Linear. "
            f"Got {type(first).__name__}."
        )

    if not isinstance(second, supported):
        raise TypeError(
            "The second CLE layer must be nn.Conv2d or nn.Linear. "
            f"Got {type(second).__name__}."
        )

    first_out_channels = first.weight.shape[0]
    second_in_channels = second.weight.shape[1]

    if first_out_channels != second_in_channels:
        raise ValueError(
            "CLE requires first.out_channels to match second.in_channels. "
            f"Got {first_out_channels} and {second_in_channels}."
        )


def _reshape_scale_for_first_weight(
    scale: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Reshape a per-output-channel scale for the first layer weight.

    Args:
        scale: Per-channel scale tensor with shape ``[C]``.
        weight: First layer weight tensor.

    Returns:
        Reshaped scale tensor broadcastable to the first layer weight.
    """
    return scale.reshape(-1, *([1] * (weight.ndim - 1)))


def _reshape_scale_for_second_weight(
    scale: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Reshape a per-input-channel scale for the second layer weight.

    Args:
        scale: Per-channel scale tensor with shape ``[C]``.
        weight: Second layer weight tensor.

    Returns:
        Reshaped scale tensor broadcastable to the second layer weight.
    """
    shape = [1] * weight.ndim
    shape[1] = -1
    return scale.reshape(*shape)


@torch.no_grad()
def equalize_layer_pair(
    first: SupportedLayer,
    second: SupportedLayer,
    *,
    method: str = "absmax",
    eps: float = 1e-12,
    scale_range: Tuple[float, float] = (1e-8, 1e8),
    equalize_bias: bool = True,
) -> torch.Tensor:
    """
    Apply Cross-Layer Equalization to one adjacent layer pair.

    The transformation scales the output channels of the first layer and
    inversely scales the corresponding input channels of the second layer.

    Args:
        first: First layer. Supported types are ``nn.Conv2d`` and ``nn.Linear``.
        second: Second layer. Supported types are ``nn.Conv2d`` and ``nn.Linear``.
        method: Range method. Either ``"absmax"`` or ``"range"``.
        eps: Small value used to avoid division by zero.
        scale_range: Minimum and maximum clamp values for the scale.
        equalize_bias: Whether to scale the first layer bias together with
            the first layer output channels.

    Returns:
        The applied per-channel scale tensor.

    Raises:
        TypeError: If an unsupported layer type is provided.
        ValueError: If the two layers have incompatible channel dimensions.
    """
    _validate_pair(first, second)

    first_weight = first.weight
    second_weight = second.weight

    first_range = _channel_range(first_weight, dim=0, method=method)
    second_range = _channel_range(second_weight, dim=1, method=method)

    scale = torch.sqrt((second_range + eps) / (first_range + eps))
    scale = torch.clamp(scale, min=scale_range[0], max=scale_range[1])
    scale = scale.to(device=first_weight.device, dtype=first_weight.dtype)

    first_weight.mul_(_reshape_scale_for_first_weight(scale, first_weight))
    second_weight.div_(_reshape_scale_for_second_weight(scale, second_weight))

    if equalize_bias and first.bias is not None:
        first.bias.mul_(scale)

    return scale


@torch.no_grad()
def apply_cross_layer_equalization(
    model: nn.Module,
    config: CLEConfig,
) -> Dict[Tuple[str, str], torch.Tensor]:
    """
    Apply Cross-Layer Equalization to manually specified layer pairs.

    Args:
        model: Target PyTorch model.
        config: Cross-Layer Equalization configuration.

    Returns:
        A dictionary mapping each layer-name pair to the applied scale tensor.
    """
    applied_scales: Dict[Tuple[str, str], torch.Tensor] = {}
    concrete_pairs = _expand_pairs(model, config.pairs)

    for iter_idx in range(config.max_iter):
        iterator = concrete_pairs

        if config.show_progress:
            iterator = tqdm(
                concrete_pairs,
                total=len(concrete_pairs),
                desc=f"Applying Cross-Layer Equalization [{iter_idx + 1}/{config.max_iter}]",
            )

        for first_name, second_name in iterator:
            first = cast(SupportedLayer, _get_module_by_name(model, first_name))
            second = cast(SupportedLayer, _get_module_by_name(model, second_name))

            scale = equalize_layer_pair(
                first,
                second,
                method=config.method,
                eps=config.eps,
                scale_range=config.scale_range,
                equalize_bias=config.equalize_bias,
            )

            applied_scales[(first_name, second_name)] = scale.detach().cpu()

    return applied_scales
