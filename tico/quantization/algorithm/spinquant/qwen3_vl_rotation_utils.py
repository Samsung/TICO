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

import math
from typing import Optional

import torch
import torch.nn as nn

from tico.quantization.algorithm.spinquant.fuse_norm_utils import (
    fuse_norm_into_linears,
    reset_norm_affine_to_identity,
)
from tico.quantization.algorithm.spinquant.hadamard_utils import (
    make_random_hadamard_rotation,
)
from tico.quantization.algorithm.spinquant.qwen3_vl_model_utils import (
    require_linear_attr,
    resolve_qwen3_vl_spinquant_components,
)
from tico.quantization.algorithm.spinquant.rotation_utils import (
    copy_linear_weight_,
    infer_device,
    left_multiply_linear_weight_,
    make_random_orthogonal_rotation,
    right_multiply_linear_weight_,
    rotate_attention_inputs,
    rotate_attention_output,
    rotate_mlp_input,
    rotate_mlp_output,
    rotate_ov_proj,
    validate_square_rotation,
)
from tico.quantization.config.qwen3_vl_spinquant import Qwen3VLSpinQuantConfig


def get_qwen3_vl_text_hidden_size(model: nn.Module) -> int:
    """
    Return the Qwen3-VL text hidden size.

    Parameters:
        model: Target Qwen3-VL model.

    Returns:
        Text hidden size.

    Raises:
        AttributeError: If the text configuration is missing.
    """
    if not hasattr(model, "config") or not hasattr(model.config, "text_config"):
        raise AttributeError("Expected `model.config.text_config` to exist.")
    return int(model.config.text_config.hidden_size)


def get_qwen3_vl_head_dim(model: nn.Module) -> int:
    """
    Return the Qwen3-VL text attention head dimension.

    Parameters:
        model: Target Qwen3-VL model.

    Returns:
        Attention head dimension.

    Raises:
        AttributeError: If the text configuration is missing.
        ValueError: If the inferred dimensions are invalid.
    """
    if not hasattr(model, "config") or not hasattr(model.config, "text_config"):
        raise AttributeError("Expected `model.config.text_config` to exist.")

    text_config = model.config.text_config
    if hasattr(text_config, "head_dim") and text_config.head_dim is not None:
        return int(text_config.head_dim)

    hidden_size = int(text_config.hidden_size)
    num_heads = int(text_config.num_attention_heads)
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_heads})."
        )
    return hidden_size // num_heads


def get_qwen3_vl_vision_hidden_size(model: nn.Module) -> int:
    """
    Return the Qwen3-VL vision hidden size.

    Parameters:
        model: Target Qwen3-VL model.

    Returns:
        Vision tower hidden size.

    Raises:
        AttributeError: If the vision configuration is missing.
    """
    if not hasattr(model, "config") or not hasattr(model.config, "vision_config"):
        raise AttributeError("Expected `model.config.vision_config` to exist.")
    return int(model.config.vision_config.hidden_size)


def get_qwen3_vl_vision_head_dim(model: nn.Module) -> int:
    """
    Return the Qwen3-VL vision attention head dimension.

    Parameters:
        model: Target Qwen3-VL model.

    Returns:
        Vision attention head dimension.

    Raises:
        AttributeError: If the vision configuration is missing.
        ValueError: If the inferred dimensions are invalid.
    """
    if not hasattr(model, "config") or not hasattr(model.config, "vision_config"):
        raise AttributeError("Expected `model.config.vision_config` to exist.")

    vision_config = model.config.vision_config
    hidden_size = int(vision_config.hidden_size)
    num_heads = int(vision_config.num_heads)
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"vision hidden_size ({hidden_size}) must be divisible by "
            f"vision num_heads ({num_heads})."
        )
    return hidden_size // num_heads


def build_qwen3_vl_r1(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> torch.Tensor:
    """
    Build the Qwen3-VL global text hidden-dimension R1 rotation.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.

    Returns:
        A square rotation matrix with shape ``[text_hidden_size, text_hidden_size]``.

    Raises:
        ValueError: If the configuration is invalid.
    """
    hidden_size = get_qwen3_vl_text_hidden_size(model)
    device = infer_device(model)

    if config.init_method == "random":
        return make_random_orthogonal_rotation(hidden_size, device=device)

    if config.init_method == "hadamard":
        return make_random_hadamard_rotation(hidden_size, device=device)

    if config.init_method == "external":
        if config.r1 is None:
            raise ValueError("`r1` must be provided when init_method='external'.")
        rotation = config.r1.to(device=device, dtype=torch.float64)
        validate_square_rotation("r1", rotation, hidden_size)
        return rotation

    raise ValueError(f"Unsupported init_method: {config.init_method!r}")


def build_qwen3_vl_r2(
    *,
    init_method: str,
    r2_map: Optional[dict[str, torch.Tensor]],
    layer_idx: int,
    head_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Build or fetch a Qwen3-VL text per-layer head-dimension R2 rotation.

    Parameters:
        init_method: Rotation initialization strategy.
        r2_map: Optional mapping from keys to external R2 matrices.
        layer_idx: Text decoder layer index.
        head_dim: Attention head dimension.
        device: Target device.

    Returns:
        A square R2 matrix or None.

    Raises:
        ValueError: If an external R2 has an invalid shape.
    """
    if init_method == "random":
        return make_random_orthogonal_rotation(head_dim, device=device)

    if init_method == "hadamard":
        return make_random_hadamard_rotation(head_dim, device=device)

    if init_method == "external":
        if r2_map is None:
            return None

        candidate_keys = (
            f"model.language_model.layers.{layer_idx}.self_attn.R2",
            f"model.layers.{layer_idx}.self_attn.R2",
        )
        for key in candidate_keys:
            rotation = r2_map.get(key)
            if rotation is None:
                continue

            rotation = rotation.to(device=device, dtype=torch.float64)
            validate_square_rotation(key, rotation, head_dim)
            return rotation

        return None

    raise ValueError(f"Unsupported init_method: {init_method!r}")


def _make_zero_mean_subspace_basis(
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build an orthonormal basis whose first vector is the normalized all-ones vector.

    Parameters:
        size: Ambient dimension.
        device: Target device.
        dtype: Tensor dtype.

    Returns:
        A square orthonormal basis matrix.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")

    mean_axis = torch.ones(size, 1, device=device, dtype=dtype) / math.sqrt(size)
    if size == 1:
        return mean_axis

    random_tail = torch.randn(size, size - 1, device=device, dtype=dtype)
    random_tail = random_tail - mean_axis @ (mean_axis.T @ random_tail)
    tail_basis, _ = torch.linalg.qr(random_tail, mode="reduced")
    return torch.cat([mean_axis, tail_basis], dim=1)


def make_layernorm_compatible_rotation(
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    tail_init_method: str = "random",
) -> torch.Tensor:
    """
    Create an orthogonal rotation that preserves LayerNorm equivariance.

    The returned matrix R satisfies ``R @ 1 = 1``. Therefore, for row-vector
    inputs and identity-affine LayerNorm, ``LayerNorm(x @ R) == LayerNorm(x) @ R``
    up to numerical precision.

    Parameters:
        size: Rotation dimension.
        device: Target device.
        dtype: Tensor dtype.
        tail_init_method: Initialization for the zero-mean subspace. ``"hadamard"``
            uses a Hadamard rotation when the subspace size supports it and falls
            back to a dense random orthogonal rotation otherwise.

    Returns:
        A square orthogonal matrix with shape ``[size, size]``.
    """
    if size == 1:
        return torch.eye(1, device=device, dtype=dtype)

    basis = _make_zero_mean_subspace_basis(size, device=device, dtype=torch.float64)

    if tail_init_method == "hadamard":
        try:
            tail_rotation = make_random_hadamard_rotation(
                size - 1,
                device=device,
                dtype=torch.float64,
            )
        except ValueError:
            tail_rotation = make_random_orthogonal_rotation(
                size - 1,
                device=device,
                dtype=torch.float64,
            )
    elif tail_init_method == "random":
        tail_rotation = make_random_orthogonal_rotation(
            size - 1,
            device=device,
            dtype=torch.float64,
        )
    else:
        raise ValueError(f"Unsupported tail_init_method: {tail_init_method!r}")

    block = torch.eye(size, device=device, dtype=torch.float64)
    block[1:, 1:] = tail_rotation
    rotation = basis @ block @ basis.T
    return rotation.to(dtype=dtype)


def validate_layernorm_compatible_rotation(
    name: str,
    rotation: torch.Tensor,
    size: int,
    *,
    atol: float,
) -> None:
    """
    Validate that a rotation is orthogonal and preserves the all-ones direction.

    Parameters:
        name: Logical name used in error messages.
        rotation: Rotation matrix to validate.
        size: Expected matrix size.
        atol: Absolute tolerance.

    Raises:
        ValueError: If the rotation is invalid.
    """
    validate_square_rotation(name, rotation, size)

    identity = torch.eye(size, device=rotation.device, dtype=torch.float64)
    gram = rotation.T @ rotation
    orthogonal_error = (gram - identity).abs().max().item()
    if orthogonal_error > atol:
        raise ValueError(
            f"{name} must be orthogonal within atol={atol}, "
            f"but max |R.T @ R - I| is {orthogonal_error}."
        )

    ones = torch.ones(size, 1, device=rotation.device, dtype=torch.float64)
    ones_error = (rotation @ ones - ones).abs().max().item()
    if ones_error > atol:
        raise ValueError(
            f"{name} must preserve the all-ones direction for LayerNorm-safe "
            f"vision R1 fusion, but max |R @ 1 - 1| is {ones_error}."
        )


def build_qwen3_vl_vision_r1(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> torch.Tensor:
    """
    Build the Qwen3-VL vision hidden-dimension R1 rotation.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.

    Returns:
        A LayerNorm-compatible square rotation matrix.

    Raises:
        ValueError: If the configuration is invalid.
    """
    hidden_size = get_qwen3_vl_vision_hidden_size(model)
    device = infer_device(model)
    init_method = config.vision_init_method or config.init_method

    if init_method in {"random", "hadamard"}:
        return make_layernorm_compatible_rotation(
            hidden_size,
            device=device,
            dtype=torch.float64,
            tail_init_method=init_method,
        )

    if init_method == "external":
        if config.vision_r1 is None:
            raise ValueError(
                "`vision_r1` must be provided when vision_init_method='external'."
            )
        rotation = config.vision_r1.to(device=device, dtype=torch.float64)
        if config.require_vision_r1_layernorm_compatible:
            validate_layernorm_compatible_rotation(
                "vision_r1",
                rotation,
                hidden_size,
                atol=float(config.vision_rotation_tolerance),
            )
        else:
            validate_square_rotation("vision_r1", rotation, hidden_size)
        return rotation

    raise ValueError(f"Unsupported vision init_method: {init_method!r}")


def build_qwen3_vl_vision_r2(
    *,
    init_method: str,
    r2_map: Optional[dict[str, torch.Tensor]],
    layer_idx: int,
    head_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Build or fetch a Qwen3-VL vision per-layer head-dimension R2 rotation.

    Parameters:
        init_method: Rotation initialization strategy.
        r2_map: Optional mapping from vision block keys to external R2 matrices.
        layer_idx: Vision block index.
        head_dim: Vision attention head dimension.
        device: Target device.

    Returns:
        A square R2 matrix, or None when external mode does not provide one.
    """
    if init_method == "random":
        return make_random_orthogonal_rotation(head_dim, device=device)

    if init_method == "hadamard":
        try:
            return make_random_hadamard_rotation(head_dim, device=device)
        except ValueError:
            return make_random_orthogonal_rotation(head_dim, device=device)

    if init_method == "external":
        if r2_map is None:
            return None

        candidate_keys = (
            f"model.visual.blocks.{layer_idx}.attn.R2",
            f"model.visual.blocks.{layer_idx}.attn.vision_R2",
        )
        for key in candidate_keys:
            rotation = r2_map.get(key)
            if rotation is None:
                continue

            rotation = rotation.to(device=device, dtype=torch.float64)
            validate_square_rotation(key, rotation, head_dim)
            return rotation

        return None

    raise ValueError(f"Unsupported vision init_method: {init_method!r}")


@torch.no_grad()
def fuse_qwen3_vl_text_layer_norms(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> nn.Module:
    """
    Fold Qwen3-VL text RMSNorm affine weights into adjacent Linear layers.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.

    Returns:
        The same model instance.

    Raises:
        AttributeError: If required norm modules are missing.
        TypeError: If adjacent modules are not Linear layers.
    """
    if getattr(model, "_qwen3_vl_spinquant_norms_fused", False):
        return model

    components = resolve_qwen3_vl_spinquant_components(model, config)

    for layer in components.text_layers:
        input_norm = getattr(layer, "input_layernorm")
        post_attn_norm = getattr(layer, "post_attention_layernorm")

        q_proj = require_linear_attr(layer.self_attn, "q_proj")
        k_proj = require_linear_attr(layer.self_attn, "k_proj")
        v_proj = require_linear_attr(layer.self_attn, "v_proj")
        gate_proj = require_linear_attr(layer.mlp, "gate_proj")
        up_proj = require_linear_attr(layer.mlp, "up_proj")

        fuse_norm_into_linears(input_norm, [q_proj, k_proj, v_proj])
        fuse_norm_into_linears(post_attn_norm, [gate_proj, up_proj])

        reset_norm_affine_to_identity(input_norm)
        reset_norm_affine_to_identity(post_attn_norm)

    setattr(model, "_qwen3_vl_spinquant_norms_fused", True)
    return model


@torch.no_grad()
def _fuse_repeated_norm_into_linear(
    norm: nn.Module,
    linear: nn.Linear,
    repeated_dim: int,
) -> None:
    """
    Fold a per-token norm affine into a Linear that consumes concatenated tokens.

    Parameters:
        norm: Normalization module exposing ``weight`` and optionally ``bias``.
        linear: Linear layer that consumes repeated chunks of size ``repeated_dim``.
        repeated_dim: Chunk size matched by the norm affine.

    Raises:
        ValueError: If the linear input dimension is not a multiple of the norm size.
    """
    if not hasattr(norm, "weight") or norm.weight is None:
        raise AttributeError(
            f"Normalization module `{type(norm).__name__}` must expose `weight`."
        )

    gamma = norm.weight.data.to(torch.float64)
    if gamma.numel() != repeated_dim:
        raise ValueError(f"Expected norm size {repeated_dim}, got {gamma.numel()}.")

    weight = linear.weight.data
    if weight.shape[1] % repeated_dim != 0:
        raise ValueError(
            f"linear.in_features={weight.shape[1]} is not divisible by {repeated_dim}."
        )

    device = weight.device
    dtype = weight.dtype
    num_repeats = weight.shape[1] // repeated_dim
    w64 = weight.to(torch.float64)
    w64_chunks = w64.reshape(weight.shape[0], num_repeats, repeated_dim)

    if hasattr(norm, "bias") and norm.bias is not None:
        beta = norm.bias.data.to(torch.float64)
        if linear.bias is None:
            linear.bias = nn.Parameter(
                torch.zeros(linear.out_features, device=device, dtype=dtype)
            )
        b64 = linear.bias.data.to(torch.float64)
        bias_delta = torch.einsum("ord,d->or", w64_chunks, beta).sum(dim=1)
        linear.bias.data.copy_((b64 + bias_delta).to(device=device, dtype=dtype))

    fused = w64_chunks * gamma.view(1, 1, repeated_dim)
    linear.weight.data.copy_(fused.reshape_as(w64).to(device=device, dtype=dtype))


@torch.no_grad()
def _fuse_vision_merger_norm_into_fc1(
    merger: nn.Module,
    vision_hidden_size: int,
) -> None:
    """
    Fold a Qwen3-VL vision merger LayerNorm affine into ``linear_fc1``.

    Parameters:
        merger: Qwen3-VL vision patch merger.
        vision_hidden_size: Hidden size of one vision token.
    """
    if not hasattr(merger, "norm"):
        raise AttributeError("Expected vision merger to expose `norm`.")

    norm = merger.norm
    linear_fc1 = require_linear_attr(merger, "linear_fc1")

    if not hasattr(norm, "weight") or norm.weight is None:
        raise AttributeError("Expected vision merger norm to expose `weight`.")

    norm_dim = int(norm.weight.numel())
    if norm_dim == linear_fc1.in_features:
        fuse_norm_into_linears(norm, [linear_fc1])
    elif norm_dim == vision_hidden_size and linear_fc1.in_features % norm_dim == 0:
        _fuse_repeated_norm_into_linear(norm, linear_fc1, norm_dim)
    else:
        raise ValueError(
            "Unsupported vision merger norm/linear shape: "
            f"norm_dim={norm_dim}, linear_fc1.in_features={linear_fc1.in_features}, "
            f"vision_hidden_size={vision_hidden_size}."
        )

    reset_norm_affine_to_identity(norm)


@torch.no_grad()
def fuse_qwen3_vl_vision_layer_norms(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> nn.Module:
    """
    Fold Qwen3-VL vision LayerNorm affine weights into adjacent Linear layers.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.

    Returns:
        The same model instance.
    """
    if getattr(model, "_qwen3_vl_spinquant_vision_norms_fused", False):
        return model

    components = resolve_qwen3_vl_spinquant_components(model, config)
    vision_hidden_size = get_qwen3_vl_vision_hidden_size(model)

    for block in components.vision_blocks:
        norm1 = getattr(block, "norm1")
        norm2 = getattr(block, "norm2")
        qkv = require_linear_attr(block.attn, "qkv")
        linear_fc1 = require_linear_attr(block.mlp, "linear_fc1")

        fuse_norm_into_linears(norm1, [qkv])
        fuse_norm_into_linears(norm2, [linear_fc1])
        reset_norm_affine_to_identity(norm1)
        reset_norm_affine_to_identity(norm2)

    if hasattr(components.visual_model, "merger"):
        _fuse_vision_merger_norm_into_fc1(
            components.visual_model.merger,
            vision_hidden_size,
        )

    for merger in components.visual_deepstack_mergers:
        _fuse_vision_merger_norm_into_fc1(merger, vision_hidden_size)

    setattr(model, "_qwen3_vl_spinquant_vision_norms_fused", True)
    return model


@torch.no_grad()
def extract_and_reset_qwen3_vl_final_norm_scale(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> torch.Tensor:
    """
    Extract the Qwen3-VL final text RMSNorm scale and reset it to identity.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.

    Returns:
        The original final norm scale as a float64 tensor.

    Raises:
        AttributeError: If the final norm does not expose a weight.
    """
    components = resolve_qwen3_vl_spinquant_components(model, config)
    final_norm = components.language_model.norm

    if not hasattr(final_norm, "weight") or final_norm.weight is None:
        raise AttributeError("Expected final text norm to expose `weight`.")

    gamma = (
        final_norm.weight.data.detach()
        .to(device=final_norm.weight.device, dtype=torch.float64)
        .clone()
    )

    reset_norm_affine_to_identity(final_norm)
    return gamma


@torch.no_grad()
def apply_qwen3_vl_embedding_side_rotation(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
    r1: torch.Tensor,
) -> None:
    """
    Store the input-side R1 rotation in the Qwen3-VL runtime boundary module.

    Parameters:
        model: Target SpinQwen3VL model.
        config: Qwen3-VL SpinQuant configuration.
        r1: Global hidden-dimension rotation matrix.

    Notes:
        The runtime transform should compute ``x @ R1``. For ``nn.Linear``,
        ``output = x @ weight.T``, so the stored weight is ``R1.T``.
    """
    components = resolve_qwen3_vl_spinquant_components(model, config)
    rotate_embedding = require_linear_attr(
        components.language_model, "rotate_embedding"
    )
    copy_linear_weight_(rotate_embedding, r1.T)


@torch.no_grad()
def apply_qwen3_vl_lm_head_side_rotation(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
    r1: torch.Tensor,
    norm_scale: torch.Tensor,
) -> None:
    """
    Store the output-side R1 correction and final norm scale.

    Parameters:
        model: Target SpinQwen3VL model.
        config: Qwen3-VL SpinQuant configuration.
        r1: Global hidden-dimension rotation matrix.
        norm_scale: Original final RMSNorm affine scale.

    Notes:
        The runtime transform should compute ``x @ R1.T @ diag(norm_scale)``.
        For ``nn.Linear``, this is represented by storing
        ``diag(norm_scale) @ R1`` as the weight.
    """
    rotate_lm_head = require_linear_attr(model, "rotate_lm_head")

    gamma = norm_scale.to(device=r1.device, dtype=torch.float64)
    if gamma.ndim != 1 or gamma.numel() != r1.shape[0]:
        raise ValueError(
            f"`norm_scale` must have shape [{r1.shape[0]}], got {tuple(gamma.shape)}."
        )

    weight = gamma.view(-1, 1) * r1
    copy_linear_weight_(rotate_lm_head, weight)


@torch.no_grad()
def rotate_qwen3_vl_text_layer_r1(
    layer: nn.Module,
    r1: torch.Tensor,
) -> None:
    """
    Fuse R1 into one Qwen3-VL text decoder layer.

    Parameters:
        layer: Qwen3-VL text decoder layer.
        r1: Global hidden-dimension rotation matrix.
    """
    rotate_attention_inputs(layer, r1)
    rotate_attention_output(layer, r1)
    rotate_mlp_input(layer, r1)
    rotate_mlp_output(layer, r1)


@torch.no_grad()
def rotate_qwen3_vl_ov_r2(
    layer: nn.Module,
    r2: Optional[torch.Tensor],
    head_dim: int,
) -> None:
    """
    Fuse OV-side R2 into one Qwen3-VL text attention layer.

    Parameters:
        layer: Qwen3-VL text decoder layer.
        r2: Optional head-dimension R2 matrix.
        head_dim: Attention head dimension.
    """
    rotate_ov_proj(layer, r2, head_dim)


def _require_conv3d_attr(module: nn.Module, attr_name: str) -> nn.Conv3d:
    """
    Return a required Conv3d attribute from a module.

    Parameters:
        module: Parent module.
        attr_name: Attribute name.

    Returns:
        The resolved Conv3d module.
    """
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Expected attribute {attr_name!r} on module {type(module).__name__}."
        )
    value = getattr(module, attr_name)
    if not isinstance(value, nn.Conv3d):
        raise TypeError(
            f"Expected {attr_name!r} to be nn.Conv3d, got {type(value).__name__}."
        )
    return value


def _require_embedding_attr(module: nn.Module, attr_name: str) -> nn.Embedding:
    """
    Return a required Embedding attribute from a module.

    Parameters:
        module: Parent module.
        attr_name: Attribute name.

    Returns:
        The resolved Embedding module.
    """
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Expected attribute {attr_name!r} on module {type(module).__name__}."
        )
    value = getattr(module, attr_name)
    if not isinstance(value, nn.Embedding):
        raise TypeError(
            f"Expected {attr_name!r} to be nn.Embedding, got {type(value).__name__}."
        )
    return value


@torch.no_grad()
def _left_multiply_conv3d_output_(conv: nn.Conv3d, rotation_t: torch.Tensor) -> None:
    """
    Rotate the output-channel basis of a Conv3d module.

    Parameters:
        conv: Target Conv3d module.
        rotation_t: Transposed rotation matrix with shape ``[out_channels, out_channels]``.
    """
    weight = conv.weight.data
    out_channels = weight.shape[0]
    validate_square_rotation("conv output rotation", rotation_t, out_channels)

    rotation_t = rotation_t.to(device=weight.device, dtype=torch.float64)
    flat = weight.to(torch.float64).reshape(out_channels, -1)
    updated = rotation_t @ flat
    conv.weight.data.copy_(
        updated.reshape_as(weight).to(device=weight.device, dtype=weight.dtype)
    )

    if conv.bias is not None:
        bias = conv.bias.data
        updated_bias = rotation_t @ bias.to(torch.float64)
        conv.bias.data.copy_(updated_bias.to(device=bias.device, dtype=bias.dtype))


@torch.no_grad()
def _right_multiply_embedding_weight_(
    embedding: nn.Embedding,
    rotation: torch.Tensor,
) -> None:
    """
    Rotate an embedding table on its feature axis.

    Parameters:
        embedding: Target embedding module.
        rotation: Rotation matrix with shape ``[embedding_dim, embedding_dim]``.
    """
    weight = embedding.weight.data
    validate_square_rotation("embedding rotation", rotation, weight.shape[1])
    updated = weight.to(torch.float64) @ rotation.to(
        device=weight.device, dtype=torch.float64
    )
    embedding.weight.data.copy_(updated.to(device=weight.device, dtype=weight.dtype))


@torch.no_grad()
def _right_multiply_linear_weight_blockwise_(
    module: nn.Linear,
    rotation: torch.Tensor,
    block_size: int,
) -> None:
    """
    Right-multiply a Linear input axis by the same rotation in each block.

    Parameters:
        module: Target Linear module.
        rotation: Block rotation matrix.
        block_size: Size of each repeated input block.
    """
    weight = module.weight.data
    if weight.shape[1] % block_size != 0:
        raise ValueError(
            f"linear.in_features={weight.shape[1]} is not divisible by block_size={block_size}."
        )
    validate_square_rotation("blockwise input rotation", rotation, block_size)

    rotation = rotation.to(device=weight.device, dtype=torch.float64)
    out_features = weight.shape[0]
    num_blocks = weight.shape[1] // block_size
    working = weight.to(torch.float64).reshape(out_features, num_blocks, block_size)
    updated = working @ rotation
    module.weight.data.copy_(
        updated.reshape_as(weight).to(device=weight.device, dtype=weight.dtype)
    )


@torch.no_grad()
def _left_multiply_qkv_v_output_blockwise_(
    qkv: nn.Linear,
    rotation: torch.Tensor,
    vision_hidden_size: int,
    head_dim: int,
) -> None:
    """
    Rotate only the V slice of a fused QKV projection on its output axis.

    Parameters:
        qkv: Fused QKV Linear with output size ``3 * vision_hidden_size``.
        rotation: Per-head rotation matrix.
        vision_hidden_size: Vision hidden dimension.
        head_dim: Per-head dimension.
    """
    if qkv.out_features != 3 * vision_hidden_size:
        raise ValueError(
            "Expected fused qkv out_features to equal 3 * vision_hidden_size, "
            f"got {qkv.out_features} and {vision_hidden_size}."
        )
    if vision_hidden_size % head_dim != 0:
        raise ValueError(
            f"vision_hidden_size={vision_hidden_size} is not divisible by head_dim={head_dim}."
        )
    validate_square_rotation("vision R2", rotation, head_dim)

    rotation = rotation.to(device=qkv.weight.device, dtype=torch.float64)
    start = 2 * vision_hidden_size
    end = 3 * vision_hidden_size
    v_weight = qkv.weight.data[start:end]
    in_features = v_weight.shape[1]
    num_heads = vision_hidden_size // head_dim

    working_t = v_weight.to(torch.float64).T.reshape(in_features, num_heads, head_dim)
    updated_t = working_t @ rotation
    qkv.weight.data[start:end].copy_(
        updated_t.reshape(in_features, vision_hidden_size).T.to(
            device=v_weight.device, dtype=v_weight.dtype
        )
    )

    if qkv.bias is not None:
        v_bias = qkv.bias.data[start:end]
        bias_working = v_bias.to(torch.float64).reshape(num_heads, head_dim)
        updated_bias = bias_working @ rotation
        qkv.bias.data[start:end].copy_(
            updated_bias.reshape_as(v_bias).to(device=v_bias.device, dtype=v_bias.dtype)
        )


@torch.no_grad()
def rotate_qwen3_vl_vision_patch_and_position_embeddings(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
    r1: torch.Tensor,
) -> None:
    """
    Fuse vision R1 into the patch projection and absolute position embedding.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.
        r1: Vision hidden-dimension R1 rotation.
    """
    components = resolve_qwen3_vl_spinquant_components(model, config)
    visual = components.visual_model

    if not hasattr(visual, "patch_embed"):
        raise AttributeError("Expected vision model to expose `patch_embed`.")
    patch_proj = _require_conv3d_attr(visual.patch_embed, "proj")
    pos_embed = _require_embedding_attr(visual, "pos_embed")

    _left_multiply_conv3d_output_(patch_proj, r1.T)
    _right_multiply_embedding_weight_(pos_embed, r1)


@torch.no_grad()
def rotate_qwen3_vl_vision_layer_r1(
    layer: nn.Module,
    r1: torch.Tensor,
) -> None:
    """
    Fuse LayerNorm-compatible R1 into one Qwen3-VL vision transformer block.

    Parameters:
        layer: Qwen3-VL vision block.
        r1: Vision hidden-dimension R1 rotation.
    """
    qkv = require_linear_attr(layer.attn, "qkv")
    proj = require_linear_attr(layer.attn, "proj")
    linear_fc1 = require_linear_attr(layer.mlp, "linear_fc1")
    linear_fc2 = require_linear_attr(layer.mlp, "linear_fc2")

    right_multiply_linear_weight_(qkv, r1)
    left_multiply_linear_weight_(proj, r1.T)
    right_multiply_linear_weight_(linear_fc1, r1)
    left_multiply_linear_weight_(linear_fc2, r1.T)


@torch.no_grad()
def rotate_qwen3_vl_vision_merger_inputs(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
    r1: torch.Tensor,
) -> None:
    """
    Compensate vision mergers for a rotated vision residual basis.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.
        r1: Vision hidden-dimension R1 rotation.
    """
    components = resolve_qwen3_vl_spinquant_components(model, config)
    vision_hidden_size = get_qwen3_vl_vision_hidden_size(model)

    mergers = [components.visual_model.merger]
    mergers.extend(list(components.visual_deepstack_mergers))

    for merger in mergers:
        linear_fc1 = require_linear_attr(merger, "linear_fc1")
        _right_multiply_linear_weight_blockwise_(
            linear_fc1,
            r1,
            block_size=vision_hidden_size,
        )


@torch.no_grad()
def rotate_qwen3_vl_vision_ov_r2(
    layer: nn.Module,
    r2: Optional[torch.Tensor],
    head_dim: int,
) -> None:
    """
    Fuse OV-side R2 into one Qwen3-VL vision attention block.

    Parameters:
        layer: Qwen3-VL vision block.
        r2: Optional per-head R2 matrix. If None, this function is a no-op.
        head_dim: Vision attention head dimension.
    """
    if r2 is None:
        return

    qkv = require_linear_attr(layer.attn, "qkv")
    proj = require_linear_attr(layer.attn, "proj")

    if hasattr(layer.attn, "dim"):
        vision_hidden_size = int(layer.attn.dim)
    else:
        if qkv.out_features % 3 != 0:
            raise ValueError(
                f"Expected qkv.out_features to be divisible by 3, got {qkv.out_features}."
            )
        vision_hidden_size = qkv.out_features // 3

    _left_multiply_qkv_v_output_blockwise_(
        qkv,
        r2,
        vision_hidden_size=vision_hidden_size,
        head_dim=head_dim,
    )
    _right_multiply_linear_weight_blockwise_(proj, r2, block_size=head_dim)


@torch.no_grad()
def rotate_qwen3_vl_deepstack_outputs(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
    r1: torch.Tensor,
) -> None:
    """
    Fuse text R1 into Qwen3-VL DeepStack visual output projections.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.
        r1: Text global hidden-dimension rotation matrix.

    Notes:
        Main visual merger outputs are not fused here because they are inserted
        into ``inputs_embeds`` before the language-model entry rotation.
        DeepStack outputs are injected inside the text decoder, so they must be
        converted to the rotated residual basis.
    """
    if not config.fuse_deepstack_visual_outputs:
        return

    components = resolve_qwen3_vl_spinquant_components(model, config)
    for merger in components.visual_deepstack_mergers:
        linear_fc2 = require_linear_attr(merger, "linear_fc2")
        left_multiply_linear_weight_(linear_fc2, r1.T)
