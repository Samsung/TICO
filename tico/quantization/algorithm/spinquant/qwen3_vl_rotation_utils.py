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


def build_qwen3_vl_r1(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
) -> torch.Tensor:
    """
    Build the Qwen3-VL global hidden-dimension R1 rotation.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.

    Returns:
        A square rotation matrix with shape ``[hidden_size, hidden_size]``.

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
    Build or fetch a Qwen3-VL per-layer head-dimension R2 rotation.

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


@torch.no_grad()
def rotate_qwen3_vl_deepstack_outputs(
    model: nn.Module,
    config: Qwen3VLSpinQuantConfig,
    r1: torch.Tensor,
) -> None:
    """
    Fuse R1 into Qwen3-VL DeepStack visual output projections.

    Parameters:
        model: Target Qwen3-VL model.
        config: Qwen3-VL SpinQuant configuration.
        r1: Global hidden-dimension rotation matrix.

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
