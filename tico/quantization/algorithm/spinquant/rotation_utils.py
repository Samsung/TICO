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

from tico.quantization.algorithm.spinquant.hadamard_utils import (
    fuse_blockwise_transform_to_linear,
    is_pow2,
    make_random_hadamard_rotation,
)


def infer_device(model: nn.Module) -> torch.device:
    """
    Return the device of the first parameter in the model.

    Parameters:
        model: Target model.

    Returns:
        The device where the model currently resides.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def infer_dtype(model: nn.Module) -> torch.dtype:
    """
    Return the dtype of the first parameter in the model.

    Parameters:
        model: Target model.

    Returns:
        The dtype where the model currently resides.
    """
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def make_random_orthogonal_rotation(
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Create a random orthogonal matrix using QR decomposition.

    Parameters:
        size: Matrix dimension.
        device: Target device.
        dtype: Tensor dtype.

    Returns:
        A square orthogonal matrix.
    """
    random_matrix = torch.randn(size, size, device=device, dtype=dtype)
    q, r = torch.linalg.qr(random_matrix)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q


def validate_square_rotation(name: str, rotation: torch.Tensor, size: int) -> None:
    """
    Validate that a rotation tensor is square and matches the expected size.

    Parameters:
        name: Logical name used in error messages.
        rotation: Rotation matrix to validate.
        size: Expected matrix size.

    Raises:
        ValueError: If the rotation shape is invalid.
    """
    if rotation.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got ndim={rotation.ndim}.")
    if rotation.shape != (size, size):
        raise ValueError(
            f"{name} must have shape ({size}, {size}), got {tuple(rotation.shape)}."
        )


def copy_linear_weight_(module: nn.Linear, value: torch.Tensor) -> None:
    """
    Copy a tensor into a linear weight while preserving parameter dtype and device.

    Parameters:
        module: Destination linear module.
        value: Source tensor.
    """
    with torch.no_grad():
        module.weight.copy_(
            value.to(device=module.weight.device, dtype=module.weight.dtype)
        )


def right_multiply_linear_weight_(module: nn.Linear, rotation: torch.Tensor) -> None:
    """
    Right-multiply a linear weight by a rotation matrix.

    This updates:
        W <- W @ R

    Parameters:
        module: Target linear module with weight shape [out_features, in_features].
        rotation: Square rotation matrix of shape [in_features, in_features].

    Notes:
            PyTorch linear computes y = x @ W.T (row-vector convention).
            Right-multiplying W by R gives W_new = W @ R, so:
                y = x @ W_new.T = x @ R.T @ W.T
            This is equivalent to rotating the input by R.T before the original weight.
    """
    weight = module.weight.data
    rotation = rotation.to(device=weight.device, dtype=torch.float64)
    updated = weight.to(torch.float64) @ rotation
    module.weight.data.copy_(updated.to(device=weight.device, dtype=weight.dtype))


def left_multiply_linear_weight_(module: nn.Linear, rotation_t: torch.Tensor) -> None:
    """
    Left-multiply a linear weight by a transpose rotation matrix.

    This updates:
        W <- R^T @ W

    Parameters:
        module: Target linear module with weight shape [out_features, in_features].
        rotation_t: Transposed square rotation matrix of shape [out_features, out_features].

     Notes:
            The caller passes r1.T (i.e., rotation_t = R^T).
            This stores W_new = R^T @ W, so:
                y = x @ W_new.T = x @ W.T @ R
            Equivalent to rotating the output by R after the original weight.
    """
    weight = module.weight.data
    rotation_t = rotation_t.to(device=weight.device, dtype=torch.float64)
    updated = rotation_t @ weight.to(torch.float64)
    module.weight.data.copy_(updated.to(device=weight.device, dtype=weight.dtype))

    if module.bias is not None:
        bias = module.bias.data
        updated_bias = rotation_t @ bias.to(torch.float64)
        module.bias.data.copy_(updated_bias.to(device=bias.device, dtype=bias.dtype))


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Return the decoder layer list from a causal LM model.

    Parameters:
        model: Target model.

    Returns:
        The decoder layer container.

    Raises:
        AttributeError: If the expected LLaMA-style structure is not found.
    """
    if not hasattr(model, "model"):
        raise AttributeError(
            "Expected `model.model.layers` to exist, but `model` is missing."
        )
    if not hasattr(model.model, "layers"):
        raise AttributeError(
            "Expected `model.model.layers` to exist, but `model.model.layers` is missing."
        )
    return model.model.layers


def require_linear_attr(module: nn.Module, attr_name: str) -> nn.Linear:
    """
    Fetch and validate that a module attribute is an nn.Linear.

    Parameters:
        module: Parent module.
        attr_name: Attribute name.

    Returns:
        The linear module.

    Raises:
        AttributeError: If the attribute is missing.
        TypeError: If the attribute is not nn.Linear.
    """
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Expected attribute `{attr_name}` on module `{type(module).__name__}`."
        )
    value = getattr(module, attr_name)
    if not isinstance(value, nn.Linear):
        raise TypeError(
            f"Expected `{attr_name}` to be nn.Linear, got {type(value).__name__}."
        )
    return value


def build_r1(
    model: nn.Module, init_method: str, r1: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Resolve the global hidden-dimension rotation matrix R1.

    Parameters:
        model: Target model.
        init_method: Rotation initialization mode.
        r1: Optional externally supplied R1 matrix.

    Returns:
        A square rotation matrix of shape [hidden_size, hidden_size].

    Raises:
        ValueError: If the configuration is invalid.
    """
    hidden_size = int(model.config.hidden_size)
    device = infer_device(model)

    if init_method == "random":
        return make_random_orthogonal_rotation(hidden_size, device=device)

    if init_method == "hadamard":
        return make_random_hadamard_rotation(hidden_size, device=device)

    if init_method == "external":
        if r1 is None:
            raise ValueError("`r1` must be provided when init_method='external'.")
        rotation = r1.to(device=device, dtype=torch.float64)
        validate_square_rotation("r1", rotation, hidden_size)
        return rotation

    raise ValueError(f"Unsupported init_method: {init_method}")


def build_r2(
    *,
    init_method: str,
    r2_map: Optional[dict[str, torch.Tensor]],
    layer_idx: int,
    head_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Resolve the per-layer head-dimension rotation matrix R2.

    Parameters:
        init_method: Rotation initialization mode.
        r2_map: Optional mapping from module keys to per-layer R2 matrices.
        layer_idx: Decoder layer index.
        head_dim: Attention head dimension.
        device: Target device.

    Returns:
        A square rotation matrix of shape [head_dim, head_dim], or None if
        the external configuration does not provide a layer-specific R2.
    """
    if init_method == "random":
        return make_random_orthogonal_rotation(head_dim, device=device)

    if init_method == "hadamard":
        return make_random_hadamard_rotation(head_dim, device=device)

    if init_method == "external":
        if r2_map is None:
            return None

        key = f"model.layers.{layer_idx}.self_attn.R2"
        rotation = r2_map.get(key)
        if rotation is None:
            return None

        rotation = rotation.to(device=device, dtype=torch.float64)
        validate_square_rotation(key, rotation, head_dim)
        return rotation

    raise ValueError(f"Unsupported init_method: {init_method}")


@torch.no_grad()
def extract_and_reset_final_norm_scale(model: nn.Module) -> torch.Tensor:
    """
    Extract the final norm affine weight and reset it to identity.

    Parameters:
        model: Target SpinLlama / LLaMA-style model.

    Returns:
        The original final norm weight as float64 tensor on the model device.

    Raises:
        AttributeError: If the expected final norm is missing.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "norm"):
        raise AttributeError("Expected `model.model.norm` to exist.")

    norm = model.model.norm
    if not hasattr(norm, "weight") or norm.weight is None:
        raise AttributeError("Expected final norm to expose `weight`.")

    gamma = (
        norm.weight.data.detach()
        .to(
            device=norm.weight.device,
            dtype=torch.float64,
        )
        .clone()
    )

    norm.weight.data.copy_(torch.ones_like(norm.weight.data))
    assert not hasattr(norm, "bias")

    return gamma


def apply_embedding_side_rotation(model: nn.Module, r1: torch.Tensor) -> None:
    """
    Store the embedding-side rotation in `model.model.rotate_embedding`.

    Parameters:
        model: Target model.
        r1: Global hidden-dimension rotation matrix.

    Notes:
        The original SpinQuant offline fusion rotates the embedding table as:
            E <- E @ R1

        Since this code must preserve tied embeddings, the equivalent runtime
        transform is inserted as:
            x <- rotate_embedding(x)

        For nn.Linear, output = x @ W^T, so to realize x @ R1 we must store:
            W = R1^T
    """
    if not hasattr(model, "model") or not hasattr(model.model, "rotate_embedding"):
        raise AttributeError(
            "Expected `model.model.rotate_embedding` for tied-embedding-safe SpinQuant."
        )

    rotate_embedding = require_linear_attr(model.model, "rotate_embedding")
    copy_linear_weight_(rotate_embedding, r1.T)


@torch.no_grad()
def apply_lm_head_side_rotation(
    model: nn.Module,
    r1: torch.Tensor,
    norm_scale: Optional[torch.Tensor] = None,
) -> None:
    """
    Store the LM-head-side correction transform in `model.rotate_lm_head`.

    Parameters:
        model: Target model.
        r1: Global hidden-dimension rotation matrix.
        norm_scale: Optional final norm affine weight (gamma).

    Notes:
        In the rotated model, the decoder operates in the rotated basis:
            h_rot = h @ R1

        The final RMSNorm (with weight gamma) is applied after the decoder:
            x = RMSNorm_gamma(h_rot)

        Since RMSNorm (without affine) is rotation-equivariant:
            RMSNorm_1(h @ R1) = RMSNorm_1(h) @ R1

        If we remove the affine (set gamma = 1), we must compensate it here.

        The original model expects:
            logits = lm_head(RMSNorm_gamma(h))

        In the rotated model (with norm affine removed), we have:
            x = RMSNorm_1(h) @ R1

        To match the original behavior, we need:
            x -> x @ R1^T @ D_gamma

        Therefore, the runtime transform becomes:
            x -> x @ (R1^T @ D_gamma)

        For nn.Linear, output = x @ W^T, so we must store:
            W = (R1^T @ D_gamma)^T = D_gamma @ R1

        If `norm_scale` is None, this falls back to pure rotation correction:
            x -> x @ R1^T
            W = R1
    """
    if not hasattr(model, "rotate_lm_head"):
        raise AttributeError(
            "Expected `rotate_lm_head` for tied-embedding-safe SpinQuant."
        )

    rotate_lm_head = require_linear_attr(model, "rotate_lm_head")

    if norm_scale is None:
        copy_linear_weight_(rotate_lm_head, r1)
        return

    gamma = norm_scale.to(device=r1.device, dtype=torch.float64)
    if gamma.ndim != 1 or gamma.numel() != r1.shape[0]:
        raise ValueError(
            f"`norm_scale` must have shape [{r1.shape[0]}], got {tuple(gamma.shape)}."
        )

    # W = D_gamma @ R1
    weight = gamma.view(-1, 1) * r1
    copy_linear_weight_(rotate_lm_head, weight)


def rotate_attention_inputs(layer: nn.Module, r1: torch.Tensor) -> None:
    """
    Apply R1 to Q, K, and V input-side weights.

    Parameters:
        layer: Decoder layer.
        r1: Global hidden-dimension rotation matrix.
    """
    q_proj = require_linear_attr(layer.self_attn, "q_proj")
    k_proj = require_linear_attr(layer.self_attn, "k_proj")
    v_proj = require_linear_attr(layer.self_attn, "v_proj")

    right_multiply_linear_weight_(q_proj, r1)
    right_multiply_linear_weight_(k_proj, r1)
    right_multiply_linear_weight_(v_proj, r1)


def rotate_attention_output(layer: nn.Module, r1: torch.Tensor) -> None:
    """
    Apply R1^T to the attention output projection.

    Parameters:
        layer: Decoder layer.
        r1: Global hidden-dimension rotation matrix.
    """
    o_proj = require_linear_attr(layer.self_attn, "o_proj")
    left_multiply_linear_weight_(o_proj, r1.T)


def rotate_mlp_input(layer: nn.Module, r1: torch.Tensor) -> None:
    """
    Apply R1 to the MLP input-side projections.

    Parameters:
        layer: Decoder layer.
        r1: Global hidden-dimension rotation matrix.
    """
    gate_proj = require_linear_attr(layer.mlp, "gate_proj")
    up_proj = require_linear_attr(layer.mlp, "up_proj")

    right_multiply_linear_weight_(gate_proj, r1)
    right_multiply_linear_weight_(up_proj, r1)


def rotate_mlp_output(layer: nn.Module, r1: torch.Tensor) -> None:
    """
    Apply R1^T to the MLP output projection and fuse exact Hadamard.

    Parameters:
        layer: Decoder layer.
        r1: Global hidden-dimension rotation matrix.
    """
    down_proj = require_linear_attr(layer.mlp, "down_proj")
    left_multiply_linear_weight_(down_proj, r1.T)


def rotate_ov_proj(layer: nn.Module, r2: Optional[torch.Tensor], head_dim: int) -> None:
    """
    Fuse the attention OV-side exact Hadamard and optional R2 rotations.

    Parameters:
        layer: Decoder layer.
        r2: Optional per-layer head-dimension rotation matrix.
        head_dim: Attention head dimension.

    Raises:
        ValueError: If `head_dim` is not a power of two.
    """
    if not is_pow2(head_dim):
        raise ValueError(
            f"SpinQuant OV fusion requires a power-of-two head_dim, got {head_dim}."
        )

    v_proj = require_linear_attr(layer.self_attn, "v_proj")
    o_proj = require_linear_attr(layer.self_attn, "o_proj")

    fuse_blockwise_transform_to_linear(
        v_proj,
        had_dim=head_dim,
        output=True,
        R2=r2,
    )
    fuse_blockwise_transform_to_linear(
        o_proj,
        had_dim=head_dim,
        output=False,
        R2=r2,
    )
