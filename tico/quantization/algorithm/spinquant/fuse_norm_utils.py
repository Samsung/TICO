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

from typing import Iterable

import torch
import torch.nn as nn


def _require_attr(module: nn.Module, attr_name: str) -> nn.Module:
    """
    Return a required attribute from a module.

    Parameters:
        module: Parent module.
        attr_name: Attribute name.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If the attribute does not exist.
    """
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Expected attribute `{attr_name}` on `{type(module).__name__}`."
        )
    return getattr(module, attr_name)


def _require_linear(module: nn.Module, attr_name: str) -> nn.Linear:
    """
    Return a required nn.Linear attribute from a module.

    Parameters:
        module: Parent module.
        attr_name: Attribute name.

    Returns:
        The linear submodule.

    Raises:
        AttributeError: If the attribute does not exist.
        TypeError: If the attribute is not nn.Linear.
    """
    value = _require_attr(module, attr_name)
    if not isinstance(value, nn.Linear):
        raise TypeError(
            f"Expected `{attr_name}` to be nn.Linear, got {type(value).__name__}."
        )
    return value


@torch.no_grad()
def fuse_norm_into_linears(
    norm: nn.Module,
    linear_layers: Iterable[nn.Linear],
) -> None:
    """
    Fold an elementwise normalization affine into the following linear layers.

    This function assumes the normalization contributes an elementwise affine
    transform of the form:

        y = x * gamma + beta

    and folds it into each adjacent linear:

        linear(y) = linear(x * gamma + beta)

    which becomes:

        W <- W * gamma
        b <- b + W @ beta

    For LLaMA RMSNorm, beta is typically absent, so only the weight scaling path
    is used.

    Parameters:
        norm: Normalization module that exposes `weight` and optionally `bias`.
        linear_layers: Linear layers that immediately consume the normalized tensor.

    Raises:
        AttributeError: If the normalization module does not expose `weight`.
        TypeError: If any item in `linear_layers` is not nn.Linear.
        ValueError: If shapes are incompatible.
    """
    if not hasattr(norm, "weight") or norm.weight is None:
        raise AttributeError(
            f"Normalization module `{type(norm).__name__}` must expose `weight`."
        )

    gamma = norm.weight.data.to(torch.float64)
    beta = None
    if hasattr(norm, "bias") and norm.bias is not None:
        beta = norm.bias.data.to(torch.float64)

    for linear in linear_layers:
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear in `linear_layers`, got {type(linear).__name__}."
            )

        weight = linear.weight.data
        if weight.shape[1] != gamma.numel():
            raise ValueError(
                "Normalization weight size must match linear input dimension, "
                f"but got norm={gamma.numel()} and linear.in_features={weight.shape[1]}."
            )

        original_dtype = weight.dtype
        original_device = weight.device

        w64 = weight.to(torch.float64)

        # Fold the elementwise scale into the input axis of the linear layer.
        fused_w = w64 * gamma.unsqueeze(0)
        linear.weight.data.copy_(
            fused_w.to(device=original_device, dtype=original_dtype)
        )

        if beta is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(
                    torch.zeros(
                        linear.out_features,
                        device=original_device,
                        dtype=original_dtype,
                    )
                )

            b64 = linear.bias.data.to(torch.float64)
            fused_b = b64 + (w64 @ beta)
            linear.bias.data.copy_(
                fused_b.to(device=original_device, dtype=original_dtype)
            )


@torch.no_grad()
def reset_norm_affine_to_identity(norm: nn.Module) -> None:
    """
    Reset a normalization affine to identity.

    Parameters:
        norm: Normalization module.

    Notes:
        - `weight` is set to ones.
        - `bias` is set to zeros when present.
    """
    if hasattr(norm, "weight") and norm.weight is not None:
        norm.weight.data.copy_(torch.ones_like(norm.weight.data))

    if hasattr(norm, "bias") and norm.bias is not None:
        norm.bias.data.zero_()


@torch.no_grad()
def center_embedding_weights(embedding: nn.Embedding) -> None:
    """
    Center each embedding row by subtracting its mean.

    Parameters:
        embedding: Target embedding layer.
    """
    weight = embedding.weight.data
    centered = weight.to(torch.float64)
    centered = centered - centered.mean(dim=-1, keepdim=True)
    embedding.weight.data.copy_(centered.to(dtype=weight.dtype, device=weight.device))


@torch.no_grad()
def fuse_spinquant_layer_norms(
    model: nn.Module,
    *,
    center_input_embeddings: bool = False,
    fuse_lm_head: bool = False,
) -> nn.Module:
    """
    Apply the normalization-folding step required before SpinQuant-style rotation.

    By default, this function intentionally skips:
        - token embedding centering
        - final norm -> lm_head fusion

    This behavior is suitable when tied embeddings must be preserved and the
    caller does not want to modify token embedding or lm_head weights.

    Parameters:
        model: Target causal LM model with a LLaMA-style structure.
        center_input_embeddings: Whether to center token embedding rows.
        fuse_lm_head: Whether to fold final norm into lm_head.

    Returns:
        The same model instance for convenience.

    Raises:
        AttributeError: If expected submodules are missing.
    """
    if getattr(model, "_spinquant_norms_fused", False):
        return model

    if not hasattr(model, "model"):
        raise AttributeError("Expected model to have a `model` submodule.")
    if not hasattr(model.model, "layers"):
        raise AttributeError("Expected model.model to have `layers`.")
    if not hasattr(model.model, "norm"):
        raise AttributeError("Expected model.model to have final `norm`.")

    if center_input_embeddings:
        embed_tokens = _require_attr(model.model, "embed_tokens")
        if not isinstance(embed_tokens, nn.Embedding):
            raise TypeError(
                f"Expected `embed_tokens` to be nn.Embedding, got {type(embed_tokens).__name__}."
            )
        center_embedding_weights(embed_tokens)

    for layer in model.model.layers:
        input_norm = _require_attr(layer, "input_layernorm")
        post_attn_norm = _require_attr(layer, "post_attention_layernorm")

        q_proj = _require_linear(layer.self_attn, "q_proj")
        k_proj = _require_linear(layer.self_attn, "k_proj")
        v_proj = _require_linear(layer.self_attn, "v_proj")
        gate_proj = _require_linear(layer.mlp, "gate_proj")
        up_proj = _require_linear(layer.mlp, "up_proj")

        # Fold input RMSNorm into attention input projections.
        fuse_norm_into_linears(input_norm, [q_proj, k_proj, v_proj])

        # Fold post-attention RMSNorm into MLP input projections.
        fuse_norm_into_linears(post_attn_norm, [gate_proj, up_proj])

        # After folding, the norm affine should become identity.
        reset_norm_affine_to_identity(input_norm)
        reset_norm_affine_to_identity(post_attn_norm)

    if fuse_lm_head:
        if not hasattr(model, "lm_head"):
            raise AttributeError("Expected model to have `lm_head`.")
        final_norm = model.model.norm
        lm_head = model.lm_head
        if not isinstance(lm_head, nn.Linear):
            raise TypeError(
                f"Expected `lm_head` to be nn.Linear, got {type(lm_head).__name__}."
            )

        fuse_norm_into_linears(final_norm, [lm_head])
        reset_norm_affine_to_identity(final_norm)

    setattr(model, "_spinquant_norms_fused", True)
    return model
