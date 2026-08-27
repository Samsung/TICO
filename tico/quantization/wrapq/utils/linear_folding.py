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

from typing import Union

import torch
import torch.nn as nn


AffineValue = Union[float, int, torch.Tensor]
_REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR = (
    "_tico_reused_weight_qparam_scale_multiplier"
)


def fold_input_affine_into_linear(
    linear: nn.Linear,
    *,
    scale: AffineValue,
    shift: AffineValue,
) -> nn.Linear:
    """Return a new linear layer that absorbs an affine input transform.

    The returned layer is equivalent to ``linear(x * scale + shift)`` for
    scalar values or vectors with ``linear.in_features`` elements. Its
    parameters are computed as ``W' = W * scale`` and ``b' = b + W @ shift``.
    The source layer is not modified.

    This helper is intended for inference-time rewrites before calibration and
    quantization. Training the returned layer independently does not preserve
    the parameter coupling of the original affine-plus-linear expression.

    The returned layer also records whether weight qparams produced before the
    fold can be reused. Positive uniform input scaling permits reuse after
    multiplying each weight scale by the same factor. Other scaling patterns
    are marked as unsupported so qparam handoff fails instead of silently
    loading incompatible values.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            "fold_input_affine_into_linear expects nn.Linear, "
            f"got {type(linear).__name__}."
        )
    if not torch.is_floating_point(linear.weight):
        raise TypeError(
            "fold_input_affine_into_linear requires floating-point weights, "
            f"got {linear.weight.dtype}."
        )

    scale_vector = _as_feature_vector(
        scale,
        name="scale",
        in_features=linear.in_features,
        reference=linear.weight,
    )
    shift_vector = _as_feature_vector(
        shift,
        name="shift",
        in_features=linear.in_features,
        reference=linear.weight,
    )
    use_bias = linear.bias is not None or bool(torch.any(shift_vector != 0).item())

    folded = nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=use_bias,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        folded.weight.copy_(linear.weight * scale_vector.unsqueeze(0))
        if folded.bias is not None:
            bias = torch.mv(linear.weight, shift_vector)
            if linear.bias is not None:
                bias = bias + linear.bias
            folded.bias.copy_(bias)

    folded.weight.requires_grad_(linear.weight.requires_grad)
    if folded.bias is not None:
        bias_requires_grad = (
            linear.bias.requires_grad
            if linear.bias is not None
            else linear.weight.requires_grad
        )
        folded.bias.requires_grad_(bias_requires_grad)
    folded.train(linear.training)
    setattr(
        folded,
        _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR,
        _uniform_positive_scale_multiplier(scale_vector),
    )
    return folded


def reused_weight_qparam_scale_multiplier(module: nn.Module) -> float:
    """Return the weight-scale multiplier required by a folded linear layer.

    Modules without input-affine fold metadata return ``1.0``. A folded layer
    with positive uniform input scaling returns that scaling value. A folded
    layer whose scaling cannot safely reuse pre-fold weight qparams raises a
    ``RuntimeError`` so callers can recompute qparams instead.
    """
    parameter_owner = getattr(module, "module", module)
    if not hasattr(parameter_owner, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR):
        return 1.0

    multiplier = getattr(
        parameter_owner,
        _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR,
    )
    if multiplier is None:
        raise RuntimeError(
            "Pre-fold weight qparams cannot be reused after non-uniform or "
            "non-positive input-affine scaling. Recompute the weight qparams."
        )
    return float(multiplier)


def _as_feature_vector(
    value: AffineValue,
    *,
    name: str,
    in_features: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Convert one scalar or feature vector to the linear weight domain."""
    tensor = torch.as_tensor(
        value,
        device=reference.device,
        dtype=reference.dtype,
    )
    if tensor.ndim == 0 or tensor.shape == (1,):
        tensor = tensor.reshape(1).expand(in_features)
    elif tensor.shape != (in_features,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({in_features},), "
            f"got {tuple(tensor.shape)}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor.detach()


def _uniform_positive_scale_multiplier(scale: torch.Tensor) -> float | None:
    """Return one reusable qparam multiplier for positive uniform scaling."""
    if scale.numel() == 0:
        return None
    first = scale.reshape(-1)[0]
    if not bool(torch.all(scale == first).item()) or not bool((first > 0).item()):
        return None
    return float(first.item())
