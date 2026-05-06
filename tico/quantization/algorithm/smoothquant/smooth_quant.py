# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import Callable, Dict, List, Optional

import torch


@torch.no_grad()
def smooth_weights(
    front_module: torch.nn.Module,
    back_modules: torch.nn.Module | List[torch.nn.Module],
    activation_max: torch.Tensor,
    alpha: float,
):
    """
    Applies SmoothQuant-style smoothing to the weights and biases of two
     connected modules using activation maximum values.

    NOTE All modules **MUST** have `weight` and optionally `bias` attributes.

    Parameters
    -----------
        front_module
            The front module whose weights and biases will be adjusted.
        back_modules
            A list of back modules whose weights and biases will be adjusted.
        activation_max
            A tensor of channel-wise maximum activation values for the front module.
        alpha
            The smoothing factor that determines the scaling for weight adjustments.

    Raises
    -------
    AttributeError
        If `front_module` or any module in `back_modules` does not have `weight` attributes.
    ValueError
        If the shape of tensors in `activation_max` does not match the number of channels
         in `front_module`'s weight.
    NoteImplementedError
        If `front_module` or any module in `back_modules` is of an unsupported type.
    """
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    if not isinstance(back_modules, list):
        back_modules = [back_modules]

    # Check attributes
    if not hasattr(front_module, "weight"):
        raise AttributeError(
            f"The front module '{type(front_module).__name__}' does not have a 'weight' attribute."
        )
    for back_m in back_modules:
        if not hasattr(back_m, "weight"):
            raise AttributeError(
                f"The front module '{type(back_m).__name__}' does not have a 'weight' attribute."
            )
    # Check shapes
    if isinstance(front_module, LlamaRMSNorm):
        front_numel = front_module.weight.numel()
    elif isinstance(front_module, torch.nn.LayerNorm):
        front_numel = front_module.weight.numel()
    else:
        raise NotImplementedError(
            f"Unsupported module type: {type(front_module).__name__}"
        )
    for back_m in back_modules:
        if isinstance(back_m, torch.nn.Linear):
            back_numel = back_m.in_features
        else:
            raise NotImplementedError(
                f"Unsupported module type: {type(front_module).__name__}"
            )

        if front_numel != back_numel or back_numel != activation_max.numel():
            raise ValueError(
                f"Shape mismatch: front_numel({front_numel}), back_numel({back_numel}), activation_max_numel({activation_max.numel()})"
            )

    # Compute scales
    device, dtype = back_modules[0].weight.device, back_modules[0].weight.dtype
    activation_max = activation_max.to(device=device, dtype=dtype)  # type: ignore[arg-type]
    weight_scales = torch.cat(
        [back_m.weight.abs().max(dim=0, keepdim=True)[0] for back_m in back_modules],  # type: ignore[operator]
        dim=0,
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (activation_max.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)  # type: ignore[arg-type]
        .to(dtype)  # type: ignore[arg-type]
    )

    # Smooth
    front_module.weight.div_(scales)
    if hasattr(front_module, "bias"):
        front_module.bias.div_(scales)

    for back_m in back_modules:
        back_m.weight.mul_(scales.view(1, -1))  # type: ignore[operator]


# TODO Split the files per model
# ────────────────────────────────────────────────────────────
# fairseq ReLU bridge (input-hook stats) helpers
# ────────────────────────────────────────────────────────────


@torch.no_grad()
def _compute_s_for_linear(
    linear_like: torch.nn.Module,  # 2D weight [out, in]
    activation_max: torch.Tensor,  # shape [in]
    alpha: float,
) -> torch.Tensor:
    """
    s = (amax^alpha / w_col_max^(1-alpha))
      - amax: channel-wise max of the input to this module
      - w_col_max: max(|W|) per input column
    """
    if not hasattr(linear_like, "weight"):
        raise RuntimeError(f"{type(linear_like).__name__} has no 'weight' attribute.")
    W = linear_like.weight  # [out, in]
    assert isinstance(W, torch.Tensor)
    if W.ndim != 2:
        raise RuntimeError(
            f"Expected 2D weight, got {W.ndim}D for {type(linear_like).__name__}"
        )

    device, dtype = W.device, W.dtype
    amax = activation_max.to(device=device, dtype=dtype)

    if amax.numel() != W.shape[1]:
        raise ValueError(
            f"activation_max numel({amax.numel()}) != in_features({W.shape[1]})"
        )

    w_col_max = W.abs().max(dim=0)[0].clamp(min=1e-5)  # [in]
    s = (amax.pow(alpha) / w_col_max.pow(1.0 - alpha)).clamp(min=1e-5)  # [in]
    return s


@torch.no_grad()
def _fuse_relu_bridge_no_runtime_mul(
    fc1: torch.nn.Module,
    fc2: torch.nn.Module,
    s_hidden: torch.Tensor,
):
    """
    Fuse scaling across fc1 → ReLU → fc2 without runtime multiplies:
      - fc1 rows *= 1/s, (fc1.bias *= 1/s)
      - fc2 cols *= s
    Assumes middle activation is ReLU (positive homogeneous).
    """
    if not hasattr(fc1, "weight") or not hasattr(fc2, "weight"):
        raise RuntimeError("fc1/fc2 must have 'weight' attributes.")

    W1, W2 = fc1.weight, fc2.weight
    assert isinstance(W1, torch.Tensor) and isinstance(W2, torch.Tensor)
    if W1.ndim != 2 or W2.ndim != 2:
        raise RuntimeError("fc1/fc2 weights must be 2D.")

    hidden = W1.shape[0]
    if W2.shape[1] != hidden or s_hidden.numel() != hidden:
        raise ValueError(
            f"Dimension mismatch: hidden={hidden}, W2.in={W2.shape[1]}, s={s_hidden.numel()}"
        )

    s = s_hidden.to(device=W1.device, dtype=W1.dtype).clamp(min=1e-5)  # [hidden]
    inv_s = (1.0 / s).clamp(min=1e-5)

    # fc1: row-wise scale
    W1.mul_(inv_s.view(-1, 1))
    if hasattr(fc1, "bias") and getattr(fc1, "bias") is not None:
        assert isinstance(fc1.bias, torch.Tensor)
        fc1.bias.mul_(inv_s)

    # fc2: column-wise scale
    W2.mul_(s.view(1, -1))


# ────────────────────────────────────────────────────────────
# Per-layer appliers (uniform protocol): return True if applied, else False
# ────────────────────────────────────────────────────────────


@torch.no_grad()
def _apply_if_llama_decoder(
    name: str,
    module: torch.nn.Module,
    activation_max: Dict[str, torch.Tensor],
    alpha_to_apply: float,
) -> bool:
    """
    Apply LLaMA decoder-layer smoothing (input-hook stats).
    Returns True if this handler applied smoothing to `module`.
    """
    try:
        from transformers.models.llama.modeling_llama import (  # type: ignore
            LlamaDecoderLayer,
        )
    except Exception:
        return False

    if not isinstance(module, LlamaDecoderLayer):
        return False

    attn_ln = module.input_layernorm
    qkv = [
        module.self_attn.q_proj,
        module.self_attn.k_proj,
        module.self_attn.v_proj,
    ]
    # Input-hook stats for q_proj input
    qkv_input_scales = activation_max[name + ".self_attn.q_proj"]
    smooth_weights(attn_ln, qkv, qkv_input_scales, alpha_to_apply)

    ffn_ln = module.post_attention_layernorm
    fcs = [module.mlp.gate_proj, module.mlp.up_proj]
    # Input-hook stats for gate_proj input
    fcs_input_scales = activation_max[name + ".mlp.gate_proj"]
    smooth_weights(ffn_ln, fcs, fcs_input_scales, alpha_to_apply)

    return True


@torch.no_grad()
def _apply_if_fairseq_relu_bridge(
    name: str,
    module: torch.nn.Module,
    activation_max: Dict[str, torch.Tensor],
    alpha_to_apply: float,
) -> bool:
    """
    Apply fairseq Transformer (Encoder/Decoder) ReLU-FFN bridge fusion
    using input-hook stats at '{name}.fc1'. Returns True if applied.
    """
    try:
        from fairseq.modules.transformer_layer import (
            TransformerDecoderLayerBase,
            TransformerEncoderLayerBase,
        )  # type: ignore
    except Exception:
        return False

    if not isinstance(
        module, (TransformerEncoderLayerBase, TransformerDecoderLayerBase)
    ):
        return False

    # Only when FFN activation is ReLU (positive homogeneity)
    act_fn = getattr(module, "activation_fn", None)
    is_relu = (act_fn is torch.nn.functional.relu) or getattr(
        act_fn, "__name__", ""
    ) == "relu"
    if not is_relu:
        return False

    fc1_key = f"{name}.fc1"
    amax2 = activation_max.get(fc1_key)
    if amax2 is None:
        return False

    fc1 = getattr(module, "fc1", None)
    fc2 = getattr(module, "fc2", None)
    if fc1 is None or fc2 is None or not hasattr(fc2, "weight") or fc2.weight.ndim != 2:
        return False

    s_hidden = _compute_s_for_linear(fc2, amax2, alpha_to_apply)  # [hidden]
    _fuse_relu_bridge_no_runtime_mul(fc1, fc2, s_hidden)
    return True


# ────────────────────────────────────────────────────────────
# Qwen3-VL Vision Components (LayerNorm-based)
# ────────────────────────────────────────────────────────────


@torch.no_grad()
def _apply_if_qwen3vl_vision_block(
    name: str,
    module: torch.nn.Module,
    activation_max: Dict[str, torch.Tensor],
    alpha_to_apply: float,
) -> bool:
    """
    Apply SmoothQuant smoothing to Qwen3VLVisionBlock (LayerNorm-based).

    Qwen3VLVisionBlock structure:
        - norm1 (LayerNorm) → attn (attention)
        - norm2 (LayerNorm) → mlp (feed-forward)

    Returns True if this handler applied smoothing to `module`.
    """
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock
    except Exception:
        return False

    if not isinstance(module, Qwen3VLVisionBlock):
        return False

    # Check for required attributes
    if not hasattr(module, "norm1") or not hasattr(module, "norm2"):
        return False
    if not hasattr(module, "attn") or not hasattr(module, "mlp"):
        return False

    # Get attention input projections (qkv is typically combined or separate)
    attn = module.attn
    if hasattr(attn, "qkv"):
        # Combined qkv projection
        back_modules_attn = [attn.qkv]
    elif (
        hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj")
    ):
        # Separate q, k, v projections
        back_modules_attn = [attn.q_proj, attn.k_proj, attn.v_proj]
    else:
        # Try to find any linear layers in attention
        back_modules_attn = []
        for attr_name in ["q", "k", "v", "query", "key", "value"]:
            if hasattr(attn, attr_name):
                attr = getattr(attn, attr_name)
                if isinstance(attr, torch.nn.Linear):
                    back_modules_attn.append(attr)
        if not back_modules_attn:
            return False

    # Smooth norm1 → attention
    applied_attn = False
    attn_key = f"{name}.attn"
    for key_suffix in ["", ".qkv", ".q_proj", ".q", ".query"]:
        potential_key = f"{attn_key}{key_suffix}"
        if potential_key in activation_max:
            smooth_weights(
                module.norm1,
                back_modules_attn,
                activation_max[potential_key],
                alpha_to_apply,
            )
            applied_attn = True
            break

    if not applied_attn:
        print(
            f"[SmoothQuant] Warning: activation stats not found for "
            f"{name} attention input."
        )

    # Get MLP input projections
    mlp = module.mlp
    back_modules_mlp = []
    for attr_name in ["linear_fc1", "fc1", "up_proj", "gate_proj", "w1"]:
        if hasattr(mlp, attr_name):
            attr = getattr(mlp, attr_name)
            if isinstance(attr, torch.nn.Linear):
                back_modules_mlp.append(attr)

    if not back_modules_mlp:
        return True  # Already applied attention smoothing

    # Smooth norm2 → mlp
    applied_mlp = False
    mlp_key = f"{name}.mlp"
    for key_suffix in [".linear_fc1", ".fc1", ".up_proj", ".gate_proj", ".w1", ""]:
        potential_key = f"{mlp_key}{key_suffix}"
        if potential_key in activation_max:
            smooth_weights(
                module.norm2,
                back_modules_mlp,
                activation_max[potential_key],
                alpha_to_apply,
            )
            applied_mlp = True
            break

    if not applied_mlp:
        print(
            f"[SmoothQuant] Warning: activation stats not found for "
            f"{name} mlp input."
        )

    return True


@torch.no_grad()
def _apply_if_qwen3vl_vision_patch_merger(
    name: str,
    module: torch.nn.Module,
    activation_max: Dict[str, torch.Tensor],
    alpha_to_apply: float,
) -> bool:
    """
    Apply SmoothQuant smoothing to Qwen3VLVisionPatchMerger (LayerNorm-based).

    Qwen3VLVisionPatchMerger structure:
        - norm (LayerNorm) → linear_fc1 → act_fn → linear_fc2

    NOTE: In Qwen3-VL, the PatchMerger has a special structure where:
        - norm.weight shape: [1024] (normalized_shape)
        - linear_fc1.in_features: 4096 (hidden_size)

    This means the LayerNorm is applied to a smaller dimension, then reshaped
    before the linear layer. SmoothQuant cannot be directly applied here
    because the dimensions don't match. We skip smoothing for this module.

    Returns True if this handler applied smoothing to `module`, False otherwise.
    """
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchMerger,
        )
    except Exception:
        return False

    if not isinstance(module, Qwen3VLVisionPatchMerger):
        return False

    # Check for required attributes
    if not hasattr(module, "norm"):
        return False
    if not hasattr(module, "linear_fc1"):
        return False

    # Check if dimensions are compatible for SmoothQuant
    # LayerNorm weight shape = normalized_shape
    # Linear in_features must match normalized_shape for smoothing
    norm_numel = module.norm.weight.numel()
    linear_in_features = module.linear_fc1.in_features

    if norm_numel != linear_in_features:
        # Dimensions don't match - PatchMerger reshapes between norm and linear
        # SmoothQuant cannot be applied here
        return False

    # Smooth norm → linear_fc1
    fc1_key = f"{name}.linear_fc1"
    if fc1_key in activation_max:
        fc1_input_scales = activation_max[fc1_key]
        smooth_weights(
            module.norm, [module.linear_fc1], fc1_input_scales, alpha_to_apply
        )
        return True
    else:
        print(
            f"[SmoothQuant] Warning: activation stats not found for "
            f"{name} linear_fc1 input."
        )

    return False


# Registry of appliers (order matters: try LLaMA first, then Qwen3-VL vision, then fairseq)
_APPLIERS: List[
    Callable[[str, torch.nn.Module, Dict[str, torch.Tensor], float], bool]
] = [
    _apply_if_llama_decoder,
    _apply_if_qwen3vl_vision_block,
    _apply_if_qwen3vl_vision_patch_merger,
    _apply_if_fairseq_relu_bridge,
]


@torch.no_grad()
def apply_smoothing(
    model: torch.nn.Module,
    activation_max: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    custom_alpha_map: Optional[Dict[str, float]] = None,
):
    """
    Applies SmoothQuant-style smoothing to the model's weights using activation maximum values.

    Parameters
    -----------
        model
            A torch module whose weights will be smoothed.
        activation_max
            The channel-wise maximum activation values for the model.
        alpha
            The default smoothing factor to apply across all modules.
        custom_alpha_map
            A dictionary mapping layer/module names to custom alpha values.
            Layers specified in this dictionary will use the corresponding alpha
             value instead of the default.
    """
    for name, module in model.named_modules():
        alpha_to_apply = (
            custom_alpha_map.get(name, alpha) if custom_alpha_map else alpha
        )
        if alpha_to_apply > 1.0:
            raise RuntimeError(
                f"Alpha value cannot exceed 1.0. Given alpha: {alpha_to_apply}"
            )

        # Try each applier until one succeeds.
        for applier in _APPLIERS:
            if applier(name, module, activation_max, alpha_to_apply):
                break  # applied → stop trying others
