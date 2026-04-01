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

import copy
from typing import Any, Dict, Optional, Tuple

from tico.quantization.config.ptq import PTQConfig, WrapperVariant
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme


def _auto_qscheme_for(dtype: DType, obs_name: Optional[str] = None) -> QScheme:
    """
    Choose the default qscheme associated with a dtype and observer name.

    Default policy:
      - signed dtype   -> symmetric per-tensor
      - unsigned dtype -> asymmetric per-tensor
      - unsigned weight -> asymmetric per-channel
    """
    if not dtype.signed:
        if obs_name == "weight":
            return QScheme.PER_CHANNEL_ASYMM
        return QScheme.PER_TENSOR_ASYMM
    return QScheme.PER_TENSOR_SYMM


def _weight_dtype_from_bits(bits: int) -> DType:
    """
    Convert a commonly used bit-width into a corresponding quantized dtype.

    This helper provides a simple mapping for frequently used quantization
    settings. It is intended as a convenience fallback when an explicit dtype
    is not provided by the user.

    Currently supported mappings:
      - 16 → int16
      - 8  → uint8
      - 4  → uint4

    Parameters
    ----------
    bits : int
        Target bit-width for weight quantization.

    Returns
    -------
    DType
        Quantized dtype corresponding to the given bit-width.

    Raises
    ------
    ValueError
        If the provided bit-width is not supported.
    """
    if bits == 16:
        return DType.int(16)
    elif bits == 8:
        return DType.uint(8)
    elif bits == 4:
        return DType.uint(4)

    raise ValueError(
        f"Unsupported bit-width: {bits}. "
        "Supported values are {16, 8, 4}. "
        "Please provide an explicit dtype instead."
    )


def _resolve_weight_dtype(
    *,
    dtype: Optional[DType],
    bits: Optional[int],
) -> Optional[DType]:
    """
    Resolve a weight dtype from either an explicit dtype or a bit-width.

    Resolution order:
      1. explicit dtype
      2. bit-width fallback
      3. None

    Parameters
    ----------
    dtype : Optional[DType]
        Explicit dtype requested by the caller.
    bits : Optional[int]
        Bit-width shorthand used when no explicit dtype is provided.

    Returns
    -------
    Optional[DType]
        Resolved weight dtype, or None if neither input is specified.
    """
    if dtype is not None:
        return dtype
    if bits is not None:
        return _weight_dtype_from_bits(bits)
    return None


def _set_nested_override(
    root: Dict[str, Any],
    path: Tuple[str, ...],
    value: Dict[str, Any],
) -> None:
    """
    Set an override value at the given nested path.

    Parameters
    ----------
    root : Dict[str, Any]
        Root override dictionary to update in-place.
    path : Tuple[str, ...]
        Sequence of keys describing the nested override path.
    value : Dict[str, Any]
        Override payload to assign at the target path.
    """
    current = root
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = copy.deepcopy(value)


def _build_weight_override(weight_dtype: Optional[DType]) -> Dict[str, Any]:
    """
    Build a weight override dictionary.

    The override explicitly carries both dtype and qscheme so that local dtype
    changes do not accidentally inherit an incompatible or less suitable
    qscheme from an outer scope.

    Parameters
    ----------
    weight_dtype : Optional[DType]
        Explicit dtype for the weight observer.

    Returns
    -------
    Dict[str, Any]
        Weight override dictionary. Returns an empty dictionary when
        `weight_dtype` is None.
    """
    if weight_dtype is None:
        return {}
    return {
        "weight": {
            "dtype": weight_dtype,
            "qscheme": _auto_qscheme_for(weight_dtype, "weight"),
        }
    }


def _build_norm_override(
    *,
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
) -> Dict[str, Any]:
    """
    Build an override dictionary for a norm module.

    Parameters
    ----------
    norm_dtype : Optional[DType]
        Explicit module-level dtype override for the norm module.
    norm_weight_dtype : Optional[DType]
        Explicit weight dtype override for the norm weight.

    Returns
    -------
    Dict[str, Any]
        Override dictionary for the norm module. Returns an empty dictionary
        if no explicit override is requested.
    """
    override: Dict[str, Any] = {}

    if norm_dtype is not None:
        override["dtype"] = norm_dtype
        override["qscheme"] = _auto_qscheme_for(norm_dtype)

    if norm_weight_dtype is not None:
        override["weight"] = {
            "dtype": norm_weight_dtype,
            "qscheme": _auto_qscheme_for(norm_weight_dtype, "weight"),
        }

    return override


def _build_llama_layer_overrides(
    *,
    linear_weight_dtype: Optional[DType],
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
) -> Dict[str, Any]:
    """
    Build per-layer overrides for a Llama decoder block.

    The generated overrides can cover:
      - self_attn.q_proj
      - self_attn.k_proj
      - self_attn.v_proj
      - self_attn.o_proj
      - mlp.gate_proj
      - mlp.up_proj
      - mlp.down_proj
      - input_layernorm
      - post_attention_layernorm

    Parameters
    ----------
    linear_weight_dtype : Optional[DType]
        Explicit or resolved dtype applied to decoder-layer linear projection
        weights. If None, no linear override is emitted.
    norm_dtype : Optional[DType]
        Explicit module-level dtype override for per-layer norm modules.
    norm_weight_dtype : Optional[DType]
        Explicit weight dtype override for per-layer norm weights.

    Returns
    -------
    Dict[str, Any]
        Nested override dictionary for one decoder layer.
    """
    layer_overrides: Dict[str, Any] = {}

    linear_override = _build_weight_override(linear_weight_dtype)
    if linear_override:
        _set_nested_override(layer_overrides, ("self_attn", "q_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "k_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "v_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "o_proj"), linear_override)

        _set_nested_override(layer_overrides, ("mlp", "gate_proj"), linear_override)
        _set_nested_override(layer_overrides, ("mlp", "up_proj"), linear_override)
        _set_nested_override(layer_overrides, ("mlp", "down_proj"), linear_override)

    norm_override = _build_norm_override(
        norm_dtype=norm_dtype,
        norm_weight_dtype=norm_weight_dtype,
    )
    if norm_override:
        _set_nested_override(layer_overrides, ("input_layernorm",), norm_override)
        _set_nested_override(
            layer_overrides, ("post_attention_layernorm",), norm_override
        )

    return layer_overrides


def _build_llama_overrides(
    *,
    num_hidden_layers: int,
    linear_weight_dtype: Optional[DType],
    embedding_weight_dtype: Optional[DType],
    lm_head_weight_dtype: Optional[DType],
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
) -> Dict[str, Any]:
    """
    Build PTQ overrides for a Llama-style causal LM.

    This helper generates overrides for:
      - input embedding: model.embed_tokens
      - all decoder layers: model.layers.{i}
      - final model norm: model.norm
      - output projection: lm_head

    Modules that are not explicitly overridden continue to use PTQConfig
    defaults.

    Parameters
    ----------
    num_hidden_layers : int
        Number of decoder layers in the model.
    linear_weight_dtype : Optional[DType]
        Weight dtype override for decoder-layer linear projections.
    embedding_weight_dtype : Optional[DType]
        Weight dtype override for model.embed_tokens.weight.
    lm_head_weight_dtype : Optional[DType]
        Weight dtype override for lm_head.weight.
    norm_dtype : Optional[DType]
        Module-level dtype override for norm modules.
    norm_weight_dtype : Optional[DType]
        Weight dtype override for norm weights.

    Returns
    -------
    Dict[str, Any]
        Nested override dictionary consumable by PTQConfig.
    """
    overrides: Dict[str, Any] = {
        "model": {
            "layers": {},
        }
    }

    embedding_override = _build_weight_override(embedding_weight_dtype)
    if embedding_override:
        _set_nested_override(overrides, ("model", "embed_tokens"), embedding_override)

    lm_head_override = _build_weight_override(lm_head_weight_dtype)
    if lm_head_override:
        overrides["lm_head"] = lm_head_override

    final_norm_override = _build_norm_override(
        norm_dtype=norm_dtype,
        norm_weight_dtype=norm_weight_dtype,
    )
    if final_norm_override:
        _set_nested_override(overrides, ("model", "norm"), final_norm_override)

    for layer_idx in range(num_hidden_layers):
        overrides["model"]["layers"][str(layer_idx)] = _build_llama_layer_overrides(
            linear_weight_dtype=linear_weight_dtype,
            norm_dtype=norm_dtype,
            norm_weight_dtype=norm_weight_dtype,
        )

    return overrides


def build_llm_ptq_config(
    *,
    model_type: str,
    num_hidden_layers: int,
    wrapper_variant: WrapperVariant = "prefill",
    activation_dtype: DType = DType.int(16),
    default_qscheme: QScheme = QScheme.PER_TENSOR_SYMM,
    linear_weight_bits: Optional[int] = None,
    linear_weight_dtype: Optional[DType] = None,
    embedding_weight_bits: Optional[int] = None,
    embedding_weight_dtype: Optional[DType] = None,
    lm_head_weight_bits: Optional[int] = None,
    lm_head_weight_dtype: Optional[DType] = None,
    norm_dtype: Optional[DType] = None,
    norm_weight_bits: Optional[int] = None,
    norm_weight_dtype: Optional[DType] = None,
    strict_wrap: bool = True,
) -> PTQConfig:
    """
    Build a PTQConfig for an LLM using model-family-aware override generation.

    This helper reduces repetitive manual construction of `PTQConfig.overrides`
    for common LLM quantization setups. The public API is model-agnostic, while
    the internal override generation is dispatched by `model_type`.

    Explicit dtypes take precedence over bit-width shorthands. Bit-width-based
    resolution is preserved as a convenience fallback for existing usage.

    Currently supported model types:
      - "llama"

    Parameters
    ----------
    model_type : str
        Model family used to select the internal override mapping logic.
    num_hidden_layers : int
        Number of decoder layers in the model.
    wrapper_variant : WrapperVariant, default="prefill"
        Wrapper specialization to request when resolving quantized wrappers.
    activation_dtype : DType, default=DType.int(16)
        Default dtype for observers that do not receive an explicit override.
    default_qscheme : QScheme, default=QScheme.PER_TENSOR_SYMM
        Default quantization scheme for observers that do not receive an
        explicit override.
    linear_weight_bits : Optional[int], default=None
        Convenience bit-width for decoder-layer linear projection weights.
        Used only when `linear_weight_dtype` is not provided.
    linear_weight_dtype : Optional[DType], default=None
        Explicit dtype for decoder-layer linear projection weights.
    embedding_weight_bits : Optional[int], default=None
        Convenience bit-width for the input embedding weight.
        Used only when `embedding_weight_dtype` is not provided.
    embedding_weight_dtype : Optional[DType], default=None
        Explicit dtype for the input embedding weight.
    lm_head_weight_bits : Optional[int], default=None
        Convenience bit-width for the LM head weight.
        Used only when `lm_head_weight_dtype` is not provided.
    lm_head_weight_dtype : Optional[DType], default=None
        Explicit dtype for the LM head weight.
    norm_dtype : Optional[DType], default=None
        Explicit module-level dtype override for norm modules.
    norm_weight_bits : Optional[int], default=None
        Convenience bit-width for norm weights.
        Used only when `norm_weight_dtype` is not provided.
    norm_weight_dtype : Optional[DType], default=None
        Explicit dtype for norm weights.
    strict_wrap : bool, default=True
        If True, preparing a model will raise when a required module cannot be
        wrapped.

    Returns
    -------
    PTQConfig
        PTQ configuration object ready to pass into `prepare()`.

    Raises
    ------
    NotImplementedError
        If the requested `model_type` is not supported.
    """
    resolved_linear_weight_dtype = _resolve_weight_dtype(
        dtype=linear_weight_dtype,
        bits=linear_weight_bits,
    )
    resolved_embedding_weight_dtype = _resolve_weight_dtype(
        dtype=embedding_weight_dtype,
        bits=embedding_weight_bits,
    )
    resolved_lm_head_weight_dtype = _resolve_weight_dtype(
        dtype=lm_head_weight_dtype,
        bits=lm_head_weight_bits,
    )
    resolved_norm_weight_dtype = _resolve_weight_dtype(
        dtype=norm_weight_dtype,
        bits=norm_weight_bits,
    )

    if model_type == "llama":
        overrides = _build_llama_overrides(
            num_hidden_layers=num_hidden_layers,
            linear_weight_dtype=resolved_linear_weight_dtype,
            embedding_weight_dtype=resolved_embedding_weight_dtype,
            lm_head_weight_dtype=resolved_lm_head_weight_dtype,
            norm_dtype=norm_dtype,
            norm_weight_dtype=resolved_norm_weight_dtype,
        )
    else:
        raise NotImplementedError(
            f"Unsupported model_type: {model_type!r}. " "Currently supported: ['llama']"
        )

    return PTQConfig(
        default_dtype=activation_dtype,
        default_qscheme=default_qscheme,
        wrapper_variant=wrapper_variant,
        overrides=overrides,
        strict_wrap=strict_wrap,
    )
