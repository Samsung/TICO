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
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from tico.quantization.config.llama_attention import (
    DEFAULT_EXECUTION_PROFILE,
    ExecutionProfile,
    normalize_execution_profile,
)
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.utils import auto_qscheme_for
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.dtypes import DType as AffineDType
from tico.quantization.wrapq.dtypes import MXDtype, QuantDtype
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme


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


def _observer_from_dtype(qdtype: QuantDtype) -> Type[ObserverBase]:
    """
    Select a default observer class based on a quantization dtype.

    Parameters
    ----------
    qdtype : QuantDtype
        Quantization dtype used to select the observer.

    Returns
    -------
    Type[ObserverBase]
        ``MXObserver`` for MX dtypes, ``MinMaxObserver`` for integer dtypes.
    """
    if qdtype.is_mx:
        return MXObserver
    return MinMaxObserver


def _build_weight_override(
    weight_dtype: Optional[DType],
    *,
    observer: Optional[Type[ObserverBase]] = None,
) -> Dict[str, Any]:
    """
    Build a weight override dictionary.

    The override explicitly carries both dtype and qscheme so that local dtype
    changes do not accidentally inherit an incompatible or less suitable
    qscheme from an outer scope.

    Parameters
    ----------
    weight_dtype : Optional[DType]
        Explicit dtype for the weight observer.
    observer : Optional[Type[ObserverBase]]
        Explicit observer class for the weight. When ``None`` the observer
        is inferred from ``weight_dtype`` (MX → MXObserver, else MinMaxObserver).

    Returns
    -------
    Dict[str, Any]
        Weight override dictionary. Returns an empty dictionary when
        `weight_dtype` is None.
    """
    if weight_dtype is None:
        return {}
    resolved_observer = observer if observer is not None else _observer_from_dtype(weight_dtype)
    return {
        "weight": {
            "dtype": weight_dtype,
            "qscheme": auto_qscheme_for(weight_dtype, "weight"),
            "observer": resolved_observer,
        }
    }


def _build_activation_override(
    activation_observer: Optional[Type[ObserverBase]] = None,
    *,
    dtype: Optional[QuantDtype] = None,
) -> Dict[str, Any]:
    """
    Build an activation override dictionary (act_in / act_out).

    Parameters
    ----------
    activation_observer : Optional[Type[ObserverBase]]
        Observer class for both act_in and act_out. When ``None`` the
        observer is inferred from *dtype* (if provided).
    dtype : Optional[QuantDtype]
        Explicit dtype for both act_in and act_out observers.

    Returns
    -------
    Dict[str, Any]
        Activation override dictionary with ``act_in`` and ``act_out`` keys.
        Returns an empty dictionary when neither *activation_observer* nor
        *dtype* is provided.
    """
    if activation_observer is None and dtype is None:
        return {}

    resolved_observer = activation_observer
    if resolved_observer is None and dtype is not None:
        resolved_observer = _observer_from_dtype(dtype)
    if resolved_observer is None:
        resolved_observer = MinMaxObserver

    act_desc: Dict[str, Any] = {"observer": resolved_observer}
    if dtype is not None:
        act_desc["dtype"] = dtype
    return {
        "act_in": {**act_desc},
        "act_out": {**act_desc},
    }


def _build_norm_override(
    *,
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
    norm_io_dtype: Optional[QuantDtype] = None,
    norm_io_observer: Optional[Type[ObserverBase]] = None,
) -> Dict[str, Any]:
    """
    Build an override dictionary for a norm module.

    Parameters
    ----------
    norm_dtype : Optional[DType]
        Explicit module-level dtype override for the norm module.
    norm_weight_dtype : Optional[DType]
        Explicit weight dtype override for the norm weight.
    norm_io_dtype : Optional[QuantDtype]
        Explicit dtype for norm act_in / act_out observers.
        When provided, ``act_in`` and ``act_out`` overrides are emitted.
    norm_io_observer : Optional[Type[ObserverBase]]
        Explicit observer class for norm act_in / act_out.
        When ``None`` and *norm_io_dtype* is provided, the observer is
        inferred from the dtype.

    Returns
    -------
    Dict[str, Any]
        Override dictionary for the norm module. Returns an empty dictionary
        if no explicit override is requested.
    """
    override: Dict[str, Any] = {}

    if norm_dtype is not None:
        override["dtype"] = norm_dtype
        override["qscheme"] = auto_qscheme_for(norm_dtype)

    if norm_weight_dtype is not None:
        resolved_observer = _observer_from_dtype(norm_weight_dtype)
        override["weight"] = {
            "dtype": norm_weight_dtype,
            "qscheme": auto_qscheme_for(norm_weight_dtype, "weight"),
            "observer": resolved_observer,
        }

    if norm_io_dtype is not None or norm_io_observer is not None:
        io_override = _build_activation_override(
            norm_io_observer, dtype=norm_io_dtype
        )
        override.update(io_override)

    return override


def _build_llama_layer_overrides(
    *,
    linear_weight_dtype: Optional[DType],
    linear_activation_observer: Optional[Type[ObserverBase]] = None,
    linear_io_dtype: Optional[QuantDtype] = None,
    linear_io_observer: Optional[Type[ObserverBase]] = None,
    rms_norm_io_dtype: Optional[QuantDtype] = None,
    rms_norm_observer: Optional[Type[ObserverBase]] = None,
    softmax_dtype: Optional[QuantDtype] = None,
    softmax_observer: Optional[Type[ObserverBase]] = None,
    norm_dtype: Optional[DType] = None,
    norm_weight_dtype: Optional[DType] = None,
) -> Dict[str, Any]:
    """
    Build per-layer overrides for a Llama decoder block.

    The generated overrides cover:
      - self_attn.q_proj / k_proj / v_proj / o_proj  (weight + act_in/act_out)
      - self_attn.hidden, attn_mask, attn_out, logits  (activations)
      - self_attn.softmax, mask_add  (activations)
      - mlp.gate_proj / up_proj / down_proj  (weight + act_in/act_out)
      - mlp.act_in, mlp.mul  (activations)
      - input_layernorm / post_attention_layernorm  (weight + act_in/act_out)
      - attn_mask, self_attn_residual_out, mlp_residual_out  (decoder-layer-level activations)

    Parameters
    ----------
    linear_weight_dtype : Optional[DType]
        Explicit or resolved dtype applied to decoder-layer linear projection
        weights. If None, no linear weight override is emitted.
    linear_activation_observer : Optional[Type[ObserverBase]]
        Observer class for linear act_in / act_out. Kept for backward
        compatibility; prefer ``linear_io_dtype`` / ``linear_io_observer``.
    linear_io_dtype : Optional[QuantDtype]
        Dtype for linear-layer act_in / act_out and general-purpose
        activations (hidden, attn_mask, logits, mul, residual, …).
    linear_io_observer : Optional[Type[ObserverBase]]
        Observer class paired with *linear_io_dtype*.
    rms_norm_io_dtype : Optional[QuantDtype]
        Dtype for norm act_in / act_out and MLP act_in.
    rms_norm_observer : Optional[Type[ObserverBase]]
        Observer class paired with *rms_norm_io_dtype*.
    softmax_dtype : Optional[QuantDtype]
        Dtype for the softmax observer inside self_attn.
    softmax_observer : Optional[Type[ObserverBase]]
        Observer class paired with *softmax_dtype*.
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

    # --- Resolve linear I/O dtype / observer ---
    resolved_linear_io_dtype = linear_io_dtype
    resolved_linear_io_observer = linear_io_observer
    if resolved_linear_io_observer is None and resolved_linear_io_dtype is not None:
        resolved_linear_io_observer = _observer_from_dtype(resolved_linear_io_dtype)
    if resolved_linear_io_observer is None and linear_activation_observer is not None:
        resolved_linear_io_observer = linear_activation_observer

    # --- Resolve RMS norm I/O dtype / observer ---
    resolved_rms_io_dtype = rms_norm_io_dtype
    resolved_rms_observer = rms_norm_observer
    if resolved_rms_observer is None and resolved_rms_io_dtype is not None:
        resolved_rms_observer = _observer_from_dtype(resolved_rms_io_dtype)

    # --- Resolve softmax dtype / observer ---
    resolved_softmax_dtype = softmax_dtype
    resolved_softmax_observer = softmax_observer
    if resolved_softmax_observer is None and resolved_softmax_dtype is not None:
        resolved_softmax_observer = _observer_from_dtype(resolved_softmax_dtype)

    # --- Build linear projection override (weight + act_in + act_out) ---
    linear_override = _build_weight_override(linear_weight_dtype)
    linear_io_desc = _build_activation_override(
        resolved_linear_io_observer, dtype=resolved_linear_io_dtype
    )
    linear_override.update(linear_io_desc)
    if linear_override:
        _set_nested_override(layer_overrides, ("self_attn", "q_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "k_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "v_proj"), linear_override)
        _set_nested_override(layer_overrides, ("self_attn", "o_proj"), linear_override)

        _set_nested_override(layer_overrides, ("mlp", "gate_proj"), linear_override)
        _set_nested_override(layer_overrides, ("mlp", "up_proj"), linear_override)
        _set_nested_override(layer_overrides, ("mlp", "down_proj"), linear_override)

    # --- Self-attention fine-grained activation overrides ---
    if resolved_rms_io_dtype is not None or resolved_rms_observer is not None:
        rms_act_desc: Dict[str, Any] = {"observer": resolved_rms_observer or MinMaxObserver}
        if resolved_rms_io_dtype is not None:
            rms_act_desc["dtype"] = resolved_rms_io_dtype

    if resolved_linear_io_dtype is not None or resolved_linear_io_observer is not None:
        linear_act_desc: Dict[str, Any] = {"observer": resolved_linear_io_observer or MinMaxObserver}
        if resolved_linear_io_dtype is not None:
            linear_act_desc["dtype"] = resolved_linear_io_dtype
        _set_nested_override(
            layer_overrides, ("self_attn", "hidden"), {**linear_act_desc}
        )
        _set_nested_override(
            layer_overrides, ("self_attn", "attn_mask"), {**linear_act_desc}
        )
        _set_nested_override(
            layer_overrides, ("self_attn", "attn_out"), {**linear_act_desc}
        )
        _set_nested_override(
            layer_overrides, ("self_attn", "logits"), {**linear_act_desc}
        )

    if resolved_softmax_dtype is not None or resolved_softmax_observer is not None:
        softmax_act_desc: Dict[str, Any] = {"observer": resolved_softmax_observer or MinMaxObserver}
        if resolved_softmax_dtype is not None:
            softmax_act_desc["dtype"] = resolved_softmax_dtype
        _set_nested_override(
            layer_overrides, ("self_attn", "softmax"), {**softmax_act_desc}
        )
        _set_nested_override(
            layer_overrides, ("self_attn", "mask_add"), {**softmax_act_desc}
        )

    # --- MLP fine-grained activation overrides ---
    if resolved_rms_io_dtype is not None or resolved_rms_observer is not None:
        _set_nested_override(
            layer_overrides, ("mlp", "act_in"), {**rms_act_desc}
        )

    if resolved_linear_io_dtype is not None or resolved_linear_io_observer is not None:
        _set_nested_override(
            layer_overrides, ("mlp", "mul"), {**linear_act_desc}
        )

    # --- Norm overrides (weight + act_in + act_out) ---
    norm_override = _build_norm_override(
        norm_dtype=norm_dtype,
        norm_weight_dtype=norm_weight_dtype,
        norm_io_dtype=resolved_rms_io_dtype,
        norm_io_observer=resolved_rms_observer,
    )
    if norm_override:
        _set_nested_override(layer_overrides, ("input_layernorm",), norm_override)
        _set_nested_override(
            layer_overrides, ("post_attention_layernorm",), norm_override
        )

    # --- Decoder-layer-level activation overrides ---
    if resolved_linear_io_dtype is not None or resolved_linear_io_observer is not None:
        _set_nested_override(layer_overrides, ("attn_mask",), {**linear_act_desc})
        _set_nested_override(
            layer_overrides, ("self_attn_residual_out",), {**linear_act_desc}
        )
        _set_nested_override(
            layer_overrides, ("mlp_residual_out",), {**linear_act_desc}
        )

    return layer_overrides


def _build_llama_overrides(
    *,
    num_hidden_layers: int,
    linear_weight_dtype: Optional[DType],
    linear_activation_observer: Optional[Type[ObserverBase]] = None,
    linear_io_dtype: Optional[QuantDtype] = None,
    linear_io_observer: Optional[Type[ObserverBase]] = None,
    rms_norm_io_dtype: Optional[QuantDtype] = None,
    rms_norm_observer: Optional[Type[ObserverBase]] = None,
    softmax_dtype: Optional[QuantDtype] = None,
    softmax_observer: Optional[Type[ObserverBase]] = None,
    embedding_weight_dtype: Optional[DType] = None,
    lm_head_weight_dtype: Optional[DType] = None,
    spin_rotation_weight_dtype: Optional[DType] = None,
    norm_dtype: Optional[DType] = None,
    norm_weight_dtype: Optional[DType] = None,
) -> Dict[str, Any]:
    """
    Build PTQ overrides for a Llama-style causal LM.

    This helper generates overrides for:
      - input embedding: model.embed_tokens
      - SpinLlama input rotation: model.rotate_embedding
      - all decoder layers: model.layers.{i}
      - final model norm: model.norm
      - SpinLlama output rotation: rotate_lm_head
      - output projection: lm_head
      - model-level causal_mask activation

    Modules that are not explicitly overridden continue to use PTQConfig
    defaults. SpinLlama rotation overrides are emitted only when
    `spin_rotation_weight_dtype` is provided.

    Parameters
    ----------
    num_hidden_layers : int
        Number of decoder layers in the model.
    linear_weight_dtype : Optional[DType]
        Weight dtype override for decoder-layer linear projections.
    linear_activation_observer : Optional[Type[ObserverBase]]
        Observer class for linear act_in / act_out. Kept for backward
        compatibility; prefer ``linear_io_dtype`` / ``linear_io_observer``.
    linear_io_dtype : Optional[QuantDtype]
        Dtype for linear-layer act_in / act_out and general-purpose
        activations (attn_mask, logits, mul, residual, causal_mask, …).
    linear_io_observer : Optional[Type[ObserverBase]]
        Observer class paired with *linear_io_dtype*.
    rms_norm_io_dtype : Optional[QuantDtype]
        Dtype for norm act_in / act_out.
    rms_norm_observer : Optional[Type[ObserverBase]]
        Observer class paired with *rms_norm_io_dtype*.
    softmax_dtype : Optional[QuantDtype]
        Dtype for the softmax observer inside self_attn.
    softmax_observer : Optional[Type[ObserverBase]]
        Observer class paired with *softmax_dtype*.
    embedding_weight_dtype : Optional[DType]
        Weight dtype override for model.embed_tokens.weight.
    lm_head_weight_dtype : Optional[DType]
        Weight dtype override for lm_head.weight.
    spin_rotation_weight_dtype : Optional[DType]
        Weight dtype override for SpinLlama rotation weights.
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

    # --- Resolve linear I/O dtype / observer ---
    resolved_linear_io_dtype = linear_io_dtype
    resolved_linear_io_observer = linear_io_observer
    if resolved_linear_io_observer is None and resolved_linear_io_dtype is not None:
        resolved_linear_io_observer = _observer_from_dtype(resolved_linear_io_dtype)
    if resolved_linear_io_observer is None and linear_activation_observer is not None:
        resolved_linear_io_observer = linear_activation_observer

    # --- Resolve RMS norm I/O dtype / observer ---
    resolved_rms_io_dtype = rms_norm_io_dtype
    resolved_rms_observer = rms_norm_observer
    if resolved_rms_observer is None and resolved_rms_io_dtype is not None:
        resolved_rms_observer = _observer_from_dtype(resolved_rms_io_dtype)

    # --- Resolve softmax dtype / observer ---
    resolved_softmax_dtype = softmax_dtype
    resolved_softmax_observer = softmax_observer
    if resolved_softmax_observer is None and resolved_softmax_dtype is not None:
        resolved_softmax_observer = _observer_from_dtype(resolved_softmax_dtype)

    # --- Embedding ---
    embedding_override = _build_weight_override(embedding_weight_dtype)
    if embedding_override:
        _set_nested_override(overrides, ("model", "embed_tokens"), embedding_override)

    # --- LM head (full linear desc: weight + act_in + act_out) ---
    lm_head_override = _build_weight_override(lm_head_weight_dtype)
    lm_head_io = _build_activation_override(
        resolved_linear_io_observer, dtype=resolved_linear_io_dtype
    )
    lm_head_override.update(lm_head_io)
    if lm_head_override:
        overrides["lm_head"] = lm_head_override

    # --- Spin rotation ---
    spin_rotation_override = _build_weight_override(spin_rotation_weight_dtype)
    if spin_rotation_override:
        _set_nested_override(
            overrides, ("model", "rotate_embedding"), spin_rotation_override
        )
        _set_nested_override(overrides, ("rotate_lm_head",), spin_rotation_override)

    # --- Final model norm (weight + act_in + act_out) ---
    final_norm_override = _build_norm_override(
        norm_dtype=norm_dtype,
        norm_weight_dtype=norm_weight_dtype,
        norm_io_dtype=resolved_rms_io_dtype,
        norm_io_observer=resolved_rms_observer,
    )
    if final_norm_override:
        _set_nested_override(overrides, ("model", "norm"), final_norm_override)

    # --- Model-level causal_mask activation ---
    if resolved_linear_io_dtype is not None or resolved_linear_io_observer is not None:
        linear_act_desc: Dict[str, Any] = {"observer": resolved_linear_io_observer or MinMaxObserver}
        if resolved_linear_io_dtype is not None:
            linear_act_desc["dtype"] = resolved_linear_io_dtype
        _set_nested_override(overrides, ("model", "causal_mask"), {**linear_act_desc})

    # --- Decoder layers ---
    for layer_idx in range(num_hidden_layers):
        overrides["model"]["layers"][str(layer_idx)] = _build_llama_layer_overrides(
            linear_weight_dtype=linear_weight_dtype,
            linear_activation_observer=linear_activation_observer,
            linear_io_dtype=linear_io_dtype,
            linear_io_observer=linear_io_observer,
            rms_norm_io_dtype=rms_norm_io_dtype,
            rms_norm_observer=rms_norm_observer,
            softmax_dtype=softmax_dtype,
            softmax_observer=softmax_observer,
            norm_dtype=norm_dtype,
            norm_weight_dtype=norm_weight_dtype,
        )

    return overrides


def build_llm_ptq_config(
    *,
    model_type: str,
    num_hidden_layers: int,
    activation_dtype: DType = DType.int(16),
    default_qscheme: QScheme = QScheme.PER_TENSOR_SYMM,
    default_observer: Type[ObserverBase] = MinMaxObserver,
    linear_weight_bits: Optional[int] = None,
    linear_weight_dtype: Optional[DType] = None,
    linear_activation_observer: Optional[Type[ObserverBase]] = None,
    linear_io_dtype: Optional[QuantDtype] = None,
    linear_io_observer: Optional[Type[ObserverBase]] = None,
    rms_norm_io_dtype: Optional[QuantDtype] = None,
    rms_norm_observer: Optional[Type[ObserverBase]] = None,
    softmax_dtype: Optional[QuantDtype] = None,
    softmax_observer: Optional[Type[ObserverBase]] = None,
    embedding_weight_bits: Optional[int] = None,
    embedding_weight_dtype: Optional[DType] = None,
    lm_head_weight_bits: Optional[int] = None,
    lm_head_weight_dtype: Optional[DType] = None,
    spin_rotation_weight_bits: Optional[int] = None,
    spin_rotation_weight_dtype: Optional[DType] = None,
    norm_dtype: Optional[DType] = None,
    norm_weight_bits: Optional[int] = None,
    norm_weight_dtype: Optional[DType] = None,
    strict_wrap: bool = True,
    profile: ExecutionProfile = DEFAULT_EXECUTION_PROFILE,
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
    activation_dtype : DType, default=DType.int(16)
        Default dtype for observers that do not receive an explicit override.
    default_qscheme : QScheme, default=QScheme.PER_TENSOR_SYMM
        Default quantization scheme for observers that do not receive an
        explicit override.
    default_observer : Type[ObserverBase], default=MinMaxObserver
        Observer class to instantiate when no explicit observer is provided
        through overrides.
    linear_weight_bits : Optional[int], default=None
        Convenience bit-width for decoder-layer linear projection weights.
        Used only when `linear_weight_dtype` is not provided.
    linear_weight_dtype : Optional[DType], default=None
        Explicit dtype for decoder-layer linear projection weights.
    linear_activation_observer : Type[ObserverBase], default=MinMaxObserver
        Observer class for linear act_in / act_out. Kept for backward
        compatibility; prefer ``linear_io_dtype`` / ``linear_io_observer``.
    linear_io_dtype : Optional[QuantDtype], default=None
        Dtype for linear-layer act_in / act_out and general-purpose
        activations (attn_mask, logits, mul, residual, causal_mask, …).
        When ``None``, the ``linear_activation_observer`` is used without
        an explicit dtype (backward compatible).
    linear_io_observer : Optional[Type[ObserverBase]], default=None
        Observer class paired with *linear_io_dtype*. When ``None`` and
        *linear_io_dtype* is provided, the observer is inferred from the
        dtype (MX → MXObserver, integer → MinMaxObserver).
    rms_norm_io_dtype : Optional[QuantDtype], default=None
        Dtype for norm act_in / act_out and MLP act_in.
    rms_norm_observer : Optional[Type[ObserverBase]], default=None
        Observer class paired with *rms_norm_io_dtype*.
    softmax_dtype : Optional[QuantDtype], default=None
        Dtype for the softmax observer inside self_attn.
    softmax_observer : Optional[Type[ObserverBase]], default=None
        Observer class paired with *softmax_dtype*.
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
    spin_rotation_weight_bits : Optional[int], default=None
        Convenience bit-width for SpinLlama rotation weights:
        `model.rotate_embedding.weight` and `rotate_lm_head.weight`.
        Used only when `spin_rotation_weight_dtype` is not provided.
    spin_rotation_weight_dtype : Optional[DType], default=None
        Explicit dtype for SpinLlama rotation weights.
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
    profile : ExecutionProfile, default="npu_export"
        Execution profile stored as `PTQConfig.model_args["profile"]`.
        "reference_eval" selects a GPU-friendly, Hugging Face-like path.
        "npu_export" preserves the existing NPU-export-oriented graph.
        Advanced users may override or extend `qcfg.model_args` directly
        before calling `prepare()`.

    Returns
    -------
    PTQConfig
        PTQ configuration object ready to pass into `prepare()`.

    Raises
    ------
    NotImplementedError
        If the requested `model_type` is not supported.
    """
    profile = normalize_execution_profile(
        profile,
        context="build_llm_ptq_config.profile",
    )

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
    resolved_spin_rotation_weight_dtype = _resolve_weight_dtype(
        dtype=spin_rotation_weight_dtype,
        bits=spin_rotation_weight_bits,
    )
    resolved_norm_weight_dtype = _resolve_weight_dtype(
        dtype=norm_weight_dtype,
        bits=norm_weight_bits,
    )

    if model_type == "llama":
        overrides = _build_llama_overrides(
            num_hidden_layers=num_hidden_layers,
            linear_weight_dtype=resolved_linear_weight_dtype,
            linear_activation_observer=linear_activation_observer,
            linear_io_dtype=linear_io_dtype,
            linear_io_observer=linear_io_observer,
            rms_norm_io_dtype=rms_norm_io_dtype,
            rms_norm_observer=rms_norm_observer,
            softmax_dtype=softmax_dtype,
            softmax_observer=softmax_observer,
            embedding_weight_dtype=resolved_embedding_weight_dtype,
            lm_head_weight_dtype=resolved_lm_head_weight_dtype,
            spin_rotation_weight_dtype=resolved_spin_rotation_weight_dtype,
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
        default_observer=default_observer,
        overrides=overrides,
        model_args={"profile": profile},
        strict_wrap=strict_wrap,
    )


def _build_qwen3_vl_norm_override(
    *,
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
) -> Dict[str, Any]:
    """
    Build an override dictionary for Qwen3-VL norm modules.

    The generated override covers both RMSNorm-style observers used by text
    modules and LayerNorm-style observers used by vision modules.

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
        norm_qscheme = auto_qscheme_for(norm_dtype)
        override["dtype"] = norm_dtype
        override["qscheme"] = norm_qscheme

        # RMSNorm observers (used in text model)
        for obs_name in ["act_in", "act_out"]:
            override[obs_name] = {"qscheme": norm_qscheme}

        # LayerNorm observers (used in vision blocks)
        for obs_name in [
            "mean",
            "centered",
            "square",
            "var",
            "eps",
            "add_eps",
            "inv_std",
            "norm",
            "affine_mul",
            "affine_add",
        ]:
            override[obs_name] = {"qscheme": norm_qscheme}

    if norm_weight_dtype is not None:
        weight_qscheme = auto_qscheme_for(norm_weight_dtype, "weight")
        override["weight"] = {
            "dtype": norm_weight_dtype,
            "qscheme": weight_qscheme,
        }
        # Also handle bias with same dtype/qscheme as weight
        override["bias"] = {
            "dtype": norm_weight_dtype,
            "qscheme": weight_qscheme,
        }

    return override


def _build_qwen3_vl_vision_block_overrides(
    *,
    linear_weight_dtype: Optional[DType],
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
) -> Dict[str, Any]:
    """
    Build per-block overrides for a Qwen3-VL vision transformer block.

    The generated overrides can cover:
      - attn.qkv - combined Q/K/V projection
      - attn.proj - attention output projection
      - mlp.linear_fc1 - MLP first linear layer
      - mlp.linear_fc2 - MLP second linear layer
      - norm1 - first layer norm
      - norm2 - second layer norm

    Parameters
    ----------
    linear_weight_dtype : Optional[DType]
        Explicit or resolved dtype applied to vision block linear projection
        weights. If None, no linear override is emitted.
    norm_dtype : Optional[DType]
        Explicit module-level dtype override for norm modules.
    norm_weight_dtype : Optional[DType]
        Explicit weight dtype override for norm weights.

    Returns
    -------
    Dict[str, Any]
        Nested override dictionary for one vision block.
    """
    block_overrides: Dict[str, Any] = {}

    linear_override = _build_weight_override(linear_weight_dtype)

    if linear_override:
        _set_nested_override(block_overrides, ("attn", "qkv"), linear_override)
        _set_nested_override(block_overrides, ("attn", "proj"), linear_override)
        _set_nested_override(block_overrides, ("mlp", "linear_fc1"), linear_override)
        _set_nested_override(block_overrides, ("mlp", "linear_fc2"), linear_override)

    norm_override = _build_qwen3_vl_norm_override(
        norm_dtype=norm_dtype,
        norm_weight_dtype=norm_weight_dtype,
    )
    if norm_override:
        _set_nested_override(block_overrides, ("norm1",), norm_override)
        _set_nested_override(block_overrides, ("norm2",), norm_override)

    return block_overrides


def _build_qwen3_vl_vision_merger_overrides(
    *,
    linear_weight_dtype: Optional[DType],
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
) -> Dict[str, Any]:
    """
    Build overrides for a Qwen3-VL vision patch merger.

    The generated overrides can cover:
      - norm - LayerNorm before merging
      - linear_fc1 - first linear layer
      - linear_fc2 - second linear layer

    Parameters
    ----------
    linear_weight_dtype : Optional[DType]
        Explicit or resolved dtype applied to merger linear projection
        weights. If None, no linear override is emitted.
    norm_dtype : Optional[DType]
        Explicit module-level dtype override for norm module.
    norm_weight_dtype : Optional[DType]
        Explicit weight dtype override for norm weight.

    Returns
    -------
    Dict[str, Any]
        Nested override dictionary for a vision merger.
    """
    merger_overrides: Dict[str, Any] = {}

    linear_override = _build_weight_override(linear_weight_dtype)
    if linear_override:
        _set_nested_override(merger_overrides, ("linear_fc1",), linear_override)
        _set_nested_override(merger_overrides, ("linear_fc2",), linear_override)

    norm_override = _build_qwen3_vl_norm_override(
        norm_dtype=norm_dtype,
        norm_weight_dtype=norm_weight_dtype,
    )
    if norm_override:
        _set_nested_override(merger_overrides, ("norm",), norm_override)

    return merger_overrides


def _build_qwen3_vl_text_layer_overrides(
    *,
    linear_weight_dtype: Optional[DType],
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
) -> Dict[str, Any]:
    """
    Build per-layer overrides for a Qwen3-VL text decoder block.

    The generated overrides can cover:
      - self_attn.q_proj
      - self_attn.k_proj
      - self_attn.v_proj
      - self_attn.o_proj
      - self_attn.q_norm - query RMSNorm
      - self_attn.k_norm - key RMSNorm
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

    norm_override = _build_qwen3_vl_norm_override(
        norm_dtype=norm_dtype,
        norm_weight_dtype=norm_weight_dtype,
    )
    if norm_override:
        _set_nested_override(layer_overrides, ("input_layernorm",), norm_override)
        _set_nested_override(
            layer_overrides, ("post_attention_layernorm",), norm_override
        )
        # Qwen3-VL specific: attention norms for Q and K
        _set_nested_override(layer_overrides, ("self_attn", "q_norm"), norm_override)
        _set_nested_override(layer_overrides, ("self_attn", "k_norm"), norm_override)

    return layer_overrides


def _build_qwen3_vl_overrides(
    *,
    num_vision_blocks: int,
    num_text_layers: int,
    num_deepstack_mergers: int,
    linear_weight_dtype: Optional[DType],
    vision_patch_embed_weight_dtype: Optional[DType],
    embedding_weight_dtype: Optional[DType],
    lm_head_weight_dtype: Optional[DType],
    norm_dtype: Optional[DType],
    norm_weight_dtype: Optional[DType],
    quantize_vision: bool = True,
    quantize_text: bool = True,
    quantize_lm_head: bool = True,
) -> Dict[str, Any]:
    """
    Build PTQ overrides for a Qwen3-VL model.

    This helper generates overrides for:
      - Vision tower: model.visual.patch_embed, model.visual.blocks.{i},
                      model.visual.merger, model.visual.deepstack_merger_list.{i}
      - Text model: model.language_model.embed_tokens, model.language_model.layers.{i},
                    model.language_model.norm
      - Output: lm_head

    Parameters
    ----------
    num_vision_blocks : int
        Number of vision transformer blocks.
    num_text_layers : int
        Number of text decoder layers.
    num_deepstack_mergers : int
        Number of deepstack merger modules in the vision tower.
    linear_weight_dtype : Optional[DType]
        Weight dtype override for linear projections.
    vision_patch_embed_weight_dtype : Optional[DType]
        Weight dtype override for vision patch embed projection.
    embedding_weight_dtype : Optional[DType]
        Weight dtype override for embedding weights.
    lm_head_weight_dtype : Optional[DType]
        Weight dtype override for lm_head.weight.
    norm_dtype : Optional[DType]
        Module-level dtype override for norm modules.
    norm_weight_dtype : Optional[DType]
        Weight dtype override for norm weights.
    quantize_vision : bool
        Whether to quantize vision tower.
    quantize_text : bool
        Whether to quantize text model.
    quantize_lm_head : bool
        Whether to quantize lm_head.

    Returns
    -------
    Dict[str, Any]
        Nested override dictionary consumable by PTQConfig.
    """
    overrides: Dict[str, Any] = {"model": {}}

    # Vision tower overrides
    if quantize_vision:
        vision_overrides: Dict[str, Any] = {}

        # Patch embedding projection (Conv3d) - uses separate dtype
        patch_embed_override = _build_weight_override(vision_patch_embed_weight_dtype)
        if patch_embed_override:
            _set_nested_override(
                vision_overrides, ("patch_embed", "proj"), patch_embed_override
            )

        # Vision blocks
        vision_overrides["blocks"] = {}
        for block_idx in range(num_vision_blocks):
            vision_overrides["blocks"][
                str(block_idx)
            ] = _build_qwen3_vl_vision_block_overrides(
                linear_weight_dtype=linear_weight_dtype,
                norm_dtype=norm_dtype,
                norm_weight_dtype=norm_weight_dtype,
            )

        # Merger (has norm, linear_fc1, linear_fc2)
        merger_override = _build_qwen3_vl_vision_merger_overrides(
            linear_weight_dtype=linear_weight_dtype,
            norm_dtype=norm_dtype,
            norm_weight_dtype=norm_weight_dtype,
        )
        if merger_override:
            vision_overrides["merger"] = merger_override

        # Deepstack mergers (each has norm, linear_fc1, linear_fc2)
        if num_deepstack_mergers > 0:
            vision_overrides["deepstack_merger_list"] = {}
            deepstack_override = _build_qwen3_vl_vision_merger_overrides(
                linear_weight_dtype=linear_weight_dtype,
                norm_dtype=norm_dtype,
                norm_weight_dtype=norm_weight_dtype,
            )
            for merger_idx in range(num_deepstack_mergers):
                vision_overrides["deepstack_merger_list"][
                    str(merger_idx)
                ] = copy.deepcopy(deepstack_override)

        overrides["model"]["visual"] = vision_overrides

    # Text model overrides
    if quantize_text:
        text_overrides: Dict[str, Any] = {}

        # Text embedding
        embedding_override = _build_weight_override(embedding_weight_dtype)
        if embedding_override:
            _set_nested_override(text_overrides, ("embed_tokens",), embedding_override)

        # Text layers
        text_overrides["layers"] = {}
        for layer_idx in range(num_text_layers):
            text_overrides["layers"][
                str(layer_idx)
            ] = _build_qwen3_vl_text_layer_overrides(
                linear_weight_dtype=linear_weight_dtype,
                norm_dtype=norm_dtype,
                norm_weight_dtype=norm_weight_dtype,
            )

        # Final norm
        final_norm_override = _build_qwen3_vl_norm_override(
            norm_dtype=norm_dtype,
            norm_weight_dtype=norm_weight_dtype,
        )
        if final_norm_override:
            _set_nested_override(text_overrides, ("norm",), final_norm_override)

        overrides["model"]["language_model"] = text_overrides

    # LM head
    if quantize_lm_head:
        lm_head_override = _build_weight_override(lm_head_weight_dtype)
        if lm_head_override:
            overrides["lm_head"] = lm_head_override

    return overrides


def build_qwen3_vl_ptq_config(
    *,
    num_vision_blocks: int,
    num_text_layers: int,
    num_deepstack_mergers: int,
    model_args: Mapping[str, Any],
    activation_dtype: DType = DType.int(16),
    default_qscheme: QScheme = QScheme.PER_TENSOR_SYMM,
    default_observer: Type[ObserverBase] = MinMaxObserver,
    linear_weight_bits: Optional[int] = None,
    linear_weight_dtype: Optional[DType] = None,
    vision_patch_embed_weight_bits: Optional[int] = None,
    vision_patch_embed_weight_dtype: Optional[DType] = None,
    embedding_weight_bits: Optional[int] = None,
    embedding_weight_dtype: Optional[DType] = None,
    lm_head_weight_bits: Optional[int] = None,
    lm_head_weight_dtype: Optional[DType] = None,
    norm_dtype: Optional[DType] = None,
    norm_weight_bits: Optional[int] = None,
    norm_weight_dtype: Optional[DType] = None,
    quantize_vision: bool = True,
    quantize_text: bool = True,
    quantize_lm_head: bool = True,
    strict_wrap: bool = True,
) -> PTQConfig:
    """
    Build a PTQConfig for Qwen3-VL model.

    This helper generates PTQ configuration for the full Qwen3-VL model including:
      - Vision tower (patch_embed, blocks, merger, deepstack_mergers)
      - Text decoder (Llama-like layers)
      - Language modeling head

    Parameters
    ----------
    num_vision_blocks : int
        Number of vision transformer blocks in the vision tower.
    num_text_layers : int
        Number of decoder layers in the text model.
    num_deepstack_mergers : int, default=0
        Number of deepstack merger modules in the vision tower.
    activation_dtype : DType, default=DType.int(16)
    default_observer : Type[ObserverBase], default=MinMaxObserver
        Observer class to instantiate when no explicit observer is provided.
    linear_weight_bits : Optional[int], default=None
        Convenience bit-width for linear projection weights.
    linear_weight_dtype : Optional[DType], default=None
        Explicit dtype for linear projection weights.
    embedding_weight_bits : Optional[int], default=None
        Convenience bit-width for embedding weights.
    embedding_weight_dtype : Optional[DType], default=None
        Explicit dtype for embedding weights.
    lm_head_weight_bits : Optional[int], default=None
        Convenience bit-width for LM head weight.
    lm_head_weight_dtype : Optional[DType], default=None
        Explicit dtype for LM head weight.
    norm_dtype : Optional[DType], default=None
        Module-level dtype override for norm modules.
    norm_weight_bits : Optional[int], default=None
        Convenience bit-width for norm weights.
    norm_weight_dtype : Optional[DType], default=None
        Explicit dtype for norm weights.
    quantize_vision : bool, default=True
        Whether to quantize the vision tower.
    quantize_text : bool, default=True
        Whether to quantize the text model.
    quantize_lm_head : bool, default=True
        Whether to quantize the language modeling head.
    strict_wrap : bool, default=True
        If True, preparing a model will raise when a required module cannot be wrapped.

    Returns
    -------
    PTQConfig
        PTQ configuration object ready to pass into `prepare()`.
    """
    resolved_linear_weight_dtype = _resolve_weight_dtype(
        dtype=linear_weight_dtype,
        bits=linear_weight_bits,
    )
    resolved_vision_patch_embed_weight_dtype = _resolve_weight_dtype(
        dtype=vision_patch_embed_weight_dtype,
        bits=vision_patch_embed_weight_bits,
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

    overrides = _build_qwen3_vl_overrides(
        num_vision_blocks=num_vision_blocks,
        num_text_layers=num_text_layers,
        num_deepstack_mergers=num_deepstack_mergers,
        linear_weight_dtype=resolved_linear_weight_dtype,
        vision_patch_embed_weight_dtype=resolved_vision_patch_embed_weight_dtype,
        embedding_weight_dtype=resolved_embedding_weight_dtype,
        lm_head_weight_dtype=resolved_lm_head_weight_dtype,
        norm_dtype=norm_dtype,
        norm_weight_dtype=resolved_norm_weight_dtype,
        quantize_vision=quantize_vision,
        quantize_text=quantize_text,
        quantize_lm_head=quantize_lm_head,
    )

    return PTQConfig(
        default_dtype=activation_dtype,
        default_qscheme=default_qscheme,
        default_observer=default_observer,
        overrides=overrides,
        strict_wrap=strict_wrap,
        model_args=model_args,
    )
