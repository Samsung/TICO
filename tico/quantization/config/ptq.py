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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Optional, Type

from tico.quantization.config.base import BaseConfig
from tico.quantization.config.utils import auto_qscheme_for, dtype_is_unsigned
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme


ExportMode = Literal["prefill", "decode"]


def _resolve_qscheme(
    *,
    dtype: DType,
    qscheme: Optional[QScheme],
    context: str,
    obs_name: Optional[str] = None,
) -> QScheme:
    """
    Resolve a dtype/qscheme pair using the option-C policy.

    Resolution policy:
      1. If `qscheme` is None, infer it from `dtype` and `obs_name`.
      2. If the caller explicitly provides an incompatible pair, raise.
    """
    resolved_qscheme = qscheme or auto_qscheme_for(dtype, obs_name)

    if dtype_is_unsigned(dtype) and resolved_qscheme.is_symmetric():
        raise ValueError(
            f"Invalid quantization config at {context}: unsigned dtype "
            f"{dtype!r} cannot be paired with symmetric qscheme "
            f"{resolved_qscheme!r}."
        )

    return resolved_qscheme


def _normalize_overrides(
    mapping: Mapping[str, Any],
    *,
    inherited_dtype: DType,
    inherited_qscheme: QScheme,
    context: str,
    current_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Recursively normalize and validate nested override mappings.

    Any node that provides `dtype` but omits `qscheme` receives an inferred
    qscheme derived from that dtype. Explicit incompatible pairs are rejected
    immediately.

    The current mapping key is tracked as `current_name` so that special
    observer names such as `weight` can receive a more suitable automatic
    default qscheme.
    """
    normalized: Dict[str, Any] = dict(mapping)

    local_dtype = normalized.get("dtype", inherited_dtype)
    local_qscheme = normalized.get("qscheme", inherited_qscheme)

    if "dtype" in normalized:
        normalized["qscheme"] = _resolve_qscheme(
            dtype=local_dtype,
            qscheme=normalized.get("qscheme"),
            context=context,
            obs_name=current_name,
        )
        local_qscheme = normalized["qscheme"]
    elif "qscheme" in normalized:
        local_qscheme = _resolve_qscheme(
            dtype=local_dtype,
            qscheme=normalized["qscheme"],
            context=context,
            obs_name=current_name,
        )
    else:
        _resolve_qscheme(
            dtype=local_dtype,
            qscheme=local_qscheme,
            context=context,
            obs_name=current_name,
        )

    for key, value in list(normalized.items()):
        if isinstance(value, Mapping):
            normalized[key] = _normalize_overrides(
                value,
                inherited_dtype=local_dtype,
                inherited_qscheme=local_qscheme,
                context=f"{context}.{key}",
                current_name=key,
            )

    return normalized


@dataclass
class PTQConfig(BaseConfig):
    """
    One object describes the quantization preferences for a single wrapper
    and its descendants.

    Parameters
    ----------
    default_dtype : DType
        Fallback dtype for every observer that DOES NOT receive an explicit
        override.
    default_observer : Type[ObserverBase], optional
        Observer class to instantiate when the caller (or an override) does
         not provide a `observer` key.
    default_qscheme : Optional[QScheme]
        Fallback quantization scheme for observers that do not receive an
        explicit override.

        When set to `None`, the qscheme is inferred automatically from the
        effective dtype and, for special observer names such as `weight`,
        from the observer role:
            - unsigned activation-like dtype -> `QScheme.PER_TENSOR_ASYMM`
            - unsigned weight dtype          -> `QScheme.PER_CHANNEL_ASYMM`
            - signed dtype                   -> `QScheme.PER_TENSOR_SYMM`

        When explicitly provided, the pair is validated. Incompatible pairs,
        such as unsigned dtype with symmetric qscheme, raise immediately.
    overrides : Mapping[str, Mapping[str, Any]]
        Two-level mapping of scopes → observer-kwargs.

        • SCOPE can be either
            - the attribute name of a child wrapper
              (e.g. "gate_proj" or "up_proj"), or
            - an observer logical name inside this wrapper
              (e.g. "mul", "act_in").

        • "Observer-kwargs" is forwarded verbatim to the observer constructor
          (`dtype`, `qscheme`, `channel_axis`, `observer`, …).
    model_args : Mapping[str, Any]
        Additional model-specific metadata required by certain wrappers.

        This is intended for inputs that are not part of quantization policy
        but are still needed to construct or run a wrapper correctly.

        Typical examples include:
            - vision grid metadata for VLMs
              (e.g. `{"vision": {"grid_thw": (T, H, W)}}`)
            - model-specific shape hints
            - execution metadata required for static export paths

        Unlike `overrides`, `model_args` is not scope-filtered by observer name.
        It is propagated as-is to child configurations.
    strict_wrap : bool
        If ``True``, any module that cannot be wrapped will raise an error.
    attention_mask_fill_value : float
        Value used to fill masked positions in attention masks before softmax.
        This affects softmax suppression strength for masked positions and
        numerical range before quantization/fake-quant. Default: -120.0.

    Example
    -------
    ```python
    from wrapq.observers import PercentileObserver

    cfg = PTQConfig(
        default_dtype   = DType.uint(8),
        default_qscheme  = QScheme.PER_TENSOR_SYMM,        # <- global scheme
        default_observer = PercentileObserver,             # <- global algorithm
        overrides={
            # local override: input observer now MinMax & 4-bit, per-channel asymmetric
            "act_in": {"observer": MinMaxObserver,
                       "dtype":    DType.uint(4),
                       "qscheme":  QScheme.PER_CHANNEL_ASYMM},
        },
        model_args={
            "vision": {
                "grid_thw": (8, 24, 24),
            },
        },
    )
    ```
    """

    default_dtype: DType = DType.uint(8)
    default_observer: Type[ObserverBase] = MinMaxObserver  # type: ignore[type-abstract]
    default_qscheme: Optional[QScheme] = None
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    model_args: Mapping[str, Any] = field(default_factory=dict)
    # If True, any module that cannot be wrapped will raise.
    strict_wrap: bool = True
    # Value used to fill masked positions in attention masks before softmax.
    attention_mask_fill_value: float = -120.0

    def __post_init__(self) -> None:
        """
        Resolve automatic qscheme defaults and validate nested overrides.
        """
        self.default_qscheme = _resolve_qscheme(
            dtype=self.default_dtype,
            qscheme=self.default_qscheme,
            context="PTQConfig.default_qscheme",
        )
        self.normalize_overrides()

    @property
    def name(self) -> str:
        return "ptq"

    def normalize_overrides(self) -> None:
        """
        Normalize and validate the entire override tree in-place.

        This method is useful when callers directly mutate `self.overrides`
        after construction and want to retroactively apply automatic qscheme
        inference and compatibility checks.
        """
        assert self.default_qscheme is not None
        self.overrides = _normalize_overrides(
            self.overrides,
            inherited_dtype=self.default_dtype,
            inherited_qscheme=self.default_qscheme,
            context="PTQConfig.overrides",
        )

    def set_override(
        self,
        path: Iterable[str],
        value: Mapping[str, Any],
    ) -> None:
        """
        Set a nested override and normalize only the affected subtree.

        Parameters
        ----------
        path : Iterable[str]
            Hierarchical path inside `self.overrides`.
            Example: `("model", "layers", "0", "self_attn", "o_proj", "weight")`
        value : Mapping[str, Any]
            Override payload to assign at the target path.

        Notes
        -----
        The inserted subtree is normalized immediately, so callers may provide
        only `dtype` and rely on automatic qscheme inference.
        """
        keys = tuple(path)
        if not keys:
            raise ValueError("Override path must not be empty.")

        root: MutableMapping[str, Any] = dict(self.overrides)
        current: MutableMapping[str, Any] = root
        parent_dtype = self.default_dtype
        parent_qscheme = self.default_qscheme
        context = "PTQConfig.overrides"

        for key in keys[:-1]:
            context = f"{context}.{key}"
            next_value = current.get(key)
            if isinstance(next_value, Mapping):
                child = dict(next_value)
            elif next_value is None:
                child = {}
            else:
                raise ValueError(
                    f"Cannot create nested override under non-mapping node at {context}."
                )

            current[key] = child
            current = child

            local_dtype = current.get("dtype", parent_dtype)
            parent_qscheme = _resolve_qscheme(
                dtype=local_dtype,
                qscheme=current.get("qscheme", parent_qscheme),
                context=context,
                obs_name=key,
            )
            parent_dtype = local_dtype

        assert parent_qscheme is not None
        leaf_key = keys[-1]
        leaf_context = f"{context}.{leaf_key}"
        current[leaf_key] = _normalize_overrides(
            deepcopy(value),
            inherited_dtype=parent_dtype,
            inherited_qscheme=parent_qscheme,
            context=leaf_context,
            current_name=leaf_key,
        )
        self.overrides = root

    def get_kwargs(self, obs_name: str) -> Dict[str, Any]:
        """
        Return user-specified kwargs for *obs_name* inside **this** wrapper.

        NOTE:
        Do NOT inject a dtype/qscheme here. `_make_obs()` resolves precedence:
            1) user override (kw_cfg["dtype" | "qscheme"])
            2) wrapper's default passed to `_make_obs(..., dtype=..., qscheme=...)`
            3) self.default_dtype / `self.default_qscheme`
        """
        return dict(self.overrides.get(obs_name, {}))

    def get_model_arg(self, key: str, default: Any = None) -> Any:
        """
        Return model-specific metadata stored under *key*.

        This is intended for wrapper-level inputs that are not observer
        configuration, such as vision grid information or static shape hints.
        """
        return self.model_args.get(key, default)

    def child(self, scope: str) -> "PTQConfig":
        """
        Produce a *view* for a child wrapper.

        The child inherits:
          • same `default_dtype`
          • same `default_observer`
          • same `default_qscheme`
          • same `model_args`
          • same `attention_mask_fill_value`
          • overrides under `self.overrides.get(scope, {})`

        Other scopes remain invisible to the child.
        """
        sub_overrides = self.overrides.get(scope, {})
        return PTQConfig(
            self.default_dtype,
            self.default_observer,
            default_qscheme=self.default_qscheme,
            overrides=sub_overrides,
            model_args=self.model_args,
            strict_wrap=self.strict_wrap,
            attention_mask_fill_value=self.attention_mask_fill_value,
        )

    def __repr__(self):
        return (
            "PTQConfig("
            f"default_dtype={self.default_dtype}, "
            f"default_observer={self.default_observer}, "
            f"default_qscheme={self.default_qscheme}, "
            f"overrides={dict(self.overrides)}, "
            f"model_args={dict(self.model_args)}, "
            f"strict_wrap={self.strict_wrap})"
        )
