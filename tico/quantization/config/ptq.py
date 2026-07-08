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
from typing import (
    Any,
    Dict,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Union,
)

from tico.quantization.config.base import BaseConfig
from tico.quantization.config.specs import affine, QuantSpec
from tico.quantization.config.utils import auto_qscheme_for, dtype_is_unsigned
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme


ExportMode = Literal["prefill", "decode", "append_prefill"]
OverridePath = Union[str, Iterable[str]]
OverrideValue = Union[QuantSpec, Mapping[str, Any]]
_PARAMETER_OBSERVER_NAMES = {"weight", "bias"}


def _resolve_qscheme(
    *,
    dtype: DType,
    qscheme: Optional[QScheme],
    context: str,
    obs_name: Optional[str] = None,
) -> QScheme:
    """Resolve and validate a dtype/qscheme pair."""
    resolved_qscheme = qscheme or auto_qscheme_for(dtype, obs_name)

    if dtype_is_unsigned(dtype) and resolved_qscheme.is_symmetric():
        raise ValueError(
            f"Invalid quantization config at {context}: unsigned dtype "
            f"{dtype!r} cannot be paired with symmetric qscheme "
            f"{resolved_qscheme!r}."
        )

    return resolved_qscheme


def _parse_path(path: OverridePath) -> tuple[str, ...]:
    """Normalize an override path into a tuple of path components."""
    if isinstance(path, str):
        keys = tuple(part for part in path.split(".") if part)
    else:
        keys = tuple(path)

    if not keys:
        raise ValueError("Override path must not be empty.")

    return keys


def _deep_merge(left: MutableMapping[str, Any], right: Mapping[str, Any]) -> None:
    """Merge nested override mappings into ``left`` in-place."""
    for key, value in right.items():
        current = left.get(key)
        if isinstance(current, MutableMapping) and isinstance(value, Mapping):
            _deep_merge(current, value)
        else:
            left[key] = deepcopy(value)


def _set_nested_override(
    root: MutableMapping[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    """Set an override value at a nested path in-place."""
    current = root
    for key in path[:-1]:
        next_value = current.get(key)
        if next_value is None:
            child: MutableMapping[str, Any] = {}
            current[key] = child
        elif isinstance(next_value, MutableMapping):
            child = next_value
        elif isinstance(next_value, Mapping):
            child = dict(next_value)
            current[key] = child
        else:
            raise ValueError(
                "Cannot create nested override under non-mapping node "
                f"at {'.'.join(path)}."
            )
        current = child

    current[path[-1]] = deepcopy(value)


def _expand_path_overrides(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand dot-path override keys into a nested override tree."""
    expanded: Dict[str, Any] = {}

    for key, value in mapping.items():
        if isinstance(key, str) and "." in key:
            _set_nested_override(expanded, _parse_path(key), value)
            continue

        existing = expanded.get(key)
        if isinstance(value, Mapping) and isinstance(existing, MutableMapping):
            _deep_merge(existing, value)
        else:
            expanded[key] = deepcopy(value)

    return expanded


def _as_override_mapping(
    value: Any,
    *,
    context: str,
    current_name: Optional[str],
) -> Dict[str, Any]:
    """Convert a QuantSpec or mapping override into a mutable dictionary."""
    if isinstance(value, QuantSpec):
        return value.to_kwargs(
            obs_name=current_name,
            context=context,
            mark_replace=True,
        )
    if isinstance(value, Mapping):
        return dict(value)

    raise TypeError(
        f"Invalid override value at {context}: expected QuantSpec or mapping, "
        f"got {type(value).__name__}."
    )


def _normalize_overrides(
    mapping: Mapping[str, Any],
    *,
    inherited_dtype: Optional[DType] = None,
    inherited_qscheme: Optional[QScheme] = None,
    context: str,
    current_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Recursively normalize and validate nested override mappings."""
    normalized = _as_override_mapping(
        mapping,
        context=context,
        current_name=current_name,
    )

    local_dtype = normalized.get("dtype", inherited_dtype)
    local_qscheme = normalized.get("qscheme", inherited_qscheme)

    if "dtype" in normalized:
        normalized["qscheme"] = _resolve_qscheme(
            dtype=normalized["dtype"],
            qscheme=normalized.get("qscheme"),
            context=context,
            obs_name=current_name,
        )
        local_dtype = normalized["dtype"]
        local_qscheme = normalized["qscheme"]
    elif "qscheme" in normalized and local_dtype is not None:
        local_qscheme = _resolve_qscheme(
            dtype=local_dtype,
            qscheme=normalized["qscheme"],
            context=context,
            obs_name=current_name,
        )

    for key, value in list(normalized.items()):
        if isinstance(value, QuantSpec):
            normalized[key] = value.to_kwargs(
                obs_name=key,
                context=f"{context}.{key}",
                mark_replace=True,
            )
        elif isinstance(value, Mapping):
            normalized[key] = _normalize_overrides(
                value,
                inherited_dtype=local_dtype,
                inherited_qscheme=local_qscheme,
                context=f"{context}.{key}",
                current_name=key,
            )

    return normalized


def _default_activation_spec() -> QuantSpec:
    """Return the default PTQ activation policy."""
    return affine(DType.uint(8))


def _default_weight_spec() -> QuantSpec:
    """Return the default PTQ parameter policy."""
    return affine(DType.uint(8))


@dataclass
class PTQConfig(BaseConfig):
    """Describe quantization preferences for one wrapper and descendants.

    Parameters
    ----------
    activation : QuantSpec
        Default policy for activation-like observers such as `act_in` and
        `act_out`.
    weight : QuantSpec
        Default policy for parameter-like observers such as `weight` and
        `bias`.
    overrides : Mapping[str, OverrideValue]
        Nested override tree or dot-path override mapping. Leaves may be
        QuantSpec objects or raw observer constructor kwargs.
    model_args : Mapping[str, Any]
        Additional model-specific metadata required by certain wrappers.
    strict_wrap : bool
        If `True`, unsupported modules raise during wrapping.
    attention_mask_fill_value : float
        Value used to fill masked positions before attention softmax.
    """

    activation: QuantSpec = field(default_factory=_default_activation_spec)
    weight: QuantSpec = field(default_factory=_default_weight_spec)
    overrides: Mapping[str, OverrideValue] = field(default_factory=dict)
    model_args: Mapping[str, Any] = field(default_factory=dict)
    strict_wrap: bool = True
    attention_mask_fill_value: float = -120.0

    def __post_init__(self) -> None:
        """Normalize path-based overrides and validate override leaves."""
        self.normalize_overrides()

    @property
    def name(self) -> str:
        return "ptq"

    def normalize_overrides(self) -> None:
        """Normalize and validate the entire override tree in-place."""
        expanded = _expand_path_overrides(self.overrides)
        self.overrides = _normalize_overrides(
            expanded,
            context="PTQConfig.overrides",
        )

    def set_override(
        self,
        path: OverridePath,
        value: OverrideValue,
    ) -> None:
        """Set a nested override using either a dot path or an iterable path.

        Parameters
        ----------
        path : OverridePath
            Dot path such as "model.layers.0.self_attn.q_proj.act_out"
            or an iterable of path components.
        value : OverrideValue
            QuantSpec or observer constructor kwargs assigned at the path.
        """
        root: MutableMapping[str, Any] = deepcopy(dict(self.overrides))
        _set_nested_override(root, _parse_path(path), value)
        self.overrides = root
        self.normalize_overrides()

    def get_kwargs(self, obs_name: str) -> Dict[str, Any]:
        """Return user-specified kwargs for an observer in this wrapper."""
        value = self.overrides.get(obs_name, {})
        if isinstance(value, QuantSpec):
            return value.to_kwargs(
                obs_name=obs_name,
                context=f"PTQConfig.{obs_name}",
                mark_replace=True,
            )
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(
            f"Invalid override for observer {obs_name!r}: "
            f"expected QuantSpec or mapping, got {type(value).__name__}."
        )

    def get_role_kwargs(self, obs_name: str) -> Dict[str, Any]:
        """Return default kwargs for the observer role represented by a name."""
        spec = self.weight if obs_name in _PARAMETER_OBSERVER_NAMES else self.activation
        return spec.to_kwargs(
            obs_name=obs_name,
            context=f"PTQConfig.{obs_name}",
            infer_qscheme=False,
        )

    def get_model_arg(self, key: str, default: Any = None) -> Any:
        """Return model-specific metadata stored under `key`."""
        return self.model_args.get(key, default)

    def child(self, scope: str) -> "PTQConfig":
        """Produce a child view scoped to overrides under `scope`."""
        sub_overrides_raw = self.overrides.get(scope, {})
        if isinstance(sub_overrides_raw, QuantSpec):
            sub_overrides = sub_overrides_raw.to_kwargs(
                obs_name=scope,
                context=f"PTQConfig.overrides.{scope}",
                mark_replace=True,
            )
        elif isinstance(sub_overrides_raw, Mapping):
            sub_overrides = sub_overrides_raw  # type: ignore[assignment]
        else:
            sub_overrides = {}

        return PTQConfig(
            activation=self.activation,
            weight=self.weight,
            overrides=sub_overrides,
            model_args=self.model_args,
            strict_wrap=self.strict_wrap,
            attention_mask_fill_value=self.attention_mask_fill_value,
        )

    def __repr__(self) -> str:
        return (
            "PTQConfig("
            f"activation={self.activation}, "
            f"weight={self.weight}, "
            f"overrides={dict(self.overrides)}, "
            f"model_args={dict(self.model_args)}, "
            f"strict_wrap={self.strict_wrap})"
        )
