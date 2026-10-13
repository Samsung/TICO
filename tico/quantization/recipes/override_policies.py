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

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tico.quantization.config.ptq import OverrideValue, PTQConfig
from tico.quantization.config.specs import QuantSpec
from tico.quantization.recipes.utils import quant_spec_from_config


# ``op_type: linear`` alias shared by adapters whose decoder layers follow the
# Hugging Face ``self_attn.{q,k,v,o}_proj`` / ``mlp.{gate,up,down}_proj``
# naming. Adapters extend or replace it in their own
# ``build_quant_target_context`` hook; the selector engine never consults it.
COMMON_LINEAR_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_COMMON_OBSERVER_ALIASES = {
    "activation": ("act_in", "act_out"),
    "parameter": ("weight",),
    "input_activation": ("act_in",),
    "output_activation": ("act_out",),
}
_VALID_COMPONENTS = {"text", "vision", "all"}
_SPEC_LIKE_KEYS = {"kind", "type", "dtype", "qscheme", "elem_format", "observer"}


@dataclass(frozen=True)
class ComponentQuantTargetInfo:
    """Describe quantization selector targets for one logical model component."""

    name: str
    layer_path_prefixes: tuple[tuple[str, ...], ...]
    num_layers: int
    op_aliases: Mapping[str, tuple[str, ...]]
    observer_aliases: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: _COMMON_OBSERVER_ALIASES
    )


@dataclass(frozen=True)
class QuantTargetResolverContext:
    """Collect all component metadata required to resolve override policies."""

    components: Mapping[str, ComponentQuantTargetInfo]


@dataclass(frozen=True)
class ResolvedOverride:
    """Represent one selector policy after it has been resolved to an exact path."""

    path: tuple[str, ...]
    value: OverrideValue
    specificity: int
    order: int
    rule_name: str


SpecRegistry = Mapping[str, QuantSpec]


def apply_ptq_override_policies(
    qcfg: PTQConfig,
    stage_cfg: Mapping[str, Any],
    *,
    family: str,
    model: Any,
    applied_paths: set[tuple[str, ...]] | None = None,
    adapter: Any | None = None,
) -> PTQConfig:
    """Apply recipe-level PTQ override policies to an existing PTQConfig.

    Args:
        qcfg: PTQConfig produced by the model adapter.
        stage_cfg: PTQ stage YAML payload.
        family: Registered recipe adapter family.
        model: Loaded model instance used to infer component metadata.
        applied_paths: Optional set that receives the exact override path of
            every selector or raw override applied to ``qcfg``. Paths are added
            as each override is applied; existing entries are never removed,
            so pass a fresh empty set to observe one call in isolation. When
            ``None``, no paths are collected.
        adapter: Optional selected adapter whose ``build_quant_target_context``
            hook is consulted before the registered family default. See
            :func:`build_quant_target_resolver_context`.

    Returns:
        The same PTQConfig instance with selector and raw overrides applied.
    """
    if not _has_override_payload(stage_cfg):
        return qcfg

    context = build_quant_target_resolver_context(
        family=family, model=model, adapter=adapter
    )
    return apply_ptq_override_policies_to_config(
        qcfg,
        stage_cfg,
        context,
        applied_paths=applied_paths,
    )


def apply_ptq_override_policies_to_config(
    qcfg: PTQConfig,
    stage_cfg: Mapping[str, Any],
    context: QuantTargetResolverContext,
    *,
    applied_paths: set[tuple[str, ...]] | None = None,
) -> PTQConfig:
    """Apply PTQ override policies using an already-built resolver context.

    ``applied_paths`` follows the same contract as in
    :func:`apply_ptq_override_policies`: applied override paths are added to
    the given set in application order and existing entries are preserved.
    """
    specs = parse_named_specs(stage_cfg.get("specs", {}))
    resolved = compile_override_policies(
        stage_cfg.get("override_policies", []),
        specs=specs,
        context=context,
    )
    for override in resolved:
        qcfg.set_override(override.path, override.value)
        if applied_paths is not None:
            applied_paths.add(override.path)

    for path, value in compile_raw_overrides(
        stage_cfg.get("raw_overrides", {}),
        specs=specs,
    ):
        qcfg.set_override(path, value)
        if applied_paths is not None:
            applied_paths.add(path)

    return qcfg


def parse_named_specs(raw_specs: Any) -> dict[str, QuantSpec]:
    """Parse the optional PTQ ``specs`` mapping into named QuantSpec objects."""
    if raw_specs is None:
        return {}
    if not isinstance(raw_specs, Mapping):
        raise TypeError("ptq.specs must be a mapping of names to quant specs.")

    specs: dict[str, QuantSpec] = {}
    for name, value in raw_specs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("ptq.specs keys must be non-empty strings.")
        spec = quant_spec_from_config(value)
        if spec is None:
            raise ValueError(f"ptq.specs.{name} must not resolve to None.")
        specs[name] = spec
    return specs


def compile_override_policies(
    policies: Any,
    *,
    specs: SpecRegistry,
    context: QuantTargetResolverContext,
) -> list[ResolvedOverride]:
    """Compile selector-based override policies into exact PTQ override paths."""
    if policies is None:
        return []
    if not isinstance(policies, Sequence) or isinstance(policies, (str, bytes)):
        raise TypeError("ptq.override_policies must be a list of policy mappings.")

    seen_names: set[str] = set()
    resolved: list[ResolvedOverride] = []
    for order, policy in enumerate(policies):
        if not isinstance(policy, Mapping):
            raise TypeError(f"ptq.override_policies[{order}] must be a policy mapping.")
        name = str(policy.get("name", "")).strip()
        if not name:
            raise ValueError(
                f"ptq.override_policies[{order}] requires a non-empty name."
            )
        if name in seen_names:
            raise ValueError(f"Duplicate PTQ override policy name: {name!r}.")
        seen_names.add(name)

        if not bool(policy.get("enabled", True)):
            continue

        allow_empty = bool(policy.get("allow_empty", False))
        if "target" not in policy:
            raise ValueError(f"PTQ override policy {name!r} requires a target.")
        if "spec" not in policy:
            raise ValueError(f"PTQ override policy {name!r} requires a spec.")

        spec = _resolve_spec(policy["spec"], specs, context=f"policy {name!r}.spec")
        target = policy["target"]
        if not isinstance(target, Mapping):
            raise TypeError(f"PTQ override policy {name!r}.target must be a mapping.")

        entries = _compile_target(
            name=name,
            order=order,
            target=target,
            spec=spec,
            allow_empty=allow_empty,
            context=context,
        )
        if not entries and not allow_empty:
            raise ValueError(f"PTQ override policy {name!r} did not match any target.")
        resolved.extend(entries)

    return sorted(resolved, key=lambda item: (item.specificity, item.order))


def compile_raw_overrides(
    raw_overrides: Any,
    *,
    specs: SpecRegistry,
) -> list[tuple[tuple[str, ...], OverrideValue]]:
    """Compile raw exact-path overrides from a recipe PTQ stage."""
    if raw_overrides is None:
        return []
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("ptq.raw_overrides must be a mapping of paths to specs.")

    compiled: list[tuple[tuple[str, ...], OverrideValue]] = []
    for path, value in _iter_raw_override_leaves(raw_overrides, specs=specs):
        compiled.append((path, value))
    return compiled


def build_quant_target_resolver_context(
    *,
    family: str,
    model: Any,
    adapter: Any | None = None,
) -> QuantTargetResolverContext:
    """Build selector resolver metadata for a recipe model family.

    The structure declaration is owned by the model adapters: this helper only
    selects which adapter is asked. The explicitly selected ``adapter`` (the
    one ``resolve_adapter`` picked for the recipe) is consulted first through
    its ``build_quant_target_context(model)`` hook. When no adapter is given,
    the adapter lacks the hook, or the hook returns ``None``, the adapter
    registered under ``family`` is asked instead. ``family`` is stripped and
    lower-cased and must be the canonical family key, so registry aliases such
    as ``qwen3-vl`` are rejected exactly as before. Hook errors and invalid
    return values propagate; a valid empty context is returned as-is.

    Raises:
        ValueError: If neither the selected adapter nor the family default
            declares selector metadata, or if ``adapter`` serves a different
            family than ``family``.
    """
    # Imported lazily so that importing the selector engine never loads every
    # built-in adapter module.
    from tico.quantization.recipes.adapters import resolve_quant_target_context

    context = resolve_quant_target_context(model, family=family, adapter=adapter)
    if context is None:
        raise ValueError(
            f"PTQ override policies are not supported for model family {family!r}."
        )
    return context


def _has_override_payload(stage_cfg: Mapping[str, Any]) -> bool:
    return bool(stage_cfg.get("override_policies")) or bool(
        stage_cfg.get("raw_overrides")
    )


def _compile_target(
    *,
    name: str,
    order: int,
    target: Mapping[str, Any],
    spec: QuantSpec,
    allow_empty: bool,
    context: QuantTargetResolverContext,
) -> list[ResolvedOverride]:
    _validate_selector_exclusivity(
        target,
        fields=("module", "modules", "op_type"),
        context=f"PTQ override policy {name!r}.target",
    )
    _validate_selector_exclusivity(
        target,
        fields=("observers", "observer_role"),
        context=f"PTQ override policy {name!r}.target",
    )

    component_names = _resolve_component_names(
        target.get("component", "all"),
        context=context,
        allow_empty=allow_empty,
    )
    if not component_names:
        return []

    resolved: list[ResolvedOverride] = []
    for component_name in component_names:
        component = context.components[component_name]
        layers = _resolve_layers(
            target.get("layers", "all"),
            num_layers=component.num_layers,
            context=f"PTQ override policy {name!r}.target.layers",
        )
        modules = _resolve_modules(
            target,
            component=component,
            context=f"PTQ override policy {name!r}.target",
        )
        observers = _resolve_observers(
            target,
            component=component,
            context=f"PTQ override policy {name!r}.target",
        )
        specificity = _specificity(target, layers=layers)

        for layer_idx in layers:
            for prefix in component.layer_path_prefixes:
                layer_prefix = prefix + (str(layer_idx),)
                for module in modules:
                    module_path = _parse_dot_path(
                        module,
                        context=f"PTQ override policy {name!r}.target.module",
                    )
                    for observer in observers:
                        resolved.append(
                            ResolvedOverride(
                                path=layer_prefix + module_path + (observer,),
                                value=spec,
                                specificity=specificity,
                                order=order,
                                rule_name=name,
                            )
                        )

    return resolved


def _validate_selector_exclusivity(
    target: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
    context: str,
) -> None:
    present = [field for field in fields if target.get(field) is not None]
    if len(present) > 1:
        joined = ", ".join(present)
        raise ValueError(f"{context} must use only one of {fields}; got {joined}.")


def _resolve_component_names(
    component: Any,
    *,
    context: QuantTargetResolverContext,
    allow_empty: bool,
) -> tuple[str, ...]:
    component_name = str(component).strip().lower()
    if component_name not in _VALID_COMPONENTS:
        raise ValueError(
            "target.component must be one of {'text', 'vision', 'all'}, "
            f"got {component!r}."
        )

    if component_name == "all":
        return tuple(context.components.keys())

    if component_name in context.components:
        return (component_name,)
    if allow_empty:
        return ()
    available = sorted(context.components)
    raise ValueError(
        f"target.component={component_name!r} is not available for this model. "
        f"Available components: {available}."
    )


def _resolve_layers(value: Any, *, num_layers: int, context: str) -> tuple[int, ...]:
    if num_layers < 0:
        raise ValueError(f"{context}: num_layers must be non-negative.")

    if value is None or value == "all":
        layers = tuple(range(num_layers))
    elif isinstance(value, int):
        layers = (_validate_layer_index(value, num_layers=num_layers, context=context),)
    elif isinstance(value, str):
        raw = value.strip()
        if raw == "all":
            layers = tuple(range(num_layers))
        elif ":" in raw:
            layers = _resolve_layer_slice(raw, num_layers=num_layers, context=context)
        else:
            layers = (
                _validate_layer_index(int(raw), num_layers=num_layers, context=context),
            )
    elif isinstance(value, Sequence):
        layers = tuple(
            _validate_layer_index(int(item), num_layers=num_layers, context=context)
            for item in value
        )
    else:
        raise TypeError(
            f"{context} must be 'all', an integer, an integer list, or a slice string."
        )

    if not layers:
        raise ValueError(f"{context} did not select any layers.")
    return tuple(dict.fromkeys(layers))


def _resolve_layer_slice(
    value: str, *, num_layers: int, context: str
) -> tuple[int, ...]:
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"{context} has invalid slice syntax: {value!r}.")

    start = _parse_optional_int(parts[0], default=0, context=context)
    end = _parse_optional_int(parts[1], default=num_layers, context=context)
    step = (
        _parse_optional_int(parts[2], default=1, context=context)
        if len(parts) == 3
        else 1
    )

    if step <= 0:
        raise ValueError(f"{context} slice step must be positive.")
    if start < 0 or end < 0:
        raise ValueError(f"{context} slice indices must be non-negative.")
    if start > num_layers or end > num_layers:
        raise ValueError(
            f"{context} slice {value!r} is out of range for {num_layers} layers."
        )
    if start > end:
        raise ValueError(f"{context} slice start must be <= end.")
    return tuple(range(start, end, step))


def _parse_optional_int(value: str, *, default: int, context: str) -> int:
    raw = value.strip()
    return default if raw == "" else int(raw)


def _validate_layer_index(index: int, *, num_layers: int, context: str) -> int:
    if index < 0 or index >= num_layers:
        raise ValueError(
            f"{context} index {index} is out of range for {num_layers} layers."
        )
    return index


def _resolve_modules(
    target: Mapping[str, Any],
    *,
    component: ComponentQuantTargetInfo,
    context: str,
) -> tuple[str, ...]:
    module = target.get("module")
    modules = target.get("modules")
    op_type = target.get("op_type")

    if module is not None:
        return (str(module),)
    if modules is not None:
        if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)):
            raise TypeError(f"{context}.modules must be a list of module paths.")
        parsed = tuple(str(item) for item in modules)
        if not parsed:
            raise ValueError(f"{context}.modules must not be empty.")
        return parsed
    if op_type is not None:
        key = str(op_type).strip().lower()
        if key not in component.op_aliases:
            raise ValueError(
                f"Unsupported op_type {op_type!r} for component {component.name!r}. "
                f"Available aliases: {sorted(component.op_aliases)}."
            )
        return component.op_aliases[key]

    raise ValueError(f"{context} requires one of module, modules, or op_type.")


def _resolve_observers(
    target: Mapping[str, Any],
    *,
    component: ComponentQuantTargetInfo,
    context: str,
) -> tuple[str, ...]:
    observers = target.get("observers")
    observer_role = target.get("observer_role")

    if observers is not None:
        if not isinstance(observers, Sequence) or isinstance(observers, (str, bytes)):
            raise TypeError(f"{context}.observers must be a list of observer names.")
        parsed = tuple(str(item) for item in observers)
        if not parsed:
            raise ValueError(f"{context}.observers must not be empty.")
        return parsed
    if observer_role is not None:
        key = str(observer_role).strip().lower()
        if key not in component.observer_aliases:
            raise ValueError(
                f"Unsupported observer_role {observer_role!r} for component "
                f"{component.name!r}. Available aliases: "
                f"{sorted(component.observer_aliases)}."
            )
        return component.observer_aliases[key]

    raise ValueError(f"{context} requires one of observers or observer_role.")


def _specificity(target: Mapping[str, Any], *, layers: tuple[int, ...]) -> int:
    score = 0
    component = str(target.get("component", "all")).strip().lower()
    if component != "all":
        score += 5

    raw_layers = target.get("layers", "all")
    if raw_layers != "all" and raw_layers is not None:
        score += 20 if len(layers) == 1 else 10

    if target.get("module") is not None:
        score += 20
    elif target.get("modules") is not None:
        score += 10
    elif target.get("op_type") is not None:
        score += 0

    if target.get("observers") is not None:
        score += 20
    elif target.get("observer_role") is not None:
        score += 0

    return score


def _iter_raw_override_leaves(
    mapping: Mapping[Any, Any],
    *,
    specs: SpecRegistry,
    prefix: tuple[str, ...] = (),
):
    for raw_key, value in mapping.items():
        path = prefix + _parse_raw_key(raw_key)
        if _is_spec_leaf(value, specs):
            yield path, _resolve_spec(value, specs, context="ptq.raw_overrides")
            continue
        if isinstance(value, Mapping):
            yield from _iter_raw_override_leaves(value, specs=specs, prefix=path)
            continue
        yield path, _resolve_spec(value, specs, context="ptq.raw_overrides")


def _is_spec_leaf(value: Any, specs: SpecRegistry) -> bool:
    if isinstance(value, QuantSpec):
        return True
    if isinstance(value, str) and value in specs:
        return True
    if not isinstance(value, Mapping):
        return True
    return any(key in value for key in _SPEC_LIKE_KEYS)


def _resolve_spec(value: Any, specs: SpecRegistry, *, context: str) -> QuantSpec:
    if isinstance(value, str) and value in specs:
        return specs[value]
    spec = quant_spec_from_config(value)
    if spec is None:
        raise ValueError(f"{context} must resolve to a quantization spec.")
    return spec


def _parse_raw_key(key: Any) -> tuple[str, ...]:
    return _parse_dot_path(str(key), context="ptq.raw_overrides path")


def _parse_dot_path(value: str, *, context: str) -> tuple[str, ...]:
    path = tuple(part for part in value.split(".") if part)
    if not path:
        raise ValueError(f"{context} must not be empty.")
    return path


__all__ = [
    "COMMON_LINEAR_MODULES",
    "ComponentQuantTargetInfo",
    "QuantTargetResolverContext",
    "ResolvedOverride",
    "apply_ptq_override_policies",
    "apply_ptq_override_policies_to_config",
    "build_quant_target_resolver_context",
    "compile_override_policies",
    "compile_raw_overrides",
    "parse_named_specs",
]
