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

"""Once-only ownership traversal and transactional frozen-state admission."""

from __future__ import annotations

import torch
from torch import nn

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import (
    FrozenObserverState,
    ObserverBase,
    ObserverNotReadyError,
)
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

GroupKey = tuple[QuantModuleBase, str]
Groups = dict[GroupKey, tuple[ObserverBase, ...]]
FreezePlan = dict[ObserverBase, FrozenObserverState]
Conditions = dict[nn.Module, set[frozenset[GroupKey]]]
Boundary = tuple[GroupKey, tuple[str, ...]]
RouteContext = tuple[
    nn.Module,
    frozenset[GroupKey],
    frozenset[GroupKey],
    frozenset[Boundary],
    frozenset[nn.Module],
]


def _modules(root: nn.Module) -> list[QuantModuleBase]:
    modules = [m for m in root.modules() if isinstance(m, QuantModuleBase)]
    for module in modules:
        cls = type(module)
        if (
            cls.enable_calibration is not QuantModuleBase.enable_calibration
            or cls.freeze_qparams is not QuantModuleBase.freeze_qparams
        ):
            raise RuntimeError(
                f"{cls.__name__}: migrate lifecycle overrides to local "
                "_prepare_calibration/_validate_freeze hooks."
            )
    return modules


def _inventory(
    root: nn.Module,
    modules: list[QuantModuleBase],
) -> tuple[Groups, set[ObserverBase], list[ObserverBase], Conditions]:
    groups: Groups = {}
    declarations: dict[QuantModuleBase, frozenset[Boundary]] = {}
    for owner in modules:
        declared: set[Boundary] = set()
        for name, declared_paths in owner._conditional_quant_subtrees().items():
            if not isinstance(name, str) or not name or name == "required":
                raise RuntimeError(
                    "Conditional capability names must be nonempty and optional."
                )
            if not isinstance(declared_paths, tuple) or not declared_paths:
                raise RuntimeError(
                    f"Conditional capability {name!r} must declare a nonempty tuple of registered module paths."
                )
            key = (owner, name)
            groups.setdefault(key, ())
            for path in declared_paths:
                if (
                    not isinstance(path, str)
                    or not path
                    or any(not part for part in path.split("."))
                ):
                    raise RuntimeError(
                        "Conditional subtrees require relative registration-path strings; "
                        "migrate module-object declarations."
                    )
                parts = tuple(path.split("."))
                target: nn.Module = owner
                path_ancestors: set[nn.Module] = {owner}
                for part in parts:
                    child = target._modules.get(part)
                    if child is None:
                        raise RuntimeError(
                            f"Conditional capability {name!r}: path {path!r} is not a registered module path."
                        )
                    if child in path_ancestors:
                        raise RuntimeError(
                            "Conditional module paths must not cross registration cycles."
                        )
                    path_ancestors.add(child)
                    target = child
                declared.add((key, parts))
        declarations[owner] = frozenset(declared)

    # Pending path suffixes are anchored to each owner occurrence. A boundary
    # belongs to that registration route, even when intermediate containers or
    # the target module are aliased elsewhere. Raw registrations preserve aliases
    # that nn.Module.children() intentionally deduplicates.
    boundary_keys = set(groups)
    paths: Conditions = {}
    requirement_boundaries: dict[QuantModuleBase, set[GroupKey]] = {}
    contexts: list[RouteContext] = [
        (root, frozenset(), frozenset(), frozenset(), frozenset())
    ]
    visited: set[RouteContext] = set()
    while contexts:
        node, gates, nearest, pending_paths, ancestors = contexts.pop()
        if isinstance(node, QuantModuleBase):
            pending_paths = pending_paths | declarations[node]
        context = (node, gates, nearest, pending_paths, ancestors)
        if context in visited:
            continue
        visited.add(context)
        if isinstance(node, QuantModuleBase):
            paths.setdefault(node, set()).add(gates)
            requirement_boundaries.setdefault(node, set()).update(nearest)
        ancestors = ancestors | {node}
        for child_name, child in node._modules.items():
            if child is None:
                continue
            if child in ancestors:
                if any(remaining[0] == child_name for _, remaining in pending_paths):
                    raise RuntimeError(
                        "Conditional module paths must not cross registration cycles."
                    )
                continue
            crossed = frozenset(
                key for key, remaining in pending_paths if remaining == (child_name,)
            )
            if len(crossed) > 1:
                raise RuntimeError(
                    "A conditional registration edge must have one capability owner; "
                    "use distinct registered aliases for alternative capabilities."
                )
            remaining_paths = frozenset(
                (key, remaining[1:])
                for key, remaining in pending_paths
                if remaining[0] == child_name and len(remaining) > 1
            )
            contexts.append(
                (
                    child,
                    gates | crossed,
                    crossed or nearest,
                    remaining_paths,
                    ancestors,
                )
            )

    conditional = {
        module: alternatives
        for module, alternatives in paths.items()
        if frozenset() not in alternatives
    }
    owners: dict[ObserverBase, QuantModuleBase] = {}
    local_groups: dict[QuantModuleBase, dict[str, tuple[ObserverBase, ...]]] = {}
    for module in modules:
        local_groups[module] = {
            name: tuple(observers)
            for name, observers in module.local_observer_groups().items()
        }
        registered = set()
        pending = list(module.children())
        while pending:
            child = pending.pop()
            if isinstance(child, QuantModuleBase) or child in registered:
                continue
            registered.add(child)
            pending.extend(child.children())
        for name, observers in local_groups[module].items():
            for observer in observers:
                if not isinstance(observer, ObserverBase) or observer not in registered:
                    raise RuntimeError(
                        "Declared quantization sites need registered local observer "
                        "ownership for exact persistence."
                    )
                if observer in owners and owners[observer] is not module:
                    raise RuntimeError(
                        f"Observer {observer.name!r} has conflicting wrapper owners; "
                        "declare borrowed state via _freeze_dependencies instead."
                    )
                owners[observer] = module
            key = (module, name)
            groups[key] = tuple(dict.fromkeys(observers))
    for module in modules:
        local = local_groups[module]
        for name, observers in local.items():
            if (
                name != "required"
                and not observers
                and (module, name) not in boundary_keys
            ):
                raise RuntimeError(
                    "Empty optional capability has no declared quantization sites."
                )
        # Required work belongs to the nearest boundary. Independent optional
        # local groups and nested boundaries must not make their enclosing route
        # unavailable merely because those optional paths were not calibrated.
        for parent_key in requirement_boundaries.get(module, ()):
            groups[parent_key] = tuple(
                dict.fromkeys(groups[parent_key] + tuple(local.get("required", ())))
            )
    required: set[ObserverBase] = set()
    for (module, name), observers in groups.items():
        if name == "required" and module not in conditional:
            required.update(observers)
    all_observers = list(dict.fromkeys(o for group in groups.values() for o in group))
    return groups, required, all_observers, conditional


def _post_load_validate(root, incompatible_keys):
    validate_frozen_model(root)


def enable_calibration(root: nn.Module) -> None:
    if not any(
        hook is _post_load_validate
        for hook in root._load_state_dict_post_hooks.values()
    ):
        root.register_load_state_dict_post_hook(_post_load_validate)
    modules = _modules(root)
    # Cross-source checks belong after the whole root loads, including siblings.
    for descendant in root.modules():
        if descendant is root:
            continue
        for key, hook in list(descendant._load_state_dict_post_hooks.items()):
            if hook is _post_load_validate:
                del descendant._load_state_dict_post_hooks[key]
    _, _, observers, _ = _inventory(root, modules)
    for observer in observers:
        observer.reset()
        observer.enabled = True
    for module in modules:
        module._mode = Mode.CALIB
        module._ready_capabilities = set()
        module._conditional_unavailable = False
    for module in modules:
        module._prepare_calibration()


def _available(
    module: QuantModuleBase, groups: Groups, plan: FreezePlan, conditional: Conditions
) -> set[str]:
    if not _entry_available(module, conditional, groups, plan):
        return set()
    return {
        name
        for (owner, name), observers in groups.items()
        if owner is module and all(o in plan for o in observers)
    }


def _entry_available(
    module: QuantModuleBase, conditional: Conditions, groups: Groups, plan: FreezePlan
) -> bool:
    return module not in conditional or any(
        all(all(o in plan for o in groups[key]) for key in gates)
        for gates in conditional[module]
    )


def freeze_qparams(root: nn.Module) -> None:
    """Derive and validate without mutation, then install small owned snapshots."""
    modules = _modules(root)
    groups, required, observers, conditional = _inventory(root, modules)
    plan: FreezePlan = {}
    for observer in observers:
        try:
            plan[observer] = observer.derive_candidate()
        except ObserverNotReadyError:
            if observer in required:
                raise
    # An optional group is unavailable as a whole; never install partial grids.
    admitted = required | {
        o
        for (owner, _), group in groups.items()
        if all(o in plan for o in group)
        and _entry_available(owner, conditional, groups, plan)
        for o in group
    }
    plan = {o: state for o, state in plan.items() if o in admitted}
    for module in modules:
        for dependency in module._freeze_dependencies():
            if dependency not in plan:
                plan[dependency] = dependency.require_frozen()
        module._validate_freeze(plan, _available(module, groups, plan, conditional))

    # Installation changes only observer buffers and scalar lifecycle state. Keep
    # their references for rollback, including a failing extension's last install.
    observer_backup = {
        o: (
            o.__dict__.copy(),
            {
                name: (
                    value.detach().clone() if isinstance(value, torch.Tensor) else value
                )
                for name, value in o._buffers.items()
            },
        )
        for o in observers
    }
    module_backup = {
        m: (m._mode, m._ready_capabilities, m._conditional_unavailable) for m in modules
    }
    try:
        for observer in observers:
            if observer in plan:
                observer.install_candidate(plan[observer])
            else:
                observer.enabled = False
        for module in modules:
            module._ready_capabilities = _available(module, groups, plan, conditional)
            module._conditional_unavailable = not _entry_available(
                module, conditional, groups, plan
            )
            module._mode = Mode.QUANT
    except Exception:
        for observer, (attrs, buffers) in observer_backup.items():
            observer.__dict__.clear()
            observer.__dict__.update(attrs)
            observer._buffers = buffers
        for module, (mode, capabilities, unavailable) in module_backup.items():
            module._mode, module._ready_capabilities = mode, capabilities
            module._conditional_unavailable = unavailable
        raise


def validate_frozen_model(root: nn.Module) -> None:
    """Validate restored snapshots and cross-source semantics without deriving grids."""
    modules = _modules(root)
    groups, required, observers, conditional = _inventory(root, modules)
    plan: FreezePlan = {}
    for module in modules:
        if getattr(module, "_frozen_schema", None) != 1:
            raise RuntimeError("Legacy WrapQ object; reprepare and recalibrate.")
        if (
            not isinstance(getattr(module, "_mode", None), Mode)
            or not isinstance(getattr(module, "_conditional_unavailable", None), bool)
            or not isinstance(getattr(module, "_ready_capabilities", None), set)
            or not all(isinstance(name, str) for name in module._ready_capabilities)
        ):
            raise RuntimeError(
                "Corrupt WrapQ frozen readiness; reprepare and recalibrate."
            )
    if any(module._mode is Mode.QUANT for module in modules) and any(
        module._mode is not Mode.QUANT for module in modules
    ):
        raise RuntimeError(
            "Inconsistent frozen wrapper modes; reprepare and recalibrate the complete model."
        )
    for observer in observers:
        observer.validate_loaded_state()
        if observer.frozen_state is not None:
            plan[observer] = observer.require_frozen()
    for module in modules:
        module._validate_loaded_state()
        if module._mode is not Mode.QUANT:
            continue
        for observer in module.local_observer_groups().get("required", ()):
            if observer in required and observer not in plan:
                raise RuntimeError(f"Missing frozen site {observer.name}; recalibrate.")
        unavailable = not _entry_available(module, conditional, groups, plan)
        if module._conditional_unavailable != unavailable:
            raise RuntimeError("Corrupt conditional subtree readiness; recalibrate.")
        available = _available(module, groups, plan, conditional)
        if module._ready_capabilities != available:
            raise RuntimeError(
                "Corrupt frozen capability state; reprepare and recalibrate."
            )
        for dependency in module._freeze_dependencies():
            if dependency not in plan:
                plan[dependency] = dependency.require_frozen()
        module._validate_freeze(plan, available)
