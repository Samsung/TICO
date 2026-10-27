# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
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
    boundaries: dict[nn.Module, list[GroupKey]] = {}
    for owner in modules:
        for name, subtrees in owner._conditional_quant_subtrees().items():
            key = (owner, name)
            groups.setdefault(key, ())
            for subtree in subtrees:
                boundaries.setdefault(subtree, []).append(key)

    # A tied module may be reached through a required path or either of several
    # optional paths. Preserve those alternatives instead of making every alias
    # conditional just because one reference sits under an optional branch.
    paths: Conditions = {}
    visited: set[tuple[nn.Module, frozenset[GroupKey], frozenset[QuantModuleBase]]] = (
        set()
    )

    contexts: list[
        tuple[nn.Module, frozenset[GroupKey], frozenset[QuantModuleBase]]
    ] = [(root, frozenset(), frozenset())]
    while contexts:
        node, gates, ancestors = contexts.pop()
        gates = gates | frozenset(
            key for key in boundaries.get(node, ()) if key[0] in ancestors
        )
        context = (node, gates, ancestors)
        if context in visited:
            continue
        visited.add(context)
        if isinstance(node, QuantModuleBase):
            paths.setdefault(node, set()).add(gates)
            ancestors = ancestors | {node}
        contexts.extend(
            (child, gates, ancestors)
            for child in node.children()
            if child not in ancestors
        )

    conditional = {
        module: alternatives
        for module, alternatives in paths.items()
        if frozenset() not in alternatives
    }
    owners: dict[ObserverBase, QuantModuleBase] = {}
    for module in modules:
        registered = set()
        pending = list(module.children())
        while pending:
            child = pending.pop()
            if isinstance(child, QuantModuleBase) or child in registered:
                continue
            registered.add(child)
            pending.extend(child.children())
        for name, observers in module.local_observer_groups().items():
            observers = tuple(observers)
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
    for (module, _), owned in list(groups.items()):
        for parent_key in {
            key for gates in conditional.get(module, ()) for key in gates
        }:
            groups[parent_key] = tuple(dict.fromkeys(groups[parent_key] + owned))
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


def _available(module: QuantModuleBase, groups: Groups, plan: FreezePlan) -> set[str]:
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
        module._validate_freeze(plan, _available(module, groups, plan))

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
            module._ready_capabilities = _available(module, groups, plan)
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
        available = _available(module, groups, plan)
        if module._ready_capabilities != available:
            raise RuntimeError(
                "Corrupt frozen capability state; reprepare and recalibrate."
            )
        for dependency in module._freeze_dependencies():
            if dependency not in plan:
                plan[dependency] = dependency.require_frozen()
        module._validate_freeze(plan, available)
