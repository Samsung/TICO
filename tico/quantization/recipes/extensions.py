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

"""Explicit, config-driven loading of out-of-tree recipe extensions.

A recipe config may list extension entry points under the top-level
``extensions`` key. Each entry is a string of the form ``"package.module"`` or
``"package.module:callable"``. The module is imported and, when given, the
callable is invoked without arguments. Extensions typically register
additional adapters, quantizers, or wrappers through the public registration
APIs before the recipe entrypoint resolves ``model.adapter``.

Loading extensions executes trusted Python code from the current environment.
It is an explicit opt-in per config and is not a sandbox. Nothing is loaded
when the key is absent, and a previous import in another process never counts
as activation: every process that consumes the config loads the listed
extensions again.
"""

import importlib
from collections.abc import Sequence
from typing import Any, Mapping

EXTENSIONS_KEY = "extensions"


def parse_extension_entry(entry: Any) -> tuple[str, str | None]:
    """Split ``"module"`` or ``"module:callable"`` into its components."""
    if not isinstance(entry, str):
        raise TypeError(
            f"{EXTENSIONS_KEY} entries must be strings of the form "
            f"'package.module' or 'package.module:callable'. got {entry!r}"
        )
    text = entry.strip()
    if not text:
        raise ValueError(f"{EXTENSIONS_KEY} entries must not be empty.")
    if text.count(":") > 1:
        raise ValueError(
            f"Invalid {EXTENSIONS_KEY} entry {entry!r}: expected at most one ':'."
        )
    module_name, _, attr = text.partition(":")
    module_name = module_name.strip()
    attr = attr.strip()
    if not module_name:
        raise ValueError(f"Invalid {EXTENSIONS_KEY} entry {entry!r}: missing module.")
    if ":" in text and not attr:
        raise ValueError(
            f"Invalid {EXTENSIONS_KEY} entry {entry!r}: missing callable name."
        )
    return module_name, (attr or None)


def load_extension(entry: str) -> Any:
    """Import one extension entry and invoke its callable when present.

    Returns the callable's result, or the imported module when the entry
    names no callable. Import and attribute failures are reported as
    ``RuntimeError`` naming the offending entry; exceptions raised by the
    callable itself propagate unchanged so extension-specific errors stay
    visible.
    """
    module_name, attr = parse_extension_entry(entry)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import {EXTENSIONS_KEY} entry {entry!r}: {exc}. "
            "Install the distribution that provides it into this environment."
        ) from exc
    if attr is None:
        return module
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise RuntimeError(
            f"{EXTENSIONS_KEY} entry {entry!r} names {attr!r}, which does not "
            f"exist in module {module_name!r}."
        ) from exc
    if not callable(target):
        raise TypeError(
            f"{EXTENSIONS_KEY} entry {entry!r} must name a callable. "
            f"got {type(target).__name__}"
        )
    return target()


def load_recipe_extensions(cfg: Mapping[str, Any]) -> list[str]:
    """Load every extension listed under ``cfg["extensions"]`` in order.

    Returns the list of loaded entries. Call this before adapter resolution,
    model loading, and checkpoint loading so that registrations performed by
    the extensions are visible to all of them.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("Recipe config must be a mapping.")
    entries = cfg.get(EXTENSIONS_KEY)
    if entries is None:
        return []
    if isinstance(entries, str) or not isinstance(entries, Sequence):
        raise TypeError(
            f"{EXTENSIONS_KEY} must be a list of 'package.module[:callable]' "
            f"strings. got {type(entries).__name__}"
        )
    loaded: list[str] = []
    for entry in entries:
        load_extension(entry)
        loaded.append(str(entry).strip())
    return loaded
