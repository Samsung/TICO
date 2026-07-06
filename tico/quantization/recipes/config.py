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

import ast
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


Config = dict[str, Any]


def load_recipe_config(
    path: str | Path, overrides: Sequence[str] | None = None
) -> Config:
    """Load a YAML/JSON recipe config and apply dotted ``--set`` overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recipe config does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        cfg = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to read YAML recipe configs. "
                "Install pyyaml or use a .json config."
            )
        cfg = yaml.safe_load(text) or {}

    if not isinstance(cfg, dict):
        raise TypeError(f"Recipe config must be a mapping. got {type(cfg)}")

    cfg = copy.deepcopy(cfg)
    apply_overrides(cfg, overrides or ())
    return cfg


def save_effective_config(path: str | Path, cfg: Mapping[str, Any]) -> None:
    """Persist the fully resolved recipe config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    else:
        path.write_text(
            yaml.safe_dump(to_plain_data(cfg), sort_keys=False), encoding="utf-8"
        )


def to_plain_data(value: Any) -> Any:
    """Convert values that YAML cannot serialize into stable strings."""
    if isinstance(value, Mapping):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_data(v) for v in value]
    if isinstance(value, tuple):
        return [to_plain_data(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def apply_overrides(cfg: Any, overrides: Iterable[str]) -> None:
    """Apply overrides of the form ``a.b.c=value``.

    List indices are supported, e.g. ``pipeline.0.enabled=false``. Lists whose
    items are mappings with a ``name`` field can also be addressed by name, e.g.
    ``pipeline.gptq.mse=mse``.
    """
    for raw in overrides:
        path, value = parse_override(raw)
        set_by_path(cfg, path, value)


def parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"Override must have KEY=VALUE form. got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key must not be empty. got: {raw}")
    return key.split("."), parse_scalar(value.strip())


def parse_scalar(value: str) -> Any:
    """Parse a CLI scalar/list/dict in a YAML-like way."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", "~"}:
        return None

    if yaml is not None:
        try:
            return yaml.safe_load(value)
        except Exception:
            pass

    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _is_index(part: str) -> bool:
    return part.isdigit()


def _format_location(path: Sequence[str]) -> str:
    """Return a readable dotted path for diagnostics."""
    return ".".join(path) if path else "<root>"


def _available_item_names(items: Sequence[Any]) -> list[str]:
    """Return stringified names from mapping items in a list."""
    names: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if "name" not in item:
            continue
        names.append(str(item["name"]))
    return names


def _find_named_list_item(
    items: list[Any],
    name: str,
    *,
    location: Sequence[str],
) -> tuple[int, Mapping[str, Any]]:
    """Find a unique mapping item whose ``name`` field matches ``name``."""
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for idx, item in enumerate(items):
        if isinstance(item, Mapping) and item.get("name") == name:
            matches.append((idx, item))

    loc = _format_location(location)
    if not matches:
        available = _available_item_names(items)
        suffix = f" Available names: {available}." if available else ""
        raise KeyError(f"No list item named {name!r} at {loc}.{suffix}")

    if len(matches) > 1:
        indices = ", ".join(str(idx) for idx, _ in matches)
        raise ValueError(
            f"Multiple list items named {name!r} at {loc} "
            f"(indices: {indices}). Use a numeric index instead."
        )

    return matches[0]


def _find_mutable_named_list_item(
    items: list[Any],
    name: str,
    *,
    location: Sequence[str],
) -> tuple[int, MutableMapping[str, Any]]:
    """Find a unique mutable mapping item whose ``name`` field matches ``name``."""
    idx, item = _find_named_list_item(items, name, location=location)
    if not isinstance(item, MutableMapping):
        loc = _format_location(location)
        raise TypeError(
            f"List item named {name!r} at {loc} is not mutable. got {type(item)}"
        )
    return idx, item


def get_by_path(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a value from a dotted path.

    Mapping keys, list indices, and named list items are supported. Named list
    items are mappings with a ``name`` field, so ``pipeline.gptq.enabled`` reads
    the ``enabled`` field from the unique pipeline stage named ``gptq``.
    """
    cur: Any = cfg
    parts = path.split(".")
    for i, part in enumerate(parts):
        if isinstance(cur, Mapping):
            if part not in cur:
                return default
            cur = cur[part]
        elif isinstance(cur, list):
            if _is_index(part):
                idx = int(part)
                if idx >= len(cur):
                    return default
                cur = cur[idx]
            else:
                try:
                    _, cur = _find_named_list_item(cur, part, location=parts[:i])
                except (KeyError, ValueError):
                    return default
        else:
            return default
    return cur


def set_by_path(cfg: Any, path: Sequence[str], value: Any) -> None:
    """Set a value at a dotted override path.

    Mapping keys and numeric list indices preserve the original override
    semantics. Existing lists can also be traversed by matching mapping item
    names, e.g. ``pipeline.gptq.mse=mse`` updates the unique list item with
    ``name: gptq``.
    """
    if not path:
        raise ValueError("Override path must not be empty.")

    cur = cfg
    for i, part in enumerate(path[:-1]):
        nxt_part = path[i + 1]

        if isinstance(cur, list):
            if _is_index(part):
                idx = int(part)
                while len(cur) <= idx:
                    cur.append({} if not _is_index(nxt_part) else [])
                if cur[idx] is None:
                    cur[idx] = {} if not _is_index(nxt_part) else []
                cur = cur[idx]
            else:
                _, cur = _find_mutable_named_list_item(cur, part, location=path[:i])
            continue

        if not isinstance(cur, MutableMapping):
            raise TypeError(
                f"Cannot descend into non-container at {'.'.join(path[:i])}"
            )

        nxt = cur.get(part)
        if not isinstance(nxt, (MutableMapping, list)):
            nxt = [] if _is_index(nxt_part) else {}
            cur[part] = nxt
        cur = nxt

    leaf = path[-1]
    if isinstance(cur, list):
        if not _is_index(leaf):
            raise TypeError(
                f"Cannot assign list item {leaf!r} directly at {'.'.join(path)}. "
                "Use a numeric index or set a named item field such as "
                f"{'.'.join((*path, '<field>'))}."
            )
        idx = int(leaf)
        while len(cur) <= idx:
            cur.append(None)
        cur[idx] = value
    elif isinstance(cur, MutableMapping):
        cur[leaf] = value
    else:
        raise TypeError(
            f"Cannot set value under non-container at {'.'.join(path[:-1])}"
        )


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Config:
    """Return ``base`` recursively merged with ``overlay``."""
    out: Config = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def ensure_mapping(value: Any, *, name: str) -> MutableMapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MutableMapping):
        raise TypeError(f"{name} must be a mapping. got {type(value)}")
    return value
