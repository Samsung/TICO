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

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any


@lru_cache(maxsize=1)
def circle_schema() -> Any:
    """Return the generated Circle schema module exposed by circle-schema."""

    package = import_module("circle_schema")
    module = getattr(package, "circle", None)
    if module is not None:
        return module

    # Older development layouts exposed the generated module directly at this
    # location. Keep the fallback so source builds remain usable.
    return import_module("circle_schema.circle")


def schema_table_module(name: str) -> Any:
    """Return a generated schema table module by name."""

    module = circle_schema()
    try:
        return getattr(module, name)
    except AttributeError as error:
        raise RuntimeError(f"Circle schema does not provide table {name!r}.") from error


def object_api_type(name: str) -> type[Any]:
    """Return the Object API class for a generated Circle table."""

    module = circle_schema()
    table_module = getattr(module, name, None)
    candidates = (
        getattr(table_module, f"{name}T", None) if table_module is not None else None,
        getattr(module, f"{name}T", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Circle schema does not provide Object API type {name}T.")


def accessor_api_type(name: str) -> type[Any]:
    """Return the accessor class for a generated Circle table."""

    module = circle_schema()
    table_module = getattr(module, name, None)
    candidates = (
        getattr(table_module, name, None) if table_module is not None else None,
        getattr(module, name, None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, f"GetRootAs{name}"):
            return candidate
    raise RuntimeError(f"Circle schema does not provide accessor type {name}.")


def enum_name(enum_name: str, value: int) -> str:
    """Return a stable symbolic name for a generated Circle enum value."""

    module = circle_schema()
    enum_module = getattr(module, enum_name, None)
    enum_type = (
        getattr(enum_module, enum_name, None) if enum_module is not None else None
    )
    if enum_type is None:
        enum_type = getattr(module, enum_name, None)
    if enum_type is None:
        return str(value)

    for name, candidate in vars(enum_type).items():
        if name.startswith("_") or not isinstance(candidate, int):
            continue
        if candidate == value:
            return name
    return str(value)


def decode_text(value: Any) -> str:
    """Convert a generated schema text field to a Python string."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
