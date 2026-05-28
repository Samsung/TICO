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

"""Registry for wrapper smoke cases."""

from collections import OrderedDict
from typing import Iterable

from tico.quantization.recipes.debug.wrapper_smoke.case import WrapperSmokeCase

_REGISTRY: "OrderedDict[str, WrapperSmokeCase]" = OrderedDict()


def register_case(case: WrapperSmokeCase) -> WrapperSmokeCase:
    """Register a case instance by name."""
    if not case.name:
        raise ValueError("Wrapper smoke case name must be non-empty.")
    if case.name in _REGISTRY:
        raise ValueError(f"Duplicate wrapper smoke case: {case.name}")
    _REGISTRY[case.name] = case
    return case


def get_case(name: str) -> WrapperSmokeCase:
    """Return a registered smoke case by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(_REGISTRY)
        raise KeyError(
            f"Unknown wrapper smoke case '{name}'. Known cases: {known}"
        ) from exc


def list_cases(tags: Iterable[str] | None = None) -> list[WrapperSmokeCase]:
    """Return registered cases, optionally filtered by tags."""
    cases = list(_REGISTRY.values())
    if tags:
        tag_set = set(tags)
        cases = [case for case in cases if tag_set.intersection(case.tags)]
    return cases


def case_names(tags: Iterable[str] | None = None) -> list[str]:
    """Return registered case names, optionally filtered by tags."""
    return [case.name for case in list_cases(tags=tags)]


def _register_builtin_cases() -> None:
    """Register all built-in cases exactly once."""
    if _REGISTRY:
        return
    from .cases.llama import LLAMA_CASES
    from .cases.nn import NN_CASES
    from .cases.qwen3_vl import QWEN3_VL_CASES

    for case in (*NN_CASES, *LLAMA_CASES, *QWEN3_VL_CASES):
        register_case(case)


_register_builtin_cases()
