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

"""Helpers for optional evaluation dependencies.

Evaluation backends such as ``lm_eval``, ``datasets`` and ``transformers`` are
not required for every quantization recipe.  Import them only from the evaluation
function that actually needs them, and use these helpers to produce a clear
runtime error when an optional package is missing.
"""

import importlib
from types import ModuleType
from typing import Any


def require_module(
    module_name: str,
    *,
    feature: str,
    install_hint: str,
) -> ModuleType:
    """Import an optional module or raise a clear RuntimeError.

    Args:
        module_name: Fully-qualified module name to import.
        feature: Human-readable feature name that requires the module.
        install_hint: Installation hint shown to the user.

    Returns:
        The imported module.

    Raises:
        RuntimeError: If the module cannot be imported.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"{module_name!r} is required for {feature}. "
            f"Install the optional evaluation dependency with: {install_hint}"
        ) from exc


def require_attr(
    module_name: str,
    attr_name: str,
    *,
    feature: str,
    install_hint: str,
) -> Any:
    """Import an optional module and return one attribute from it.

    Args:
        module_name: Fully-qualified module name to import.
        attr_name: Attribute expected in the imported module.
        feature: Human-readable feature name that requires the attribute.
        install_hint: Installation hint shown to the user.

    Returns:
        The requested attribute.

    Raises:
        RuntimeError: If the module cannot be imported or lacks the attribute.
    """
    module = require_module(
        module_name,
        feature=feature,
        install_hint=install_hint,
    )
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"{module_name!r} was imported, but it does not provide "
            f"{attr_name!r}, which is required for {feature}. "
            f"Check the installed package version or reinstall with: {install_hint}"
        ) from exc
