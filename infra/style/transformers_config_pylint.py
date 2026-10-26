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

"""Model Transformers' runtime-generated configuration constructors for Pylint."""

from __future__ import annotations

from astroid import MANAGER, nodes
from astroid.brain.brain_dataclasses import (
    _generate_dataclass_init,
    _get_dataclass_attributes,
)
from astroid.builder import parse
from astroid.exceptions import InferenceError
from pylint.lint import PyLinter

_CONFIG_BASE = "transformers.configuration_utils.PreTrainedConfig"
_REGISTERED = False


def _has_generated_constructor(node: nodes.ClassDef) -> bool:
    """Select library config subclasses whose initializer is generated at runtime."""
    if (
        not node.name.endswith("Config")
        or not node.root().name.startswith("transformers.")
        or "__init__" in node.locals
    ):
        return False
    try:
        return any(base.qname() == _CONFIG_BASE for base in node.ancestors())
    except InferenceError:
        return False


def _infer_config_constructor(node: nodes.ClassDef) -> None:
    """Expose the dataclass fields and backward-compatible extra keyword arguments."""
    # PreTrainedConfig.__init_subclass__ applies dataclass(..., kw_only=True).
    # Reuse the field handling of the pinned Astroid version, including inherited
    # fields, ClassVar, default factories, and dataclasses.field(init=False).
    node.is_dataclass = True
    source = _generate_dataclass_init(
        node, list(_get_dataclass_attributes(node, init=True)), True
    )
    initializer = parse(source)["__init__"]
    initializer.parent = node
    initializer.lineno = None
    initializer.col_offset = None
    arguments = initializer.args
    positional = arguments.args or []
    defaults = arguments.defaults or []
    first_default = len(positional) - len(defaults)
    for index, argument in reversed(list(enumerate(positional[1:], start=1))):
        arguments.kwonlyargs.insert(0, argument)
        arguments.kwonlyargs_annotations.insert(0, arguments.annotations[index])
        arguments.kw_defaults.insert(
            0, defaults[index - first_default] if index >= first_default else None
        )
    arguments.args = positional[:1]
    arguments.annotations = arguments.annotations[:1]
    arguments.defaults = []
    # Transformers wraps generated initializers with wrap_init_to_accept_kwargs;
    # extra configuration keys are passed to __post_init__, rather than rejected.
    arguments.kwarg = "kwargs"
    node.locals["__init__"] = [initializer]


def register(_linter: PyLinter) -> None:
    """Register the inference transform once in each Pylint process."""
    global _REGISTERED
    if not _REGISTERED:
        MANAGER.register_transform(
            nodes.ClassDef, _infer_config_constructor, _has_generated_constructor
        )
        _REGISTERED = True
