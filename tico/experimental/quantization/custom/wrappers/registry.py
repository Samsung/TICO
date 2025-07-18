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

import importlib
from typing import Callable, Dict, Type

import torch.nn as nn

from tico.experimental.quantization.custom.wrappers.base_quant_module import (
    QuantModuleBase,
)

_WRAPPERS: Dict[Type[nn.Module], Type[QuantModuleBase]] = {}
_IMPORT_ONCE = False
_CORE_MODULES = (
    "tico.experimental.quantization.custom.wrappers.nn.quant_linear",
    "tico.experimental.quantization.custom.wrappers.nn.quant_silu",
    # add future core wrappers here
)


def _lazy_init():
    global _IMPORT_ONCE
    if _IMPORT_ONCE:
        return
    for mod in _CORE_MODULES:
        __import__(mod)  # triggers decorators
    _IMPORT_ONCE = True


# ───────────────────────────── decorator for *always*-present classes
def register(
    fp_cls: Type[nn.Module],
) -> Callable[[Type[QuantModuleBase]], Type[QuantModuleBase]]:
    def _decorator(quant_cls: Type[QuantModuleBase]):
        _WRAPPERS[fp_cls] = quant_cls
        return quant_cls

    return _decorator


# ───────────────────────────── conditional decorator
def try_register(path: str) -> Callable[[Type[QuantModuleBase]], Type[QuantModuleBase]]:
    """
    `@try_register("transformers.models.llama.modeling_llama.LlamaMLP")`

    • If import succeeds → behave like `@register`
    • If module/class not found → become a NO-OP
    """

    def _decorator(quant_cls: Type[QuantModuleBase]):
        module_name, _, cls_name = path.rpartition(".")
        try:
            mod = importlib.import_module(module_name)
            fp_cls = getattr(mod, cls_name)
            _WRAPPERS[fp_cls] = quant_cls
        except (ModuleNotFoundError, AttributeError):
            # transformers not installed or class renamed – silently skip
            pass
        return quant_cls

    return _decorator


# ───────────────────────────── lookup
def lookup(fp_cls: Type[nn.Module]) -> Type[QuantModuleBase] | None:
    _lazy_init()
    return _WRAPPERS.get(fp_cls)
