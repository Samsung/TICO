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
from typing import Dict, Optional, Type, TypeVar

from tico.quantization.config.base import BaseConfig
from tico.quantization.quantizer import BaseQuantizer

TQ = TypeVar("TQ", bound=BaseQuantizer)

# Mapping: Config type -> Quantizer type
_REGISTRY: Dict[Type[BaseConfig], Type[BaseQuantizer]] = {}


def register_quantizer(config_cls: Type[BaseConfig]):
    """
    Decorator to register a quantizer for a given config class.
    Usage:
        @register_quantizer(GPTQConfig)
        class GPTQQuantizer(BaseQuantizer): ...
    """

    def wrapper(quantizer_cls: Type[TQ]) -> Type[TQ]:
        _REGISTRY[config_cls] = quantizer_cls
        return quantizer_cls

    return wrapper


def _lookup(cfg: BaseConfig) -> Optional[Type[BaseQuantizer]]:
    """Return a quantizer class only if the exact config type is registered."""
    return _REGISTRY.get(type(cfg))


def get_quantizer(cfg: BaseConfig) -> BaseQuantizer:
    """Factory to return a quantizer instance for the given config."""
    qcls = _lookup(cfg)
    if qcls is not None:
        return qcls(cfg)

    # Lazy import by naming convention
    name = getattr(cfg, "name", None)
    if name:
        if name == "ptq":
            importlib.import_module(f"tico.quantization.wrapq.quantizer")
        else:
            try:
                importlib.import_module(f"tico.quantization.algorithm.{name}.quantizer")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to import quantizer module for config name='{name}': {e}"
                )

    qcls = _lookup(cfg)
    if qcls is not None:
        return qcls(cfg)

    raise RuntimeError(
        f"No quantizer registered for config type {type(cfg).__name__} "
        f"(name='{getattr(cfg,'name',None)}')."
    )
