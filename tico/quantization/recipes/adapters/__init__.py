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

from typing import Any, Mapping

from tico.quantization.config.base import BaseConfig
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.gemma4_assistant import Gemma4AssistantAdapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter

_ADAPTERS: dict[str, ModelAdapter] = {
    "llama": LlamaAdapter(),
    "qwen3_vl": Qwen3VLAdapter(),
    "qwen3-vl": Qwen3VLAdapter(),
    "gemma4": Gemma4Adapter(),
    "gemma4_assistant": Gemma4AssistantAdapter(),
    "gemma4-assistant": Gemma4AssistantAdapter(),
}


def _normalize_key(name: str) -> str:
    """Return the canonical registry key for an adapter name."""
    if not isinstance(name, str):
        raise TypeError(f"Adapter name must be a string. got {type(name)}")
    key = name.strip().lower()
    if not key:
        raise ValueError("Adapter name must not be empty.")
    return key


def get_adapter(family: str) -> ModelAdapter:
    """Return the adapter registered under ``family``."""
    key = _normalize_key(family)
    if key not in _ADAPTERS:
        raise KeyError(f"Unknown model family: {family}. available={sorted(_ADAPTERS)}")
    return _ADAPTERS[key]


def available_adapters() -> list[str]:
    """Return the sorted registry keys of all registered adapters."""
    return sorted(_ADAPTERS)


def register_adapter(name: str, adapter: ModelAdapter) -> ModelAdapter:
    """Register an out-of-tree adapter under an explicit key.

    The key is normalized like ``get_adapter`` lookups. Registering the same
    adapter object again under the same key is a no-op, which keeps repeated
    extension activation safe. Registering a different adapter under an
    occupied key raises ``ValueError`` so that built-in adapters are never
    replaced implicitly. Choose a distinct key and select it with
    ``model.adapter`` in the recipe config instead.
    """
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(
            "Adapter must be a ModelAdapter instance. " f"got {type(adapter).__name__}"
        )
    family = getattr(adapter, "family", None)
    if not isinstance(family, str) or not family.strip():
        raise ValueError(
            f"Adapter {type(adapter).__name__} must define a non-empty family."
        )

    key = _normalize_key(name)
    existing = _ADAPTERS.get(key)
    if existing is None:
        _ADAPTERS[key] = adapter
        return adapter
    if existing is adapter:
        return adapter
    raise ValueError(
        f"Adapter key {key!r} is already registered to "
        f"{type(existing).__name__}; refusing to replace it with "
        f"{type(adapter).__name__}. Register under a distinct key and select "
        "it with model.adapter."
    )


def resolve_adapter(cfg: Mapping[str, Any]) -> ModelAdapter:
    """Select the adapter for a recipe config.

    ``model.family`` remains the model-family identifier consumed by dataset,
    stage, and export helpers. ``model.adapter`` optionally selects a
    differently registered adapter for that family, for example an
    out-of-tree variant registered with ``register_adapter``. The selected
    adapter must declare the same ``family`` as ``model.family`` so that
    family-keyed behavior stays consistent.
    """
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, Mapping):
        raise TypeError("model must be a mapping.")
    if "family" not in model_cfg:
        raise KeyError("Recipe config requires model.family.")
    family = model_cfg["family"]
    adapter_name = model_cfg.get("adapter")
    if adapter_name is None:
        return get_adapter(family)

    adapter = get_adapter(adapter_name)
    family_key = _normalize_key(family)
    if family_key in _ADAPTERS:
        # Resolve aliases such as ``qwen3-vl`` through the registered adapter.
        family_key = _normalize_key(_ADAPTERS[family_key].family)
    if _normalize_key(adapter.family) != family_key:
        raise ValueError(
            f"model.adapter {adapter_name!r} serves family {adapter.family!r}, "
            f"but model.family is {family!r}. Keep model.family equal to the "
            "adapter family; use model.adapter only to pick an adapter variant."
        )
    return adapter


def _declared_gptq_config_class(adapter: Any) -> type[BaseConfig] | None:
    """Return the config class ``adapter`` selects through its GPTQ hook.

    Adapters without the hook (duck-typed objects predating it) and adapters
    whose hook returns ``None`` make no selection. Hook errors propagate, and a
    return value that is not a ``BaseConfig`` subclass is rejected instead of
    being replaced by the generic config.
    """
    hook = getattr(adapter, "get_gptq_config_class", None)
    if hook is None:
        return None
    selected = hook()
    if selected is None:
        return None
    if not (isinstance(selected, type) and issubclass(selected, BaseConfig)):
        raise TypeError(
            f"{type(adapter).__name__}.get_gptq_config_class() must return a "
            f"BaseConfig subclass or None. got {selected!r}"
        )
    return selected


def resolve_gptq_config_class(adapter: Any) -> type[BaseConfig] | None:
    """Select the default-variant GPTQ config class for ``adapter``.

    The adapter's own ``get_gptq_config_class()`` selection wins. Without one,
    the adapter registered under ``adapter.family`` (the family default, looked
    up like ``get_adapter``) is asked, so an out-of-tree adapter serving a
    built-in family keeps that family's config without implementing the hook.
    The family adapter is skipped when it is the selected adapter itself, so a
    hook is never consulted twice. Returns ``None`` when neither makes a
    selection; the GPTQ stage then uses the generic ``GPTQConfig``.
    """
    selected = _declared_gptq_config_class(adapter)
    if selected is not None:
        return selected
    family_adapter = _ADAPTERS.get(_normalize_key(adapter.family))
    if family_adapter is None or family_adapter is adapter:
        return None
    return _declared_gptq_config_class(family_adapter)
