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

from typing import Any, Dict, Tuple

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from tico.utils import logging
from tico.utils.installed_packages import is_transformers_installed

__all__ = [
    "register_dynamic_cache",
    "register_static_cache",
    "register_dynamic_layer",
    "register_static_layer",
    "register_encoder_decoder_cache",
]

##################################################################################
# [Dynamic/Static Cache Registration]
# Reference: https://github.com/Samsung/TICO/issues/417
#
# {transformer-version}: {status (implementation detail)}
# 4.51.0: ✅ Pre-registered (list-based)
# 4.51.1: ✅ Pre-registered (list-based)
# 4.52.0: ✅ Pre-registered (list-based)
# 4.52.1: ✅ Pre-registered (list-based)
# 4.53.0: ✅ Pre-registered (list-based)
# 4.54.0: ✅ Pre-registered (layers-based)
# 4.54.1: ✅ Pre-registered (layers-based)
# 4.55.0: ✅ Pre-registered (layers-based)
# 4.55.1: ✅ Pre-registered (layers-based)
# 4.56.0: ❌ Not registered (layers-based)
# 4.56.1: ❌ Not registered (layers-based)
# 4.57.0: ❌ Not registered (layers-based)
# 4.57.1: ❌ Not registered (layers-based)
# 4.57.2: ❌ Not registered (layers-based)
# 4.57.3: ❌ Not registered (layers-based)
# 4.57.4: ❌ Not registered (layers-based)
# 4.57.5: ❌ Not registered (layers-based)
# 4.57.6: ❌ Not registered (layers-based)
# 5.0.0: ❌ Not registered (layers-based)
#
##################################################################################

##################################################################################
#
# All _flatten_* / _unflatten_* helpers are defined at module scope (not inside
# functions) so that torch pytree serialization can locate them by name.
#
# Convention for every cache type:
#   _flatten_<type>           -> (children, aux_data)   [main pytree API]
#   _unflatten_<type>         -> reconstructed object
#   _flatten_with_keys_<type> -> keyed children list     [for pytree.register_pytree_node]
#   _flatten_<type>_for_fx    -> flat list                [for fx_pytree.register_pytree_flatten_spec]\
#
##################################################################################


# ---------------------------------------------------------------------------
# StaticCache
# ---------------------------------------------------------------------------


def _flatten_static_cache(cache) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    children = (cache.layers,)
    aux_data = {
        "layer_class_to_replicate": getattr(cache, "layer_class_to_replicate", None),
        "offloading": getattr(cache, "offloading", False),
    }
    return children, aux_data


def _unflatten_static_cache(children: Tuple[Any, ...], aux_data: Dict[str, Any]):
    from transformers.cache_utils import StaticCache

    instance = StaticCache.__new__(StaticCache)
    (instance.layers,) = children
    for key, value in aux_data.items():
        setattr(instance, key, value)
    return instance


def _flatten_with_keys_static_cache(cache):
    children, aux_data = _flatten_static_cache(cache)
    return [(pytree.GetAttrKey("layers"), children[0])], aux_data


def _flatten_static_cache_for_fx(cache, spec):
    children, _ = _flatten_static_cache(cache)
    return list(children)


def register_static_cache():
    # StaticCache uses a layers-based structure only when StaticLayer is available
    # (transformers >= ~4.57).  On older versions _flatten_static_cache with non-existent
    # '.layers' attribute must have been pre-registered to pytree, so we skip registration.
    try:
        from transformers.cache_utils import StaticCache, StaticLayer  # noqa: F401
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.debug(
            "StaticCache / StaticLayer not available in this transformers version; "
            "skipping StaticCache pytree registration."
        )
        return

    try:
        pytree.register_pytree_node(
            StaticCache,
            _flatten_static_cache,
            _unflatten_static_cache,
            serialized_type_name=f"{StaticCache.__module__}.{StaticCache.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_static_cache,
        )
        fx_pytree.register_pytree_flatten_spec(
            StaticCache, _flatten_static_cache_for_fx
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"StaticCache is already registered as pytree flattenable. {e}")


# ---------------------------------------------------------------------------
# StaticLayer
# ---------------------------------------------------------------------------


def _flatten_static_layer(layer) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Split a StaticLayer into (tensor children, static metadata)."""
    if not getattr(layer, "is_initialized", False):
        raise ValueError(
            f"{layer} cannot be flattened. StaticLayer must be initialized "
            "with tensors of a specific shape before use with torch.export."
        )

    # 1. convert tracing tensors
    children = (layer.keys, layer.values)

    # 2. extract aux with compatibility
    aux_data: Dict[str, Any] = {
        "max_cache_len": getattr(layer, "max_cache_len", 0),
        "is_initialized": getattr(layer, "is_initialized", False),
        "dtype": getattr(layer, "dtype", None),
        "device": getattr(layer, "device", None),
        "max_batch_size": getattr(
            layer, "max_batch_size", getattr(layer, "batch_size", 0)
        ),
        "num_heads": getattr(layer, "num_heads", 0),
        "head_dim": getattr(layer, "head_dim", 0),
    }
    return children, aux_data


def _unflatten_static_layer(children: Tuple[Any, ...], aux_data: Dict[str, Any]):
    """Reconstruct a StaticLayer from flattened data."""
    from transformers.cache_utils import StaticLayer

    keys, values = children

    # avoid __init__ signature mismatch error, depending on torch version
    obj = object.__new__(StaticLayer)

    for key, value in aux_data.items():
        if value is not None:
            setattr(obj, key, value)

    # For compatibility
    if not hasattr(obj, "batch_size") and "max_batch_size" in aux_data:
        setattr(obj, "batch_size", aux_data["max_batch_size"])

    obj.keys = keys
    obj.values = values

    return obj


def _flatten_with_keys_static_layer(layer):
    children, aux_data = _flatten_static_layer(layer)

    # In casse of GetAttrKey does not exist in old torch version
    KeyClass = getattr(pytree, "GetAttrKey", getattr(pytree, "MappingKey", None))

    if KeyClass is None:
        # In case of very old torch version
        return [("keys", children[0]), ("values", children[1])], aux_data

    return [
        (KeyClass("keys"), children[0]),
        (KeyClass("values"), children[1]),
    ], aux_data


def _flatten_static_layer_for_fx(layer, spec):
    children, _ = _flatten_static_layer(layer)
    return list(children)


def register_static_layer():
    try:
        from transformers.cache_utils import StaticLayer
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.debug(
            "StaticLayer not available in this transformers version; "
            "skipping StaticLayer pytree registration."
        )
        return

    try:
        pytree.register_pytree_node(
            StaticLayer,
            _flatten_static_layer,
            _unflatten_static_layer,
            serialized_type_name=f"{StaticLayer.__module__}.{StaticLayer.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_static_layer,
        )
        fx_pytree.register_pytree_flatten_spec(
            StaticLayer, _flatten_static_layer_for_fx
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"StaticLayer is already registered as pytree flattenable. {e}")


# ---------------------------------------------------------------------------
# DynamicLayer
# ---------------------------------------------------------------------------


def _flatten_dynamic_layer(layer) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if not layer.is_initialized:
        raise ValueError(
            f"{layer} cannot be flattened. DynamicLayer must be initialized "
            "with tensors of a specific shape before use with torch.export."
        )
    children = (layer.keys, layer.values)
    aux_data: Dict[str, Any] = {
        "is_initialized": layer.is_initialized,
        "dtype": layer.keys.dtype,
        "device": layer.keys.device,
    }
    return children, aux_data


def _unflatten_dynamic_layer(children: Tuple[Any, ...], aux_data: Dict[str, Any]):
    from transformers.cache_utils import DynamicLayer

    keys, values = children
    obj = DynamicLayer()
    obj.keys = keys
    obj.values = values
    obj.is_initialized = aux_data["is_initialized"]
    obj.dtype = aux_data["dtype"]
    obj.device = aux_data["device"]
    return obj


def _flatten_with_keys_dynamic_layer(layer):
    children, aux_data = _flatten_dynamic_layer(layer)
    return [
        (pytree.GetAttrKey("keys"), children[0]),
        (pytree.GetAttrKey("values"), children[1]),
    ], aux_data


def _flatten_dynamic_layer_for_fx(layer, spec):
    children, _ = _flatten_dynamic_layer(layer)
    return list(children)


def register_dynamic_layer():
    try:
        from transformers.cache_utils import DynamicLayer
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.debug(
            "DynamicLayer not available in this transformers version; "
            "skipping DynamicLayer pytree registration."
        )
        return

    try:
        pytree.register_pytree_node(
            DynamicLayer,
            _flatten_dynamic_layer,
            _unflatten_dynamic_layer,
            serialized_type_name=f"{DynamicLayer.__module__}.{DynamicLayer.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_dynamic_layer,
        )
        fx_pytree.register_pytree_flatten_spec(
            DynamicLayer, _flatten_dynamic_layer_for_fx
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"DynamicLayer is already registered as pytree flattenable. {e}")


# ---------------------------------------------------------------------------
# DynamicCache
# ---------------------------------------------------------------------------


def _flatten_dynamic_cache(cache) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    children = (cache.layers,)
    aux_data = {
        "layer_class_to_replicate": getattr(cache, "layer_class_to_replicate", None),
        "offloading": getattr(cache, "offloading", False),
    }
    return children, aux_data


def _unflatten_dynamic_cache(children: Tuple[Any, ...], aux_data: Dict[str, Any]):
    from transformers.cache_utils import DynamicCache

    instance = DynamicCache.__new__(DynamicCache)
    (instance.layers,) = children
    for key, value in aux_data.items():
        setattr(instance, key, value)
    return instance


def _flatten_with_keys_dynamic_cache(cache):
    children, aux_data = _flatten_dynamic_cache(cache)
    return [(pytree.GetAttrKey("layers"), children[0])], aux_data


def _flatten_dynamic_cache_for_fx(cache, spec):
    children, _ = _flatten_dynamic_cache(cache)
    return list(children)


# Legacy flatten/unflatten for transformers versions that do not have
# DynamicLayer (e.g. <= 4.52.x), which store tensors directly in
# key_cache / value_cache instead of the Layer-based cache.layers structure.
def _flatten_dynamic_cache_legacy(cache) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    children = (cache.key_cache, cache.value_cache)
    aux_data: Dict[str, Any] = {}
    return children, aux_data


def _unflatten_dynamic_cache_legacy(
    children: Tuple[Any, ...], aux_data: Dict[str, Any]
):
    from transformers.cache_utils import DynamicCache

    key_cache, value_cache = children
    cache = DynamicCache()
    cache.key_cache = key_cache
    cache.value_cache = value_cache
    return cache


def _flatten_with_keys_dynamic_cache_legacy(cache):
    children, aux_data = _flatten_dynamic_cache_legacy(cache)
    return [
        (pytree.GetAttrKey("key_cache"), children[0]),
        (pytree.GetAttrKey("value_cache"), children[1]),
    ], aux_data


def _flatten_dynamic_cache_for_fx_legacy(cache, spec):
    children, _ = _flatten_dynamic_cache_legacy(cache)
    return list(children)


def register_dynamic_cache():
    """Register DynamicCache as a pytree node.

    Two layouts exist across transformers versions:

    * **Layer-based** (newer, requires ``DynamicLayer`` to be importable):
      ``cache.layers`` is a list of ``DynamicLayer`` objects; each layer
      holds ``keys`` and ``values`` tensors.  Both ``DynamicCache`` and
      ``DynamicLayer`` must be registered as pytree nodes for
      ``torch.export`` to trace through the cache.

    * **Legacy** (older, e.g. transformers <= 4.52.x):
      The cache stores tensors directly in ``cache.key_cache`` and
      ``cache.value_cache`` lists.  No ``DynamicLayer`` class exists.

    The correct layout is detected by checking whether ``DynamicLayer`` can
    be imported rather than by comparing version strings, which have proven
    unreliable across patch releases.
    """
    if not is_transformers_installed:  # type: ignore[truthy-function]
        raise ImportError("transformers package is not installed")

    from transformers.cache_utils import DynamicCache

    # Feature-detect the Layer-based layout.
    try:
        from transformers.cache_utils import DynamicLayer as _DL  # noqa: F401

        _has_dynamic_layer = True
    except ImportError:
        _has_dynamic_layer = False

    if not _has_dynamic_layer:
        # Legacy layout: flatten key_cache / value_cache directly.
        try:
            pytree.register_pytree_node(
                DynamicCache,
                _flatten_dynamic_cache_legacy,
                _unflatten_dynamic_cache_legacy,
                serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
                flatten_with_keys_fn=_flatten_with_keys_dynamic_cache_legacy,
            )
            fx_pytree.register_pytree_flatten_spec(
                DynamicCache, _flatten_dynamic_cache_for_fx_legacy
            )
        except ValueError as e:
            logger = logging.getLogger(__name__)
            logger.debug(
                f"DynamicCache is already registered as pytree flattenable. {e}"
            )
        return

    try:
        pytree.register_pytree_node(
            DynamicCache,
            _flatten_dynamic_cache,
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
        )
        fx_pytree.register_pytree_flatten_spec(
            DynamicCache, _flatten_dynamic_cache_for_fx
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"DynamicCache is already registered as pytree flattenable. {e}")


# ---------------------------------------------------------------------------
# EncoderDecoderCache
# ---------------------------------------------------------------------------


def _flatten_encoder_decoder_cache(cache) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    children = (cache.self_attention_cache, cache.cross_attention_cache)
    aux_data: Dict[str, Any] = {}
    return children, aux_data


def _unflatten_encoder_decoder_cache(
    children: Tuple[Any, ...], aux_data: Dict[str, Any]
):
    from transformers.cache_utils import EncoderDecoderCache

    self_cache, cross_cache = children
    return EncoderDecoderCache(self_cache, cross_cache)


def _flatten_with_keys_encoder_decoder_cache(cache):
    children, aux_data = _flatten_encoder_decoder_cache(cache)
    return [
        (pytree.GetAttrKey("self_attention_cache"), children[0]),
        (pytree.GetAttrKey("cross_attention_cache"), children[1]),
    ], aux_data


def _flatten_encoder_decoder_cache_for_fx(cache, spec):
    children, _ = _flatten_encoder_decoder_cache(cache)
    return list(children)


def register_encoder_decoder_cache():
    from transformers.cache_utils import EncoderDecoderCache

    try:
        pytree.register_pytree_node(
            EncoderDecoderCache,
            _flatten_encoder_decoder_cache,
            _unflatten_encoder_decoder_cache,
            serialized_type_name=(
                f"{EncoderDecoderCache.__module__}.{EncoderDecoderCache.__name__}"
            ),
            flatten_with_keys_fn=_flatten_with_keys_encoder_decoder_cache,
        )
        fx_pytree.register_pytree_flatten_spec(
            EncoderDecoderCache, _flatten_encoder_decoder_cache_for_fx
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.debug(
            f"EncoderDecoderCache is already registered as pytree flattenable. {e}"
        )
