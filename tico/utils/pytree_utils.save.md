import threading

import torch
from packaging.version import Version

from tico.utils import logging
from tico.utils.installed_packages import is_transformers_installed

import torch
from packaging.version import Version
from transformers.cache_utils import StaticCache, StaticLayer, DynamicCache, DynamicLayer, EncoderDecoderCache

__all__ = [
    "register_dynamic_cache", 
    "register_static_cache",
    # "register_dynamic_layer",
    "register_static_layer",
    "register_encoder_decoder_cache",

    # "register_cache_utils_to_pytree" # Covers most of torch cache-utils
    ]

# def register_encoder_decoder_cache():
#     PyTreeRegistryHelper().register(EncoderDecoderCache)

# def register_dynamic_cache():
#     PyTreeRegistryHelper().register(DynamicCache)

# def register_static_cache():
#     PyTreeRegistryHelper().register(StaticCache)

# def register_static_layer():
#     PyTreeRegistryHelper().register(StaticLayer)

# def register_dynamic_layer():
#     PyTreeRegistryHelper().register(DynamicLayer)


# class PyTreeRegistryHelper:
#     """
#     Thread-safe singleton helper class for registering custom PyTree nodes.

#     Thread Safety:
#     - Uses a class-level threading.Lock() to ensure thread-safe singleton instantiation
#     - Uses the same lock to protect the registration process from concurrent calls
#     """

#     _instance = None  # Class variable to hold the singleton instance
#     _lock = threading.Lock()  # Class-level lock for thread-safe operations

#     def __init__(self):
#         """Private constructor to prevent direct instantiation"""
#         pass

#     def __new__(cls, *args, **kwargs):
#         """
#         Thread-safe singleton instance creation using double-checked locking pattern.

#         Returns:
#             PyTreeRegistryHelper: The singleton instance of this class
#         """
#         if not cls._instance:
#             with cls._lock:  # Acquire lock for thread-safe instantiation
#                 if not cls._instance:  # Double-check after acquiring lock
#                     cls._instance = super().__new__(cls)
#         return cls._instance

#     def register(self, cache_cls):
#         """
#         Registers torch cache utility classes as a PyTree node for torch.export compatibility.

#         Raises:
#             ImportError: If transformers package is not installed
#         """
#         with self._lock:  # Acquire lock for thread-safe registration
#             if not is_transformers_installed:
#                 raise ImportError("transformers package is not installed")

#             import transformers
#             if Version(
#                 "4.50.0"
#             ) < Version(transformers.__version__) < Version("4.56.0"):
#                 logger = logging.getLogger(__name__)
#                 logger.warn("{} is be already registered as pytree-flattenable in transformers version 4.50.0 - 4.56.0. (Your transformers version: {transformers.__version__})")

#             try:
#                 torch.utils._pytree.register_pytree_node(
#                     cache_cls,
#                     _flatten_static_cache,
#                     _unflatten_static_cache,
#                     serialized_type_name=f"{cache_cls.__module__}.{cache_cls.__name__}",
#                     flatten_with_keys_fn=_flatten_with_keys_static_cache,
#                 )
#                 torch.fx._pytree.register_pytree_flatten_spec(
#                     cache_cls, _flatten_static_cache_for_fx
#                 )
#             except ValueError as e:
#                 logger = logging.getLogger(__name__)
#                 logger.warning(f"{cache_cls} is already registered as pytree flattenable. {e}")


##################################################################################
# These _flatten_*/_unflatten_* function must be located **outside** - on module scope, not inside function,
# to be registered in pytree clearly.
##################################################################################

def _flatten_static_cache(cache):
    children = (cache.layers,)
    aux_data = {
        "layer_class_to_replicate": getattr(cache, "layer_class_to_replicate", None),
        "offloading": getattr(cache, "offloading", False),
    }
    return children, aux_data

def _unflatten_static_cache(children, aux_data):
    instance = StaticCache.__new__(StaticCache)
    layers, = children
    instance.layers = layers
    
    for key, value in aux_data.items():
        setattr(instance, key, value)
        
    return instance

def _flatten_with_keys_static_cache(cache: StaticCache):
    return torch.utils._pytree._dict_flatten_with_keys(cache.__dict__)

def _flatten_static_cache_for_fx(cache, spec):
    return torch.fx._pytree._dict_flatten_spec(cache.__dict__, spec)

def register_static_cache():
    try:
        torch.utils._pytree.register_pytree_node(
            StaticCache,
            _flatten_static_cache,
            _unflatten_static_cache,
            serialized_type_name=f"{StaticCache.__module__}.{StaticCache.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_static_cache,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            StaticCache, _flatten_static_cache_for_fx
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"StaticCache is already registered as pytree flattenable. {e}")

# def _flatten_static_layer(cache: StaticLayer):
#     nodes = {
#         "keys": cache.keys,
#         "values": cache.values,
#     }
#     return torch.utils._pytree._dict_flatten(nodes)

# def _unflatten_static_layer(values, context: torch.utils._pytree.Context):
#     data = torch.utils._pytree._dict_unflatten(values, context)
    
#     instance = StaticLayer.__new__(StaticLayer)
#     for k, v in data.items():
#         setattr(instance, k, v)
        
#     return instance
from typing import Tuple, Any, Dict
def _flatten_static_layer(cache) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        객체를 (변경 가능한 텐서들, 정적인 메타데이터)로 분리합니다.
        """
        # 1. Children: 모델 최적화나 자동 미분 시 추적해야 할 텐서 데이터
        children = (cache.keys, cache.values)
        
        # 2. Aux Data: 객체를 재구성할 때 필요한 정적 정보 (해시 가능해야 함)
        aux_data = {
            "max_cache_len": cache.max_cache_len,
            "is_initialized": cache.is_initialized,
        }
        # 초기화된 경우라면 메타데이터(dtype, device 등)를 추가로 저장할 수 있습니다.
        if cache.is_initialized:
            aux_data.update({
                "dtype": cache.keys.dtype,
                "device": cache.keys.device,
                "max_batch_size": cache.max_batch_size,
                "num_heads": cache.num_heads,
                "k_head_dim": cache.k_head_dim,
                "v_head_dim": cache.v_head_dim,
            })
            
        return children, aux_data

def _unflatten_static_layer(children: Tuple[Any, ...], aux_data: Dict[str, Any]) -> "StaticLayer":
        """
        flatten된 데이터로부터 새로운 객체를 복구합니다.
        """
        keys, values = children
        # 1. 새 인스턴스 생성
        obj = StaticLayer(max_cache_len=aux_data["max_cache_len"])
        
        # 2. 상태 복구
        obj.is_initialized = aux_data["is_initialized"]
        obj.keys = keys
        obj.values = values
        
        # 3. 초기화되었던 상태라면 나머지 속성들도 복구
        if obj.is_initialized:
            obj.dtype = aux_data["dtype"]
            obj.device = aux_data["device"]
            obj.max_batch_size = aux_data["max_batch_size"]
            obj.num_heads = aux_data["num_heads"]
            obj.k_head_dim = aux_data["k_head_dim"]
            obj.v_head_dim = aux_data["v_head_dim"]
            
        return obj

def _flatten_with_keys_static_layer(cache: StaticLayer):
    return torch.utils._pytree._dict_flatten_with_keys(cache.__dict__)

def _flatten_static_cache_layer(cache, spec):
    return torch.fx._pytree._dict_flatten_spec(cache.__dict__, spec)

def register_static_layer():
    try:
        torch.utils._pytree.register_pytree_node(
            StaticLayer,
            _flatten_static_layer,
            _unflatten_static_layer,
            serialized_type_name=f"{StaticLayer.__module__}.{StaticLayer.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_static_layer,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            StaticLayer, _flatten_static_cache_layer
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"StaticLayer is already registered as pytree flattenable. {e}")

def _flatten_dynamic_cache(cache: DynamicCache):
    return torch.utils._pytree._dict_flatten(cache.__dict__)

def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
    data = torch.utils._pytree._dict_unflatten(values, context)
    
    instance = DynamicCache.__new__(DynamicCache)
    
    instance.__dict__.update(data)
    return instance

def _flatten_with_keys_dynamic_cache(cache: DynamicCache):
    return torch.utils._pytree._dict_flatten_with_keys(cache.__dict__)

def _flatten_dynamic_cache_for_fx(cache, spec):
    return torch.fx._pytree._dict_flatten_spec(cache.__dict__, spec)

def register_dynamic_cache():
    try:
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            _flatten_dynamic_cache,
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache, _flatten_dynamic_cache_for_fx
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"DynamicCache is already registered as pytree flattenable. {e}")

def _flatten_encoder_decoder_cache(cache: EncoderDecoderCache):
    # EncoderDecoderCache는 구조적으로 self/cross cache를 가집니다.
    dictionary = {
        "self_attention_cache": cache.self_attention_cache,
        "cross_attention_cache": cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten(dictionary)

def _unflatten_encoder_decoder_cache(values, context: torch.utils._pytree.Context):
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    # __init__(*caches) 시그니처에 맞춰 복원
    return EncoderDecoderCache(
        dictionary["self_attention_cache"], 
        dictionary["cross_attention_cache"]
    )

def _flatten_with_keys_encoder_decoder_cache(cache: EncoderDecoderCache):
    dictionary = {
        "self_attention_cache": cache.self_attention_cache,
        "cross_attention_cache": cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten_with_keys(dictionary)

def _flatten_encoder_decoder_cache_for_fx(cache, spec):
    dictionary = {
        "self_attention_cache": cache.self_attention_cache,
        "cross_attention_cache": cache.cross_attention_cache,
    }
    return torch.fx._pytree._dict_flatten_spec(dictionary, spec)

def register_encoder_decoder_cache():
    try:
        torch.utils._pytree.register_pytree_node(
            EncoderDecoderCache,
            _flatten_encoder_decoder_cache,
            _unflatten_encoder_decoder_cache,
            serialized_type_name=f"{EncoderDecoderCache.__module__}.{EncoderDecoderCache.__name__}",
            flatten_with_keys_fn=_flatten_with_keys_encoder_decoder_cache,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            EncoderDecoderCache, _flatten_encoder_decoder_cache_for_fx
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"EncoderDecoderCache is already registered as pytree flattenable. {e}")
