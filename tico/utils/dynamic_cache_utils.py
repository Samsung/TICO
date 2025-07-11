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

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied from transformers.src.transformers.cache_utils

import torch
from packaging.version import Version

from tico.utils import logging
from tico.utils.installed_packages import is_transformers_installed

if is_transformers_installed:
    import transformers
    from transformers.cache_utils import DynamicCache

    HAS_TORCH_GREATER_OR_EQUAL_2_6_0 = Version(torch.__version__) >= Version("2.6.0")
    HAS_TRANSFORMERS_LESS_4_50_0 = Version(transformers.__version__) < Version("4.50.0")

    # Utilities for `DynamicCache` <> torch.export support
    def _flatten_dynamic_cache(
        dynamic_cache: DynamicCache,
    ):
        """Flattens DynamicCache into flat list of tensors for `torch.export.export` to consume"""
        if not isinstance(dynamic_cache, DynamicCache):
            raise RuntimeError(
                "This pytree flattening function should only be applied to DynamicCache"
            )

        if not HAS_TORCH_GREATER_OR_EQUAL_2_6_0:
            logger = logging.getLogger(__name__)
            logger.warning_once(
                "DynamicCache + torch.export is tested on torch 2.6.0+ and may not work on earlier versions."
            )

        # NOTE it seems _seen_tokens is deprecated, so probably doesn't need tracking
        dictionary = {
            "key_cache": getattr(dynamic_cache, "key_cache"),
            "value_cache": getattr(dynamic_cache, "value_cache"),
        }
        return torch.utils._pytree._dict_flatten(dictionary)

    def _flatten_with_keys_dynamic_cache(dynamic_cache: DynamicCache):
        dictionary = {
            "key_cache": getattr(dynamic_cache, "key_cache"),
            "value_cache": getattr(dynamic_cache, "value_cache"),
        }
        return torch.utils._pytree._dict_flatten_with_keys(dictionary)

    def _unflatten_dynamic_cache(
        values,
        context: torch.utils._pytree.Context,
    ):
        dictionary = torch.utils._pytree._dict_unflatten(values, context)
        cache = DynamicCache()
        for k, v in dictionary.items():
            setattr(cache, k, v)
        return cache

    def _flatten_dynamic_cache_for_fx(cache, spec):
        dictionary = {
            "key_cache": getattr(cache, "key_cache"),
            "value_cache": getattr(cache, "value_cache"),
        }
        return torch.fx._pytree._dict_flatten_spec(dictionary, spec)

    def make_dynamic_cache_exportable():
        # From transformers==4.50.0, DynamicCache is exportable by default.
        if HAS_TRANSFORMERS_LESS_4_50_0:
            torch.utils._pytree.register_pytree_node(
                DynamicCache,
                _flatten_dynamic_cache,
                _unflatten_dynamic_cache,
                serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
                flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
            )
            # TODO (tmanlaibaatar) This won't be needed in torch 2.7.
            torch.fx._pytree.register_pytree_flatten_spec(
                DynamicCache, _flatten_dynamic_cache_for_fx
            )
