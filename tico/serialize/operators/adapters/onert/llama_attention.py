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

from typing import cast, Dict, List, TYPE_CHECKING

import torch

from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention

if TYPE_CHECKING:
    from transformers.cache_utils import CacheLayerMixin


def llama_attention_forward_adapter(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    position_embeddings: List[torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_value: DynamicCache,
    cache_position: torch.Tensor,
    **kwargs,
):
    # Decompose the cache into tensors before calling the Circle custom operator.
    if hasattr(past_key_value, "layers"):
        layer = cast("CacheLayerMixin", past_key_value.layers[self.layer_idx])
        past_key, past_value = layer.keys, layer.values
    else:
        past_key = getattr(past_key_value, "key_cache")[self.layer_idx]
        past_value = getattr(past_key_value, "value_cache")[self.layer_idx]
    assert past_key is not None and past_value is not None
    return (
        torch.ops.circle_custom.attention(
            hidden_states,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.o_proj.weight,
            position_embeddings[0],  # cos
            position_embeddings[1],  # sin
            attention_mask,
            past_key,
            past_value,
            cache_position,
        ),
        None,
    )
