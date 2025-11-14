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

from typing import Dict, List, TYPE_CHECKING

import torch

from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention


def llama_attention_forward_adapter(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    position_embeddings: List[torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_value: DynamicCache,
    cache_position: torch.Tensor,
    **kwargs,
):
    # past_key_value is a dict with key_cache and value_cache.
    # It needs to be decomposed for tico and circle which does not know dict.
    key_cache = past_key_value.key_cache  # type: ignore[union-attr]
    value_cache = past_key_value.value_cache  # type: ignore[union-attr]
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
            key_cache[self.layer_idx],
            value_cache[self.layer_idx],  # Same to value_cache
            cache_position,
        ),
        None,
    )
