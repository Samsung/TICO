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

import torch
from tico.utils.pytree_utils import register_dynamic_cache
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

from test.modules.base import TestModuleBase


class LlamaAttentionWithKVCache(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.config = LlamaConfig(use_cache=True)
        self.model = LlamaAttention(config=self.config, layer_idx=0).to("cpu")
        self.rtol = 1e-4
        self.atol = 1e-4

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_example_inputs(self):
        seq_len = 1  # Assume token generation
        hidden_size = self.config.hidden_size
        head_dim = self.config.head_dim
        num_heads = self.config.num_attention_heads

        hidden_states = torch.randn(1, seq_len, hidden_size)
        position_embeddings = (
            torch.randn(1, seq_len, head_dim),
            torch.randn(1, seq_len, head_dim),
        )
        attention_mask = torch.Tensor([[[[0.0]] * seq_len]])  # shape: 1, 1, seq_len, 1
        # This attention_mask will become a causal_mask of shape: (batch_size, 1, query_length, key_value_length)
        prev_seq_len = 4
        past_key_values = DynamicCache()
        register_dynamic_cache()

        past_key_values.update(
            torch.randn(1, num_heads, prev_seq_len, head_dim),
            torch.randn(1, num_heads, prev_seq_len, head_dim),
            0,
        )
        return (
            hidden_states,
            position_embeddings,
            attention_mask,
        ), {"past_key_values": past_key_values}
