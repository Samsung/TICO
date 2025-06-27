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
from transformers import LlamaConfig, LlamaModel
from transformers.cache_utils import DynamicCache

class LlamaWithKVCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = LlamaConfig(
                hidden_size=512,
                num_hidden_layers=8,
                num_attention_heads=8,
                use_cache=True,
            )
        self.model = LlamaModel(
            config= self.config
        ).to("cpu")
        self.rtol = 1e-4
        self.atol = 1e-4

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_example_inputs(self):
        
        # hidden_states = torch.randn(1, seq_len, self.config.hidden_size)
        # position_embeddings = (torch.randn(1, seq_len, self.config.head_dim), torch.randn(1, seq_len, self.config.head_dim))
        # # attention_mask의 causal_mask 버젼: (batch_size, 1, query_length, key_value_length)
        # attention_mask = torch.Tensor([[[[0.]] * seq_len]]) # shape: 1, 1, 3, 1
        # past_key_values = DynamicCache()
        # past_key_values.update(torch.randn(1, prev_seq_len, self.config.num_attention_heads, self.config.head_dim), torch.randn(1, prev_seq_len, self.config.num_attention_heads, self.config.head_dim), 1)
        # cache_position = torch.tensor([prev_seq_len], dtype=torch.long)
        past_seq_len = 12
        cur_seq_len = 6
        input_ids = torch.tensor([[812]]).to(torch.long)
        attention_mask = torch.ones(1, past_seq_len + cur_seq_len)
        position_ids = torch.tensor([[past_seq_len]]).to(torch.long)
        
        past_key_values = DynamicCache()
        for layer_id in range(self.config.num_hidden_layers):
            past_key_values.update(torch.randn([1, self.config.num_attention_heads, past_seq_len , self.config.head_dim, ]), torch.randn([1, self.config.num_attention_heads, past_seq_len , self.config.head_dim, ]), layer_id)

        # >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=True, from_slow=True)
        # >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
        # [1, 15043, 29871, 1, 869]
        return (input_ids, attention_mask, position_ids, past_key_values,)

# m = LlamaWithKVCache()
# m.forward(*m.get_example_inputs())