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
from tico.utils.pytree_utils import register_dynamic_cache, register_dynamic_layer
from transformers import LlamaConfig, LlamaModel
from transformers.cache_utils import DynamicCache

from test.modules.base import TestModuleBase


class LlamaWithKVCache(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.config = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            use_cache=True,
        )
        self.model = LlamaModel(config=self.config).to("cpu")
        self.rtol = 1e-4
        self.atol = 1e-4
        # DynamicCache is both an input and an output here; register the
        # pytree flatten/unflatten so torch.export can trace through it.
        register_dynamic_cache()
        register_dynamic_layer()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_example_inputs(self):
        past_seq_len = 511
        cur_seq_len = 1
        input_ids = torch.tensor([[812]]).to(torch.long)
        attention_mask = torch.ones(1, past_seq_len + cur_seq_len)
        position_ids = torch.tensor([[past_seq_len]]).to(torch.long)

        past_key_values = DynamicCache()
        for layer_id in range(self.config.num_hidden_layers):
            past_key_values.update(
                torch.randn(
                    [
                        1,
                        self.config.num_attention_heads,
                        past_seq_len,
                        self.config.head_dim,
                    ]
                ),
                torch.randn(
                    [
                        1,
                        self.config.num_attention_heads,
                        past_seq_len,
                        self.config.head_dim,
                    ]
                ),
                layer_id,
            )

        return (
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        ), {}
