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
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

from test.modules.base import TestModuleBase
from test.utils.tag import target


seq_len = 5


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # LlamaConfig extracted from meta-llama/Llama-3.2-1B
        # Adjust num_hidden_layers to 1 for testing purpose
        self.config = LlamaConfig(
            _attn_implementation_autoset=True,
            architectures=["LlamaForCausalLM"],
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=128000,
            eos_token_id=128001,
            head_dim=64,
            hidden_act="silu",
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=8192,
            max_position_embeddings=131072,
            mlp_bias=False,
            model_type="llama",
            num_attention_heads=32,
            num_hidden_layers=1,
            num_key_value_heads=8,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling={
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            rope_theta=500000.0,
            tie_word_embeddings=True,
            torch_dtype=torch.float16,
            use_cache=True,
            vocab_size=128256,
        )
        self.model = LlamaDecoderLayer(config=self.config, layer_idx=0)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.cache_position = torch.arange(seq_len)
        position_ids = self.cache_position.unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        past_key_values = DynamicCache()
        position_embeddings = self.rotary_emb(hidden_states, self.position_ids)
        last_hidden_state = self.model.forward(
            *args,
            **{
                "position_ids": self.position_ids,
                "cache_position": self.cache_position,
                "past_key_value": past_key_values,
                "use_cache": self.config.use_cache,
                "position_embeddings": position_embeddings,
            },
        )
        return (
            last_hidden_state[0],
            past_key_values.to_legacy_cache(),
        )


@target
class Llama_32_1B_Decoder(TestModuleBase):
    def __init__(self):
        super().__init__()

        self.model = Wrapper()

        self.rtol = 1e-4
        self.atol = 1e-4

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_example_inputs(self):
        hidden_states = torch.rand([1, seq_len, 2048])
        return (hidden_states,), {}
