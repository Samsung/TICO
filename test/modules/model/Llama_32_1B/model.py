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

from test.modules.base import TestModuleBase


class Llama_32_1B(TestModuleBase):

    def __init__(self):
        super().__init__()
        self.model = LlamaModel(config=LlamaConfig(
            _attn_implementation_autoset=True,
            architectures=['LlamaForCausalLM'],
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=128000,
            eos_token_id=128001,
            head_dim=64,
            hidden_act='silu',
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=8192,
            max_position_embeddings=131072,
            mlp_bias=False,
            model_type='llama',
            num_attention_heads=32,
            num_hidden_layers=16,
            num_key_value_heads=8,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling={
                'factor': 32.0,
                'high_freq_factor': 4.0,
                'low_freq_factor': 1.0,
                'original_max_position_embeddings': 8192,
                'rope_type': 'llama3'
            },
            rope_theta=500000.0,
            tie_word_embeddings=True,
            torch_dtype=torch.float16,
            use_cache=True,
            vocab_size=128256,
        )).to("cpu")
        self.rtol = 1e-4
        self.atol = 1e-4

    def forward(self, x):
        return self.model(x)

    def get_example_inputs(self):
        # >>> tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B")
        # >>> tokenizer.encode("Hello my name is")
        # [128000, 9906, 856, 836, 374]
        return (torch.Tensor([[12800, 9906, 856, 836,
                               374]]).to(dtype=torch.int32), ), {}
