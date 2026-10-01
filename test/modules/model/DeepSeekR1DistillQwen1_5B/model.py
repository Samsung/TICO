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
from transformers import AutoConfig, AutoModelForCausalLM

from test.modules.base import TestModuleBase


class DeepSeek_R1_Distill_Qwen_1_5B(TestModuleBase):
    """
    DeepSeek-R1-Distill-Qwen-1.5B Decoder layer
    """

    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        config.use_cache = False
        # The checkpoint is stored in bfloat16; load it as float32 for the
        # reference comparison.
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            config=config,
            dtype=torch.float32,
        )
        self.rtol = 1e-4
        self.atol = 1e-4

    def forward(self, *args, **kwargs):
        return self.model.model.layers[0](*args, **kwargs)

    def get_example_inputs(self):
        batch = 1
        seq_len = 5
        hidden_size = 1536

        torch.manual_seed(5)
        hidden_states = torch.randn(batch, seq_len, hidden_size)
        position_ids = [[i for i in range(0, seq_len)]]
        position_embeddings = (
            torch.randn(batch, seq_len, 128),
            torch.randn(batch, seq_len, 128),
        )

        return (hidden_states,), {
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
        }
