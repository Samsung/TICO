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
from transformers import GPT2Config, GPT2LMHeadModel

from test.modules.base import TestModuleBase


class GPT2(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel(config=GPT2Config.from_pretrained("gpt2")).to(
            "cpu"
        )
        # The model returns its KV cache as a DynamicCache, which must be
        # pytree-flattenable for torch.export.
        register_dynamic_cache()
        register_dynamic_layer()

    def forward(self, x):
        return self.model(x)

    def get_example_inputs(self):
        # >>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        # >>> tokenizer("Hello world", return_tensors="pt")["input_ids"]
        # [[15496, 995]]
        input_ids = torch.Tensor([[15496, 995]]).to(dtype=torch.int32)
        return (input_ids,), {}
