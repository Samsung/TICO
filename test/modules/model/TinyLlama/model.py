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
from transformers import AutoModelForCausalLM

from test.modules.base import TestModuleBase


class TinyLlama(TestModuleBase):
    def __init__(self):
        super().__init__()
        # The checkpoint is stored in bfloat16; load it as float32 for the
        # reference comparison.
        self.model = AutoModelForCausalLM.from_pretrained(
            "Maykeye/TinyLLama-v0", dtype=torch.float32
        ).to("cpu")
        self.rtol = 1e-4
        self.atol = 1e-4
        # The model returns its KV cache as a DynamicCache, which must be
        # pytree-flattenable for torch.export.
        register_dynamic_cache()
        register_dynamic_layer()

    def forward(self, x):
        return self.model(x)

    def get_example_inputs(self):
        # >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=True, from_slow=True)
        # >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
        # [1, 15043, 29871, 1, 869]
        return (torch.Tensor([[1, 15043, 29871, 1, 869]]).to(dtype=torch.int32),), {}
