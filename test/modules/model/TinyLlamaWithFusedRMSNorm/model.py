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
from tico.passes.module_fusion import llama_rmsnorm

from tico.passes.module_fusion.fusion_registry import replace_modules_with_fused
from tico.utils.pytree_utils import register_dynamic_cache

from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from test.modules.base import TestModuleBase


class TinyLlamaWithFusedRMSNorm(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0").to(
            "cpu"
        )

        self.rtol = 1e-4
        self.atol = 1e-4

        replace_modules_with_fused(self.model, [LlamaRMSNorm])
        register_dynamic_cache()

    def forward(self, x):
        return self.model(x)

    def get_example_inputs(self):
        # >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=True, from_slow=True)
        # >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
        # [1, 15043, 29871, 1, 869]
        return (torch.Tensor([[1, 15043, 29871, 1, 869]]).to(dtype=torch.int32),), {}
