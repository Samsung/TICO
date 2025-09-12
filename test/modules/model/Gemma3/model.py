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
from transformers import Gemma3ForCausalLM
from transformers.integrations.executorch import sdpa_mask_without_vmap
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from test.modules.base import TestModuleBase
from test.utils import tag


@tag.use_onert
class Gemma3(TestModuleBase):
    def __init__(self):
        super().__init__()

        ckpt = "google/gemma-3-270m"

        self.model = Gemma3ForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.float32, device_map="cpu"
        )
        # This is the same as sdpa, but mask creation does not use `vmap` which is not exportable
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            "sdpa_without_vmap", sdpa_mask_without_vmap
        )
        ALL_ATTENTION_FUNCTIONS.register(
            "sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"]
        )
        self.model.config._attn_implementation = "sdpa_without_vmap"
        self.model.config.use_cache = False

        self.rtol = 1e-3
        self.atol = 1e-3

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_example_inputs(self):
        input_ids = torch.randint(0, 100000, (1, 7), dtype=torch.int64)
        attention_mask = torch.ones((1, 7), dtype=torch.int64)

        return (
            (input_ids,),
            {
                "attention_mask": attention_mask,
            },
        )
