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
from transformers import AutoModelForImageTextToText

from test.modules.base import TestModuleBase


class SmolVLM_text_model(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.rtol = 1e-2
        self.atol = 1e-2
        self.model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct"
        ).model.text_model.to("cpu")

    def forward(self, *args, **kwargs):

        return self.model(*args, **kwargs)

    def get_example_inputs(self):
        kwargs = {
            "inputs_embeds": torch.randn(1, 1739, 576),
            "attention_mask": torch.ones(1, 1739, dtype=torch.int32),
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        return (), kwargs
