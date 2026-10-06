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
from transformers import AutoTokenizer, MambaForCausalLM

from test.modules.base import TestModuleBase


class Mamba(TestModuleBase):
    def __init__(self):
        super().__init__()

        # Prevent the result from being nan
        # WARNING This removes non-determinism only partially.
        torch.use_deterministic_algorithms(True)

        # Keep the checkpoint configuration (24 layers). A default `MambaConfig`
        # has 32 layers, so the last 8 would be randomly initialized and the
        # torch/circle comparison would depend on the random seed.
        self.model = MambaForCausalLM.from_pretrained(
            "state-spaces/mamba-130m-hf", use_cache=False
        ).to("cpu")

        self.rtol = 1e-3
        self.atol = 1e-3

    def forward(self, x):
        return self.model(x)

    def get_example_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        input_ids = tokenizer("What is your name?", return_tensors="pt")[
            "input_ids"
        ].to("cpu")
        return (input_ids,), {}
