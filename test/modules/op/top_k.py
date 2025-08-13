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

from test.modules.base import TestModuleBase
from test.utils.tag import use_onert

# luci-interpreter doesn't support TopK operator yet
@use_onert
class SimpleTopK(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        values, indices = torch.topk(x, 2)
        return values, indices

    def get_example_inputs(self):
        batch_size = 1
        seq_len = 63
        num_experts = 8
        return (torch.randn(batch_size * seq_len, num_experts),), {}
