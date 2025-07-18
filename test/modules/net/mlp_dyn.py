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
from torch.export import Dim

from test.utils.tag import use_onert

B = 4
SEQ_LEN = 8
DIM = 16
INTERMEDATE = 64


@use_onert
class MLP_DynamicShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(DIM, INTERMEDATE, bias=False)
        self.up_proj = torch.nn.Linear(DIM, INTERMEDATE, bias=False)
        self.down_proj = torch.nn.Linear(INTERMEDATE, DIM, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )
        return down_proj

    def get_example_inputs(self):
        return (torch.randn(B, SEQ_LEN, DIM),)

    def get_dynamic_shapes(self):
        batch = Dim("batch", min=1, max=128)
        dynamic_shapes = {
            "x": {0: batch},
        }

        return dynamic_shapes
