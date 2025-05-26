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


class SimpleMaxDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        max = x.max(dim=0)
        return max[0]  # max: tuple (max, max_indices)

    def get_example_inputs(self):
        return (torch.randn(2, 3),)


class MaxDimKeepDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        max = x.max(dim=0, keepdim=True)
        return max[0]  # max: tuple (max, max_indices)

    def get_example_inputs(self):
        return (torch.randn(2, 3),)
