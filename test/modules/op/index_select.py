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


class SimpleIndexSelectWithDim0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_, dim, index):
        result = torch.index_select(input_, dim, index)
        return result

    def get_example_inputs(self):
        return (torch.rand(3, 4), 0, torch.tensor([0, 2]))


class SimpleIndexSelectWithDim1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_, dim, index):
        result = torch.index_select(input_, dim, index)
        return result

    def get_example_inputs(self):
        return (torch.rand(3, 4), 1, torch.tensor([0, 2]))


class SimpleIndexSelectWithConstScalarIndex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([3]))

    def forward(self, input_):
        result = torch.index_select(input_, 2, self.idx)
        return result

    def get_example_inputs(self):
        return (torch.rand(3, 4, 5),)


class SimpleIndexSelectWithConstIndex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([3, 1, 2]))

    def forward(self, x, y):
        result = torch.index_select(x, 2, self.idx) + y
        return result

    def get_example_inputs(self):
        return (torch.rand(3, 4, 5), torch.rand(3, 4, 3))
