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


class SimpleAtenIndexTensorAxis1(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([1, 2, 3, 4, 5]))

    def forward(self, x):
        y = x[:, self.idx]
        return y

    def get_example_inputs(self):
        return (torch.randn(3, 20, 4, 5),), {}


class SimpleAtenIndexTensorAxis2(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([1, 2, 3, 4, 5]))

    def forward(self, x):
        y = x[:, :, self.idx]
        return y

    def get_example_inputs(self):
        return (torch.randn(3, 1, 2048, 5),), {}


class SimpleIndexTensorBuffer(TestModuleBase):
    def __init__(self):
        super().__init__()
        idx = torch.tensor([0, 1])
        self.register_buffer("idx", idx)

    def forward(self, x, y):
        z = x + y
        return z[self.idx]

    def get_example_inputs(self):
        return (
            torch.randn(2, 3),
            torch.randn(2, 3),
        ), {}


class SimpleIndexTensor(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([0, 1]))

    def forward(self, x, y):
        z = x + y
        return z[self.idx]

    def get_example_inputs(self):
        return (
            torch.randn(2, 3),
            torch.randn(2, 3),
        ), {}


class IndexTensor2x1(TestModuleBase):
    def __init__(self):
        super().__init__()
        idx = torch.tensor([[0], [0]])
        self.register_buffer("idx", idx)

    def forward(self, x, y):
        z = x + y
        return z[self.idx]

    def get_example_inputs(self):
        return (
            torch.randn(2, 3),
            torch.randn(2, 3),
        ), {}


class IndexTensorAxis1(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.register_buffer("axis", torch.tensor([0, 0]))

    def forward(self, x, y):
        z = x + y
        return z[1, self.axis]

    def get_example_inputs(self):
        return (
            torch.randn(2, 3),
            torch.randn(2, 3),
        ), {}


class IndexTensorAxis0And1(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.register_buffer("axis", torch.tensor([2, 3]))

    def forward(self, x):
        return x[1, self.axis]

    def get_example_inputs(self):
        return (torch.randn(5, 4),), {}


class IndexTensorWithSlice(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.register_buffer("axis", torch.tensor([0, 0]))

    def forward(self, x, y):
        z = x + y
        return z[1:, self.axis, :, :2]

    def get_example_inputs(self):
        return (
            torch.randn(3, 3, 6, 6),
            torch.randn(3, 3, 6, 6),
        ), {}


class IndexTensorTwoIndices(TestModuleBase):
    """Two index tensors of the same shape select individual elements."""

    def __init__(self):
        super().__init__()
        self.register_buffer("rows", torch.tensor([0, 2, 1, 2, 0]))
        self.register_buffer("cols", torch.tensor([3, 0, 1, 2, 3]))

    def forward(self, x):
        return x[self.rows, self.cols]

    def get_example_inputs(self):
        return (torch.randn(3, 4),), {}


class IndexTensorTwoBroadcastIndices(TestModuleBase):
    """
    Two broadcast index tensors, as emitted by transformers' padding mask
    (`mask[batch_idx, kv_idx]` with shapes [1, 1, 1, 1] and [1, 1, 1, L]).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("batch_idx", torch.zeros(1, 1, 1, 1, dtype=torch.int64))
        self.register_buffer("kv_idx", torch.arange(16).reshape(1, 1, 1, 16))

    def forward(self, mask):
        return mask[self.batch_idx, self.kv_idx]

    def get_example_inputs(self):
        return (torch.randint(0, 2, (1, 16), dtype=torch.int32),), {}


class IndexTensorThreeIndicesTrailingDims(TestModuleBase):
    """Three broadcast index tensors on a rank-4 input keep the last dimension."""

    def __init__(self):
        super().__init__()
        self.register_buffer("i0", torch.tensor([[1], [0]]))
        self.register_buffer("i1", torch.tensor([[0, 2, 1]]))
        self.register_buffer("i2", torch.tensor([[3, 0, 1], [2, 2, 0]]))

    def forward(self, x):
        return x[self.i0, self.i1, self.i2]

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4, 5),), {}
