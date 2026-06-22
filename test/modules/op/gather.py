# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

"""Modules that exercise torch.gather value tests through Circle export."""

import torch

from test.modules.base import TestModuleBase
from test.support.tag import skip


def _input_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    """Return a deterministic float tensor for value comparison."""
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.arange(numel, dtype=torch.float32).reshape(shape)


@skip(reason="No kernel exist")
class SimpleGatherDim0(TestModuleBase):
    """Gather along dimension 0 with a constant index tensor."""

    def __init__(self):
        super().__init__()
        index = torch.arange(2 * 4 * 5, dtype=torch.long).reshape(2, 4, 5) % 3
        self.register_buffer("index", index)

    def forward(self, input_):
        """Return torch.gather over dimension 0."""
        return torch.gather(input_, dim=0, index=self.index)

    def get_example_inputs(self):
        """Return an input tensor compatible with the constant index."""
        return (_input_tensor((3, 4, 5)),), {}


@skip(reason="No kernel exist")
class SimpleGatherDim1(TestModuleBase):
    """Gather along dimension 1 with a constant index tensor."""

    def __init__(self):
        super().__init__()
        index = torch.arange(3 * 2 * 5, dtype=torch.long).reshape(3, 2, 5) % 4
        self.register_buffer("index", index)

    def forward(self, input_):
        """Return torch.gather over dimension 1."""
        return torch.gather(input_, dim=1, index=self.index)

    def get_example_inputs(self):
        """Return an input tensor compatible with the constant index."""
        return (_input_tensor((3, 4, 5)),), {}


@skip(reason="No kernel exist")
class SimpleGatherDim2(TestModuleBase):
    """Gather along dimension 2 with a constant index tensor."""

    def __init__(self):
        super().__init__()
        index = torch.arange(3 * 4 * 2, dtype=torch.long).reshape(3, 4, 2) % 5
        self.register_buffer("index", index)

    def forward(self, input_):
        """Return torch.gather over dimension 2."""
        return torch.gather(input_, dim=2, index=self.index)

    def get_example_inputs(self):
        """Return an input tensor compatible with the constant index."""
        return (_input_tensor((3, 4, 5)),), {}


@skip(reason="No kernel exist")
class SimpleGatherDim3(TestModuleBase):
    """Gather along dimension 3 for a rank-4 tensor."""

    def __init__(self):
        super().__init__()
        index = torch.arange(2 * 3 * 4 * 3, dtype=torch.long).reshape(2, 3, 4, 3) % 5
        self.register_buffer("index", index)

    def forward(self, input_):
        """Return torch.gather over dimension 3."""
        return torch.gather(input_, dim=3, index=self.index)

    def get_example_inputs(self):
        """Return an input tensor compatible with the constant index."""
        return (_input_tensor((2, 3, 4, 5)),), {}


@skip(reason="No kernel exist")
class SimpleGatherNegativeDim(TestModuleBase):
    """Gather along the last dimension using a negative dimension value."""

    def __init__(self):
        super().__init__()
        index = torch.arange(2 * 3 * 4 * 3, dtype=torch.long).reshape(2, 3, 4, 3) % 5
        self.register_buffer("index", index)

    def forward(self, input_):
        """Return torch.gather over the last dimension."""
        return torch.gather(input_, dim=-1, index=self.index)

    def get_example_inputs(self):
        """Return an input tensor compatible with the constant index."""
        return (_input_tensor((2, 3, 4, 5)),), {}
