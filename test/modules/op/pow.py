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

# TODO Add a test for `aten.pow.Scalar` when the operator is supported.


class SimplePowTensorScalar(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.pow(x, exponent=2.0)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 2),), {}


class SimplePowIntTensorFloatScalar(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.to(torch.int32)
        z = torch.pow(x, exponent=2.0)
        return z

    def get_example_inputs(self):
        return (torch.tensor([[-50000, -1], [1, 50000]], dtype=torch.int32),), {}


class Int32TensorInt64Scalar(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.to(torch.int32)
        z = torch.pow(x, exponent=2)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 2),), {}


class SimplePowFloatTensorIntScalar(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        e = 2
        z = torch.pow(x, exponent=e)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 2),), {}


class SimplePowFloatTensorIntScalar2(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        e = 5
        z = torch.pow(x, exponent=e)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 2),), {}


class SimplePowFloatTensorIntTensor(TestModuleBase):
    def __init__(self):
        super().__init__()
        e = torch.tensor(2, dtype=torch.int32)
        self.register_buffer("e", e)

    def forward(self, x):
        z = torch.pow(x, exponent=self.e)  # type: ignore[arg-type]
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 2),), {}


class SimplePowIntTensorFloatTensor(TestModuleBase):
    def __init__(self):
        super().__init__()
        e = torch.tensor(2.4)
        self.register_buffer("e", e)

    def forward(self, x):
        x = x.to(torch.int32)
        z = torch.pow(x, exponent=self.e)  # type: ignore[arg-type]
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 2),), {}


class SimplePowTensorTensor(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = torch.pow(x, y)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 2), torch.randn(2, 2)), {}
