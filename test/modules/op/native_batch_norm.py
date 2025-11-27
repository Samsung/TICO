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


class SimpleBatchNorm2D(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.batch_norm = torch.nn.BatchNorm2d(self.num_features)

    def forward(self, tensor):
        z = self.batch_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(2, self.num_features, 3, 3)
        return (tensor,), {}


class BatchNorm2DWithNoAffine(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.batch_norm = torch.nn.BatchNorm2d(self.num_features, affine=False)

    def forward(self, tensor):
        z = self.batch_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(2, self.num_features, 3, 3)
        return (tensor,), {}


class NativeBatchNormLegitNoTraining(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.register_buffer("mean", torch.tensor([9.0, 10.0, 11.0, 12.0]))
        self.register_buffer("variance", torch.tensor([13.0, 14.0, 15.0, 16.0]))
        self.register_parameter(
            "weight", torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        )
        self.register_parameter(
            "bias", torch.nn.Parameter(torch.tensor([5.0, 6.0, 7.0, 8.0]))
        )

    def forward(self, tensor):
        z = torch.ops.aten._native_batch_norm_legit_no_training.default(
            tensor, self.weight, self.bias, self.mean, self.variance, 0.1, 1e-05
        )
        # Not use unused outputs
        return (z[0],)

    def get_example_inputs(self):
        tensor = torch.randn(2, self.num_features, 3, 3)
        return (tensor,), {}


class SimpleBatchNorm1D(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.batch_norm = torch.nn.BatchNorm1d(self.num_features)

    def forward(self, tensor):
        z = self.batch_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        # (N, C)
        tensor = torch.randn(2, self.num_features)
        return (tensor,), {}


class BatchNorm1DWithNoAffine(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.batch_norm = torch.nn.BatchNorm1d(self.num_features, affine=False)

    def forward(self, tensor):
        z = self.batch_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(2, self.num_features)
        return (tensor,), {}


class NativeBatchNormLegitNoTraining1DInput(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.register_buffer("mean", torch.tensor([9.0, 10.0, 11.0, 12.0]))
        self.register_buffer("variance", torch.tensor([13.0, 14.0, 15.0, 16.0]))
        self.register_parameter(
            "weight", torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        )
        self.register_parameter(
            "bias", torch.nn.Parameter(torch.tensor([5.0, 6.0, 7.0, 8.0]))
        )

    def forward(self, tensor):
        z = torch.ops.aten._native_batch_norm_legit_no_training.default(
            tensor, self.weight, self.bias, self.mean, self.variance, 0.1, 1e-05
        )
        return (z[0],)

    def get_example_inputs(self):
        # (N, C)
        tensor = torch.randn(2, self.num_features)
        return (tensor,), {}


class SimpleBatchNorm1DSequence(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.batch_norm = torch.nn.BatchNorm1d(self.num_features)

    def forward(self, tensor):
        z = self.batch_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        # (N, C, L)
        tensor = torch.randn(2, self.num_features, 5)
        return (tensor,), {}
