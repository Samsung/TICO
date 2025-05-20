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


class SimpleNativeGroupNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 10
        self.C = 6
        self.H = 5
        self.W = 5

    def forward(self, tensor, weight, bias, group, eps):
        z = torch.native_group_norm(
            tensor, weight, bias, self.N, self.C, self.H * self.W, group, eps
        )[0]
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.C, self.H, self.W)
        weight = torch.randn(self.C)
        bias = torch.randn(self.C)
        group = 3
        eps = 1e-5
        return (
            tensor,
            weight,
            bias,
            group,
            eps,
        )


class SimpleNativeGroupNormWithoutWeightBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 10
        self.C = 6
        self.H = 5
        self.W = 5

    def forward(self, tensor, group, eps):
        z = torch.native_group_norm(
            tensor, None, None, self.N, self.C, self.H * self.W, group, eps
        )[0]
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.C, self.H, self.W)
        group = 3
        eps = 1e-5
        return (
            tensor,
            group,
            eps,
        )


class SimpleNativeGroupNormRedundantReshape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 20
        self.C = 1
        self.H = 512

    def forward(self, tensor, group, eps):
        z = torch.native_group_norm(
            tensor, None, None, self.N, self.C, self.H, group, eps
        )[0]
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.C, self.H)
        group = 1
        eps = 1e-5
        return (
            tensor,
            group,
            eps,
        )


class SimpleNativeGroupNormWithLayerNormClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 20
        self.C = 6
        self.H = 10
        self.W = 10
        num_groups = 3
        num_channels = self.C
        self.group_norm = torch.nn.GroupNorm(num_groups, num_channels)

    def forward(self, tensor):
        z = self.group_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.C, self.H, self.W)
        return (tensor,)


class SimpleNativeGroupNormWithLayerNorm3DInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 20
        self.C = 6
        self.W = 10
        num_groups = 3
        num_channels = self.C
        self.group_norm = torch.nn.GroupNorm(num_groups, num_channels)

    def forward(self, tensor):
        z = self.group_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.C, self.W)
        return (tensor,)


class SimpleNativeGroupNormNonAffine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 20
        self.C = 6
        self.W = 10
        num_groups = 3
        num_channels = self.C
        self.group_norm = torch.nn.GroupNorm(num_groups, num_channels, affine=False)

    def forward(self, tensor):
        z = self.group_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.C, self.W)
        return (tensor,)
