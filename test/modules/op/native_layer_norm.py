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


class SimpleNativeLayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor, normalized_shape, weight, bias, eps):
        z = torch.native_layer_norm(tensor, normalized_shape, weight, bias, eps)[0]
        return (z,)

    def get_example_inputs(self):
        N = 4
        H = 3
        W = 2
        C = 1

        tensor = torch.randn(N, H, W, C)
        normalized_shape = [H, W, C]
        weight = torch.randn(H, W, C)
        bias = torch.randn(H, W, C)
        eps = 1e-5
        return (
            tensor,
            normalized_shape,
            weight,
            bias,
            eps,
        )


class SimpleNativeLayerNormWithoutWeightBias(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor, normalized_shape, eps):
        z = torch.native_layer_norm(tensor, normalized_shape, None, None, eps)[0]
        return (z,)

    def get_example_inputs(self):
        N = 4
        H = 3
        W = 2
        C = 1

        tensor = torch.randn(N, H, W, C)
        normalized_shape = [H, W, C]
        eps = 1e-5
        return (
            tensor,
            normalized_shape,
            eps,
        )


class SimpleNativeLayerNormRedundantReshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor, normalized_shape, eps):
        z = torch.native_layer_norm(tensor, normalized_shape, None, None, eps)[0]
        return (z,)

    def get_example_inputs(self):
        N = 4
        H = 3
        W = 2
        C = 1

        tensor = torch.randn(N, H, W, C)
        normalized_shape = [C]
        eps = 1e-5
        return (
            tensor,
            normalized_shape,
            eps,
        )


class NativeLayerNormChannelLastInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor, normalized_shape, weight, bias, eps):
        z = torch.native_layer_norm(tensor, normalized_shape, weight, bias, eps)[0]
        return (z,)

    def get_example_inputs(self):
        N = 4
        C = 3
        H = 2
        W = 1

        tensor = torch.randn(N, C, H, W).to(memory_format=torch.channels_last)
        normalized_shape = [C, H, W]
        weight = torch.randn(C, H, W)
        bias = torch.randn(C, H, W)
        eps = 1e-5
        return (
            tensor,
            normalized_shape,
            weight,
            bias,
            eps,
        )


class SimpleNativeLayerNormForMultiDimensionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor, normalized_shape, weight, bias, eps):
        z = torch.native_layer_norm(tensor, normalized_shape, weight, bias, eps)[0]
        return (z,)

    def get_example_inputs(self):
        N = 4
        H = 3
        W = 2
        C = 1

        # The input tensor will be normalized over the last two dimensions.
        tensor = torch.randn(N, H, W, C)
        normalized_shape = [W, C]
        weight = torch.randn(W, C)
        bias = torch.randn(W, C)
        eps = 1e-5
        return (
            tensor,
            normalized_shape,
            weight,
            bias,
            eps,
        )


class SimpleNativeLayerNormWithLayerNormClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 4
        self.H = 3
        self.W = 2
        self.C = 1
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=[self.H, self.W, self.C], eps=1e-5
        )

    def forward(self, tensor):
        z = self.layer_norm(tensor)
        return (z,)

    def get_example_inputs(self):

        tensor = torch.randn(self.N, self.H, self.W, self.C)
        return (tensor,)


class SimpleNativeLayerNormNonAffine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 4
        self.H = 3
        self.W = 2
        self.C = 1
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=[self.H, self.W, self.C],
            eps=1e-5,
            elementwise_affine=False,
        )

    def forward(self, tensor):
        z = self.layer_norm(tensor)
        return (z,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.H, self.W, self.C)
        return (tensor,)
