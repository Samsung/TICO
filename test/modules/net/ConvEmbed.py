# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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


class ConvEmbed(TestModuleBase):
    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = torch.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor, size: tuple):
        H, W = size
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        return x, (H, W)

    def get_example_inputs(self):
        H = W = 768
        return (torch.randn(1, 3, H, W), (H, W)), {}


class ConvEmbedWithKwargs(TestModuleBase):
    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = torch.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor, size: tuple):
        H, W = size
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        return x, (H, W)

    def get_example_inputs(self):
        H = W = 768
        return (torch.randn(1, 3, H, W),), {"size": (H, W)}
