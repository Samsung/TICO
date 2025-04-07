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


"""
This module contains Op lists used for finding the target Ops in passes.
The module is introduced to reduce duplicate codes.
It should be guaranteed that Ops in the same list have the same input/output signature.
"""


class AtenOps:
    def __init__(self):
        # In alphabetical order
        self.add = [torch.ops.aten.add.Tensor]
        self.alias = [torch.ops.aten.alias.default, torch.ops.aten.alias_copy.default]
        self.cat = [torch.ops.aten.cat.default]
        self.clamp = [torch.ops.aten.clamp.default, torch.ops.aten.clamp.Tensor]
        self.clone = [torch.ops.aten.clone.default]
        self.conv2d = [
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv2d.padding,
        ]
        self.conv1d = [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv1d.padding,
        ]
        self.detach = [
            torch.ops.aten.detach_.default,
            torch.ops.aten.detach.default,
        ]
        self.expand = [
            torch.ops.aten.expand.default,
            torch.ops.aten.expand_copy.default,
        ]
        self.index_select = [torch.ops.aten.index_select.default]
        self.mean = [torch.ops.aten.mean.dim]
        self.mul_scalar = [torch.ops.aten.mul.Scalar]
        self.mul_tensor = [torch.ops.aten.mul.Tensor]
        self.permute = [torch.ops.aten.permute.default]
        self.reshape = [torch.ops.aten.reshape.default]
        self.select = [torch.ops.aten.select_copy.int, torch.ops.aten.select.int]
        self.slice = [torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor]
        self.softmax = [torch.ops.aten._softmax.default]
        self.squeeze = [torch.ops.aten.squeeze.dims, torch.ops.aten.squeeze_copy.dims]
        self.to_copy = [
            torch.ops.aten._to_copy.default,
            torch.ops.aten.to.dtype,
            torch.ops.aten.to.dtype_layout,
        ]
        self.unsqueeze = [
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.unsqueeze_copy.default,
        ]
        self.view = [
            torch.ops.aten.view,
            torch.ops.aten.view.default,
            torch.ops.aten.view_copy.default,
        ]


aten = AtenOps()
