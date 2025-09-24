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
from torch.export import Dim
from test.utils.tag import use_onert


class SimpleCopy(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(5, 5), torch.randn(5, 5)), {}


class SimpleCopyWithBroadcastTo(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(5, 5), torch.randn(1, 5)), {}

@use_onert
class SimpleCopyWithBroadcastToDynamicShape(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(5, 5), torch.randn(1, 5)), {}
    
    def get_dynamic_shapes(self):
        dim = Dim("dim", min=1, max=128)
        dynamic_shapes = {
            "dst": {0: dim},
            "src": {},
            }
        return dynamic_shapes