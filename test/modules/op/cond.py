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

class SimpleCond1(TestModuleBase):
    class Sin(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x) + 1

    class Cos(torch.nn.Module):
        def forward(self, x):
            return torch.cos(x) - 1

    def __init__(self):
        super().__init__()
        self.sin = self.Sin()
        self.cos = self.Cos()

    def forward(self, x, y):
        return torch.cond(x.sum() + y.sum() > 0,
                        lambda x_: self.sin(x_),
                          lambda x_: self.cos(x_),
                          operands=(x,))
    def get_example_inputs(self):
        return (torch.randn(3, 3), torch.randn(3, 3)), {}
    
    

class SimpleCond2(TestModuleBase):
    class Sin(torch.nn.Module):
        def forward(self, x, y):
            return torch.sin(x) + 1

    class Cos(torch.nn.Module):
        def forward(self, x, y):
            return torch.cos(x) - 1

    def __init__(self):
        super().__init__()
        self.sin = self.Sin()
        self.cos = self.Cos()

    def forward(self, x, y):
        return torch.cond(x.sum() + y.sum() > 0,
                          lambda x, y: self.sin(x, y),
                          lambda x, y: self.cos(x, y),
                          operands=(x,y))
    def get_example_inputs(self):
        return (torch.randn(3, 3), torch.randn(3, 3)), {}


if __name__ == "__main__":
    model = SimpleCond2()
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)

    # export (그래프 생성)
    exported_model = torch.export.export(model, (x, y))

    # export된 모델 호출 테스트
    output = exported_model.module()(x, y)
    exported_model.graph.print_tabular()
    print(exported_model.graph_signature.user_inputs)
    print(exported_model.graph_signature.user_outputs)
    print(output)
    breakpoint()