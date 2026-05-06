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

import torch

from tico.passes.remove_unused_placeholder import RemoveUnusedPlaceholder
from torch.export import export

from test.utils.pass_value_test import SinglePassValueTest


class UsedBiasNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.bias = torch.nn.Parameter(torch.randn(4))

    def forward(self, x):
        return torch.ops.aten.linear.default(x, self.weight, self.bias)

    def get_example_inputs(self):
        return (torch.randn(2, 4),), {}


def get_placeholder_names(exported_program):
    return [
        node.name for node in exported_program.graph.nodes if node.op == "placeholder"
    ]


class RemoveUnusedPlaceholderTest(SinglePassValueTest):
    def test_remove_unused_placeholder(self):
        self.setup(UsedBiasNet())

        graph = self.exported_program().graph

        bias_node = None
        linear_node = None

        for node in graph.nodes:
            if node.op == "placeholder" and "bias" in node.name:
                bias_node = node

            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.linear.default
            ):
                linear_node = node

        assert bias_node is not None
        assert linear_node is not None

        # Simulate QuantizeBias behavior.
        linear_node.update_arg(2, None)

        graph.eliminate_dead_code()

        placeholder_names_before = get_placeholder_names(self.exported_program())
        assert any("bias" in name for name in placeholder_names_before)

        self.run_value_test(RemoveUnusedPlaceholder())

        placeholder_names_after = get_placeholder_names(self.exported_program())
        assert not any("bias" in name for name in placeholder_names_after)

    def test_keep_used_placeholder(self):
        self.setup(UsedBiasNet())

        placeholder_names_before = get_placeholder_names(self.exported_program())
        assert any("bias" in name for name in placeholder_names_before)

        self.run_value_test(RemoveUnusedPlaceholder())

        placeholder_names_after = get_placeholder_names(self.exported_program())
        assert any("bias" in name for name in placeholder_names_after)
