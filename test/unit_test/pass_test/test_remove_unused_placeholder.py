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


class UsedBufferNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("bias", torch.randn(4))

    def forward(self, x):
        return torch.ops.aten.add.Tensor(x, self.bias)

    def get_example_inputs(self):
        return (torch.randn(2, 4),), {}


class UsedLiftedTensorConstantNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bias = torch.randn(4)

    def forward(self, x):
        return torch.ops.aten.add.Tensor(x, self.bias)

    def get_example_inputs(self):
        return (torch.randn(2, 4),), {}


def get_placeholder_names(exported_program):
    return [
        node.name for node in exported_program.graph.nodes if node.op == "placeholder"
    ]


def make_bias_placeholder_unused(exported_program, target, arg_index, replacement):
    graph = exported_program.graph

    bias_node = None
    target_node = None

    for node in graph.nodes:
        if node.op == "placeholder" and "bias" in node.name:
            bias_node = node

        if node.op == "call_function" and node.target == target:
            target_node = node

    assert bias_node is not None
    assert target_node is not None

    target_node.update_arg(arg_index, replacement)
    graph.eliminate_dead_code()
    graph.lint()
    exported_program.graph_module.recompile()

    assert len(bias_node.users) == 0

    return bias_node.name


class RemoveUnusedPlaceholderTest(SinglePassValueTest):
    def test_remove_unused_placeholder(self):
        self.setup(UsedBiasNet())

        bias_placeholder_name = make_bias_placeholder_unused(
            self.exported_program(), torch.ops.aten.linear.default, 2, None
        )
        graph_signature = self.exported_program().graph_signature

        # Simulate QuantizeBias behavior.
        assert graph_signature.inputs_to_parameters[bias_placeholder_name] == "bias"
        assert "bias" in self.exported_program().state_dict

        self.run_value_test(RemoveUnusedPlaceholder())

        placeholder_names_after = get_placeholder_names(self.exported_program())
        assert bias_placeholder_name not in placeholder_names_after
        assert bias_placeholder_name not in graph_signature.inputs_to_parameters
        assert "bias" not in self.exported_program().state_dict

    def test_remove_unused_buffer_placeholder(self):
        self.setup(UsedBufferNet())

        bias_placeholder_name = make_bias_placeholder_unused(
            self.exported_program(), torch.ops.aten.add.Tensor, 1, 0
        )
        graph_signature = self.exported_program().graph_signature

        assert graph_signature.inputs_to_buffers[bias_placeholder_name] == "bias"
        assert "bias" in self.exported_program().state_dict

        self.run_value_test(RemoveUnusedPlaceholder())

        placeholder_names_after = get_placeholder_names(self.exported_program())
        assert bias_placeholder_name not in placeholder_names_after
        assert bias_placeholder_name not in graph_signature.inputs_to_buffers
        assert "bias" not in self.exported_program().state_dict

    def test_remove_unused_lifted_tensor_constant_placeholder(self):
        self.setup(UsedLiftedTensorConstantNet())

        bias_placeholder_name = make_bias_placeholder_unused(
            self.exported_program(), torch.ops.aten.add.Tensor, 1, 0
        )
        graph_signature = self.exported_program().graph_signature

        assert (
            graph_signature.inputs_to_lifted_tensor_constants[bias_placeholder_name]
            == "bias"
        )
        assert "bias" in self.exported_program().constants

        # Avoid torch.export._unlift warnings for intentionally lifted constants.
        self.run_pass(RemoveUnusedPlaceholder())

        placeholder_names_after = get_placeholder_names(self.exported_program())
        assert bias_placeholder_name not in placeholder_names_after
        assert (
            bias_placeholder_name
            not in graph_signature.inputs_to_lifted_tensor_constants
        )
        assert "bias" not in self.exported_program().constants

    def test_keep_used_placeholder(self):
        self.setup(UsedBiasNet())

        placeholder_names_before = get_placeholder_names(self.exported_program())
        assert any("bias" in name for name in placeholder_names_before)

        self.run_value_test(RemoveUnusedPlaceholder())

        placeholder_names_after = get_placeholder_names(self.exported_program())
        assert any("bias" in name for name in placeholder_names_after)
