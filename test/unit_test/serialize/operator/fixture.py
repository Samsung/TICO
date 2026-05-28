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

import torch.fx
from circle_schema import circle

from tico.serialize.circle_graph import CircleModel, CircleSubgraph
from tico.serialize.operators.node_visitor import get_node_visitor
from torch.export import export


class SingleOpGraphFixture:
    def __init__(
        self, torch_module: torch.nn.Module, torch_target: torch._ops.OpOverload
    ):
        self.torch_target = torch_target

        self._circle_model = CircleModel()
        self._circle_model.add_buffer(circle.Buffer.BufferT())
        self.circle_graph = CircleSubgraph(self._circle_model)

        self.forward_args, self.forward_kwargs = torch_module.get_example_inputs()  # type: ignore[operator]
        self.exported_program = export(
            torch_module.eval(), self.forward_args, self.forward_kwargs
        )

        for node in self.exported_program.graph.nodes:
            if node.op == "placeholder":
                self.circle_graph.add_tensor_from_node(node)
            if node.op == "output":
                continue
            if node.op == "call_function":
                self.circle_graph.add_tensor_from_node(node)

    def target_node(self):
        """
        Get a call_function node.

        ASSUMPTION: The graph has only one call_function node.
        """

        op_nodes = [
            node
            for node in self.exported_program.graph.nodes
            if node.op == "call_function"
        ]
        assert len(op_nodes) == 1, "FIX CALLER UNLESS"

        return op_nodes[0]

    def target_visitor(self):
        return get_node_visitor(self.torch_target)(op_codes={}, graph=self.circle_graph)
