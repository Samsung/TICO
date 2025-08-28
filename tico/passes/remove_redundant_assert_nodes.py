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
from torch.export import ExportedProgram

from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node


assert_node_targets = [
    torch.ops.aten._assert_scalar.default,
    torch.ops.aten._assert_tensor_metadata.default,
    torch.ops.aten.sym_constrain_range_for_size.default, # Related to symbolic shape validation
]


@trace_graph_diff_on_pass
class RemoveRedundantAssertionNodes(PassBase):
    """
    This removes redundant assertion nodes.
    When assertion node is erased, related comparison nodes are also removed by DCE pass.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if is_target_node(node, assert_node_targets):
                graph.erase_node(node)
                modified = True

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
