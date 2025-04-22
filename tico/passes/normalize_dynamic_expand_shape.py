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

import operator
from typing import TYPE_CHECKING

from tico.utils.validate_args_kwargs import ExpandArgs

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.passes import ops
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass


@trace_graph_diff_on_pass
class NormalizeDynamicExpandShape(PassBase):
    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            if node.target not in ops.aten.expand:
                continue

            args = ExpandArgs(*node.args)
            tensor = args.input
            expand_size = args.size

            new_shape_nodes = []
            for idx, s in enumerate(expand_size):
                if isinstance(s, torch.fx.Node):
                    if s.target == torch.ops.aten.sym_size.int:
                        with graph.inserting_before(node):
                            unsqueezed = graph.call_function(torch.ops.aten.unsqueeze.default, args=(s, 0))
                        new_shape_nodes.append(unsqueezed)
                    else:
                        raise ValueError(f"Unsupported dynamic shape expression: {s.target}")
                elif isinstance(s, int):
                    # wrap into scalar tensor
                    with graph.inserting_before(node):
                        const_tensor = graph.call_function(torch.ops.aten.full.default, args=([], s), kwargs={"dtype": torch.int64})
                        unsqueezed = graph.call_function(torch.ops.aten.unsqueeze.default, args=(const_tensor, 0))
                    new_shape_nodes.append(unsqueezed)
                else:
                    raise ValueError(f"Unsupported shape element type: {type(s)}")
            
            # Concatenate all dims into shape tensor
            with graph.inserting_before(node):
                shape_tensor = graph.call_function(torch.ops.aten.cat.default, args=(new_shape_nodes, 0))
            node.update_arg(1, shape_tensor)

            modified = True

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
