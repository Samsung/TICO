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

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass


def _is_matrix_transpose(node: object) -> bool:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    if node.target not in (torch.ops.aten.permute.default, torch.ops.aten.t.default):
        return False

    weight = node.args[0]
    if not isinstance(weight, torch.fx.Node):
        return False
    value = weight.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dim() != 2:
        return False

    if node.target == torch.ops.aten.t.default:
        return True
    dims = node.args[1]
    return (
        isinstance(dims, (tuple, list))
        and len(dims) == 2
        and dims[0] in (1, -1)
        and dims[1] in (0, -2)
    )


@trace_graph_diff_on_pass
class RestoreLinear(PassBase):
    """
    Linear Op is decomposed to multiple Ops in core aten
    This pass restores linear Ops. For example,

    The RHS must transpose a rank-two weight using t or permute. Addmm
    is restored only when alpha and beta are both one; other coefficients
    are handled by the existing DecomposeAddmm pass.

    Before)

        bias   input   weight         input    weight
        |        |       |              |        |
        |        |   permute/t          |    permute/t
        |        V       |              |        |
        +----> addmm <---+              |        V
                                        +------> mm

    After)

        input  weight   bias          input    weight
        |        |       |              |        |
        |        |       |              |        |
        |        V       |              |        V
        +---> linear <---+              +----> linear
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            if node.target == torch.ops.aten.addmm.default:
                assert len(node.args) == 3
                bias, input, permute = node.args
                if node.kwargs.get("alpha", 1) != 1 or node.kwargs.get("beta", 1) != 1:
                    continue
                if not _is_matrix_transpose(permute):
                    continue
                weight = permute.args[0]

                addmm_args = (input, weight, bias)
                with graph.inserting_after(node):
                    linear_node = create_node(
                        graph,
                        torch.ops.aten.linear.default,
                        args=addmm_args,
                    )
                    node.replace_all_uses_with(linear_node, propagate_meta=True)

            elif node.target == torch.ops.aten.mm.default:
                assert len(node.args) == 2
                input, permute = node.args
                if not _is_matrix_transpose(permute):
                    continue
                weight = permute.args[0]

                mm_args = (input, weight)
                with graph.inserting_after(node):
                    linear_node = create_node(
                        graph, torch.ops.aten.linear.default, args=mm_args
                    )
                    node.replace_all_uses_with(linear_node, propagate_meta=True)

            else:
                continue

            modified = True
            logger.debug(f"{node.name} is replaced with linear")

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(modified)
