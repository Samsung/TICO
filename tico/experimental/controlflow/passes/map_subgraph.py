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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx

import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import get_quant_dtype
from tico.utils.validate_args_kwargs import CondArgs
from tico.utils.graph import create_node
from tico.utils.subgraph import get_gm_map
import operator

@trace_graph_diff_on_pass
class MapSubgraph(PassBase):
    """
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram, _) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node.target
                != torch.ops.higher_order.cond
            ):
                continue
            
            cond_args = CondArgs(*node.args, **node.kwargs)
            
            true_graph_idx = None
            false_graph_idx = None
            for gm_info in get_gm_map(exported_program):
                if gm_info["name"] == cond_args.true_graph.name:
                    true_graph_idx = gm_info["index"]
                    continue
                if gm_info["name"] == cond_args.false_graph.name:
                    false_graph_idx = gm_info["index"]
                    continue
            assert true_graph_idx is not None
            assert false_graph_idx is not None
            
            with graph.inserting_before(node):
                circle_if = create_node(
                    graph,
                    torch.ops.circle_custom.if_,
                    args=(cond_args.condition, true_graph_idx, false_graph_idx, cond_args.cond_args),
                    kwargs={},
                    origin=node,
                )
            
            # FIX ME UNLESS torch.ops.higher_order.cond generates this pattern
            assert len(node.users) == 1
            getitem_node = list(node.users.items())[0][0]
            assert getitem_node.target == operator.getitem
            getitem_node.replace_all_uses_with(circle_if)
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
