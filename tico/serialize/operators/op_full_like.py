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

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.utils.validate_args_kwargs import FullLikeArgs


@register_node_visitor
class FullLikeVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.full_like.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = FullLikeArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        fill_value = args.fill_value
        dtype = args.dtype if args.dtype is not None else input.meta["val"].dtype

        # can calculate full_like tensor before inference.
        output_data = torch.full(
            size=input.meta["val"].size(),
            fill_value=fill_value,
            dtype=dtype,
        )

        self.graph.update_tensor_buffer(output_data, node.name)

        return None  # type: ignore[return-value]
