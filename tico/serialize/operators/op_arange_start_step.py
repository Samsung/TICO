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

from tico.serialize.operators.utils import create_builtin_operator, get_op_index

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.utils.validate_args_kwargs import ArangeStartStepArgs


@register_node_visitor
class ArangeStartStepVisitor(NodeVisitor):
    """
    Fuse arange_start_step to const_tensor
    """

    target: List[torch._ops.OpOverload] = [torch.ops.aten.arange.start_step]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = ArangeStartStepArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        start = args.start
        end = args.end
        step = args.step
        
        inputs = [start, end, step]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.RANGE, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.RangeOptions
        option = circle.RangeOptions.RangeOptionsT()
        operator.builtinOptions = option

        return operator
