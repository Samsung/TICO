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

from tico.serialize.circle_graph import CircleSubgraph, is_const
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import MatmulArgs


@register_node_visitor
class MatmulDefaultVisitor(NodeVisitor):
    """
    Convert matmul to equavalent BatchMatMul
    """

    target: List[torch._ops.OpOverload] = [torch.ops.aten.mm.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    # NOTE: Matmul is equivalent to Batch MatMul (batch=1)
    def define_bmm_node(self, inputs, outputs) -> circle.Operator.OperatorT:
        def set_bmm_option(operator):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.BatchMatMulOptions
            )
            option = circle.BatchMatMulOptions.BatchMatMulOptionsT()
            option.adjointLhs, option.adjointRhs = False, False
            option.asymmetricQuantizeInputs = False
            operator.builtinOptions = option

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.BATCH_MATMUL, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        set_bmm_option(operator)

        return operator

    def define_node(
        self, node: torch.fx.Node, prior_latency=True
    ) -> circle.Operator.OperatorT:
        args = MatmulArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        other = args.other

        inputs = [input, other]
        outputs = [node]

        operator = self.define_bmm_node(inputs, outputs)

        return operator
