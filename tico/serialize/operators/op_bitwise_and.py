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
from tico.serialize.circle_mapping import extract_torch_dtype
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import BitwiseAndArgs


@register_node_visitor
class BitwiseAndVisitor(NodeVisitor):
    """
    Lower `aten.bitwise_and.Tensor` on boolean tensors to Circle LOGICAL_AND.

    For boolean operands `bitwise_and` is identical to `logical_and`, which is
    what `tensor_a & tensor_b` produces (e.g. the attention-mask construction in
    recent `transformers` releases). Integer bitwise AND has no Circle operator
    and is rejected.
    """

    target: List[torch._ops.OpOverload] = [torch.ops.aten.bitwise_and.Tensor]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = BitwiseAndArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        other = args.other

        for operand in (input, other):
            operand_dtype = extract_torch_dtype(operand)
            if operand_dtype != torch.bool:
                raise NotYetSupportedError(
                    "aten.bitwise_and.Tensor is only supported for bool operands "
                    f"(got {operand_dtype})"
                )

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.LOGICAL_AND,
            self._op_codes,
        )

        inputs = [input, other]
        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.LogicalAndOptions
        )
        option = circle.LogicalAndOptions.LogicalAndOptionsT()

        operator.builtinOptions = option

        return operator
