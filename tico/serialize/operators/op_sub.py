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
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import SubTensorArgs


@register_node_visitor
class SubVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.sub.Tensor]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SUB, self._op_codes
        )

        args = SubTensorArgs(*node.args, **node.kwargs)  # type: ignore[arg-type, call-arg]

        input = args.input
        other = args.other
        alpha = args.alpha

        if alpha not in (None, 1):
            raise NotYetSupportedError(
                f"{node.name} ({node.target}): operand other has alpha={alpha}; "
                f"source={getattr(other, 'meta', {}).get(QPARAM_KEY)}, "
                "required=Circle SUB(alpha=1); no native alpha operand. Preserve an explicit multiply "
                "with its own quantization boundary before exporting."
            )

        inputs = [input, other]
        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.SubOptions
        option = circle.SubOptions.SubOptionsT()
        option.fusedActivationFunction = (
            circle.ActivationFunctionType.ActivationFunctionType.NONE
        )
        option.potScaleInt16 = False
        operator.builtinOptions = option

        return operator
