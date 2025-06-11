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
from tico.utils.validate_args_kwargs import WhereSelfArgs


@register_node_visitor
class WhereVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.where.self]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SELECT_V2,
            self._op_codes,
        )

        args = WhereSelfArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        condition = args.condition
        input = args.input
        other = args.other

        result_true_dtype = (
            extract_torch_dtype(input)
            if isinstance(input, torch.fx.Node)
            else input.dtype  # type: ignore[union-attr]
        )
        result_false_dtype = (
            extract_torch_dtype(other)
            if isinstance(other, torch.fx.Node)
            else other.dtype  # type: ignore[union-attr]
        )

        if result_true_dtype != result_false_dtype:
            raise RuntimeError(
                f"Data type of arguments are not matched. result_true: {result_true_dtype}, result_false: {result_false_dtype}"
            )

        inputs = [condition, input, other]
        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.SelectV2Options
        )
        option = circle.SelectV2Options.SelectV2OptionsT()

        operator.builtinOptions = option

        return operator
