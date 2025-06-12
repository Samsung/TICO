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
from tico.serialize.circle_mapping import circle_legalize_dtype_to
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import ReshapeArgs


@register_node_visitor
class ReshapeVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.reshape.default,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.RESHAPE,
            self._op_codes,
        )
        args = ReshapeArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        size = args.shape

        if isinstance(size, int):
            raise NotYetSupportedError("scalar size conversion is not supported yet.")

        assert is_const(size), type(size)

        size_i32 = circle_legalize_dtype_to(size, dtype=torch.int32)
        inputs = [input, size_i32]
        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.ReshapeOptions
        )
        option = circle.ReshapeOptions.ReshapeOptionsT()
        option.newShape = size_i32

        operator.builtinOptions = option

        return operator
