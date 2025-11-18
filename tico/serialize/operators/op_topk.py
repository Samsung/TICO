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
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.circle_mapping import (
    circle_legalize_dtype_to,
    extract_circle_shape,
    extract_shape,
    extract_torch_dtype,
)
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import TopKArgs


@register_node_visitor
class TopkVisitor(NodeVisitor):
    """ """

    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.top_k,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_topk_node(
        self, inputs: List, outputs: List
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.TOPK_V2, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.TopKV2Options
        option = circle.TopKV2Options.TopKV2OptionsT()
        operator.builtinOptions = option

        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = TopKArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        k = args.k

        input_shape = extract_circle_shape(input)
        k_i32 = circle_legalize_dtype_to(k, dtype=torch.int32)
        assert args.dim == -1 or args.dim == len(input_shape) - 1

        inputs = [input, k_i32]

        outputs = [i for i in node.users.keys()]

        topk_node: circle.Operator.OperatorT = self.define_topk_node(inputs, outputs)

        return topk_node
