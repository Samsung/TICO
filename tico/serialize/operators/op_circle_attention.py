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

from torch.library import Library

from tico.serialize.circle_graph import CircleSubgraph, extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index

lib = Library("circle", "DEF")
lib.define(
    """
attention.llama(
    Tensor hidden_states,
    Tensor wq,
    Tensor wk,
    Tensor wv,
    Tensor wo,
    Tensor position_cos,
    Tensor position_sin,
    Tensor? attention_mask,
    Tensor past_key,
    Tensor past_value,
    int layer_idx,
    Tensor cache_position
) -> Tensor
"""
)


@register_node_visitor
class AttentionVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.circle.attention.llama,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        (
            hidden_states,
            wq,
            wk,
            wv,
            wo,
            position_cos,
            position_sin,
            attention_mask,
            past_key,
            past_value,
            cache_position,
            layer_idx,
        ) = node.args

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.ATTENTION, self._op_codes
        )

        inputs = node.args
        outputs = [node]
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.AttentionOptions
        )
        option = circle.AttentionOptions.AttentionOptionsT()
        option.layer_idx = layer_idx

        operator.builtinOptions = option

        return operator
