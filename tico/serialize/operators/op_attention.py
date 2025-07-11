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

from tico.serialize.circle_graph import CircleSubgraph, extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index


@register_node_visitor
class AttentionVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.llama_attention_with_kvcache,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        (
            q_w,
            k_w,
            v_w,
            o_w,
            hidden_states,
            pos_emb0,
            pos_emb1,
            attn_mask,
            key_cache,
            val_cache,
        ) = node.args

        inputs = node.args
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.LLAMA_ATTENTION, self._op_codes
        )

        inputs = node.args
        outputs = [i for i in node.users.keys()]
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        assert isinstance(pos_emb0, torch.fx.Node), "pos_emb0 should be a node"
        assert isinstance(pos_emb1, torch.fx.Node), "pos_emb1 should be a node"
        assert extract_shape(pos_emb0) == extract_shape(pos_emb1)

        # Calculate scaling from head_dim
        head_dim = extract_shape(pos_emb0)[-1]
        scaling = head_dim**-0.5

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.LlamaAttentionOptions
        )
        option = circle.LlamaAttentionOptions.LlamaAttentionOptionsT()

        option.scaling = scaling
        operator.builtinOptions = option

        return operator
