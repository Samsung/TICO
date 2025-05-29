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
from tico.utils.validate_args_kwargs import PadArgs


@register_node_visitor
class PadVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.pad.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = PadArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        pad = args.pad

        breakpoint()
        w_before = pad[0]
        w_after = pad[1]
        h_before = pad[2]
        h_after = pad[3]

        # TensorFlow NHWC 형식으로 변환
        tf_pad = [
            0,
            0,  # 배치 차원 (N)
            h_before,
            h_after,  # 높이 차원 (H)
            w_before,
            w_after,  # 너비 차원 (W)
            0,
            0,  # 채널 차원 (C)
        ]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.PAD,
            self._op_codes,
        )

        inputs = [input, tf_pad]
        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.PadOptions
        option = circle.PadOptions.PadOptionsT()

        operator.builtinOptions = option

        return operator
