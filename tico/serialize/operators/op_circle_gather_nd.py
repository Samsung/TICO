# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index


@register_node_visitor
class CircleGatherNdVisitor(NodeVisitor):
    """Serialize the internal GatherNd custom operator to Circle GATHER_ND."""

    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.gather_nd,
    ]

    def define_node(self, node: torch.fx.Node) -> circle.Operator.OperatorT:
        """Create a Circle GATHER_ND operator for a GatherNd custom node."""
        params, indices = node.args

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.GATHER_ND,
            self._op_codes,
        )

        inputs = [params, indices]
        outputs = [node]
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.GatherNdOptions
        )
        operator.builtinOptions = circle.GatherNdOptions.GatherNdOptionsT()

        return operator
