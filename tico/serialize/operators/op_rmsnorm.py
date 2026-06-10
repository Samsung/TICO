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
from tico.serialize.circle_mapping import extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import CircleRMSNormArgs, RMSNormArgs


@register_node_visitor
class RMSNormVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.circle_custom.rms_norm.default,
        torch.ops.aten.rms_norm.default,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def _parse_args(self, node):
        if node.target == torch.ops.aten.rms_norm.default:
            args = RMSNormArgs(*node.args, **node.kwargs)

            if args.weight is None:
                raise NotYetSupportedError("RMSNorm without weight is not supported")

            if len(args.normalized_shape) != 1:
                raise NotYetSupportedError(
                    "Only 1-D normalized_shape RMSNorm is supported"
                )

            if list(extract_shape(args.weight)) != list(args.normalized_shape):
                raise NotYetSupportedError(
                    "RMSNorm weight shape should match normalized_shape"
                )

            eps = args.eps
            if eps is None:
                raise NotYetSupportedError("RMSNorm eps=None is not supported yet")

            return args.input, args.weight, eps

        circle_args = CircleRMSNormArgs(*node.args, **node.kwargs)
        return circle_args.input, circle_args.weight, circle_args.eps

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        input, weight, eps = self._parse_args(node)

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.RMS_NORM, self._op_codes
        )

        inputs = [input, weight]
        outputs = [node]
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.RmsNormOptions
        )
        option = circle.RmsNormOptions.RmsNormOptionsT()
        option.epsilon = eps

        operator.builtinOptions = option

        return operator
