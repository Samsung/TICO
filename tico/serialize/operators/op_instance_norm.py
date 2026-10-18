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

from tico.serialize.circle_mapping import extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
    TorchOperator,
)
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import InstanceNormArgs


@register_node_visitor
class InstanceNormVisitor(NodeVisitor):
    """
    Input [NHWC] ---- circle_cumstom.instance_norm [NHWC] ---- OUTPUT[NHWC]
    Weight   -------/
    Bias    -------/
    """

    target: List[TorchOperator] = [
        torch.ops.circle_custom.instance_norm,
    ]

    def define_instance_norm_node(
        self, eps, inputs, outputs
    ) -> circle.Operator.OperatorT:
        def set_option(operator, eps):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.InstanceNormOptions
            )
            option = circle.InstanceNormOptions.InstanceNormOptionsT()
            option.epsilon = eps
            option.fusedActivationFunction = (
                circle.ActivationFunctionType.ActivationFunctionType.NONE
            )
            operator.builtinOptions = option

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.INSTANCE_NORM, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        set_option(operator, eps)
        return operator

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_node(self, node: torch.fx.Node) -> circle.Operator.OperatorT:
        args = InstanceNormArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        input = args.input
        weight = args.weight
        bias = args.bias
        eps = args.eps

        # Ignore training-related args
        input_shape = list(extract_shape(input))
        assert len(input_shape) == 4, len(input_shape)

        instance_norm_operator = self.define_instance_norm_node(
            eps,
            [input, weight, bias],
            [node],
        )

        return instance_norm_operator
