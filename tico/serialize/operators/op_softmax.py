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

from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import SafeSoftmaxArgs, SoftmaxArgs


@register_node_visitor
class SoftMaxVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.aten._softmax.default,
        # NOTE: Let's treat _safe_softmax as normal _softmax as its usage is for training.
        # In order for optimization during inference, it can be replaced to softmax.
        # ref: https://github.com/pytorch/pytorch/pull/133882
        torch.ops.aten._safe_softmax.default,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_softmax_node(self, inputs, outputs) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SOFTMAX, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.SoftmaxOptions
        )
        option = circle.SoftmaxOptions.SoftmaxOptionsT()
        option.beta = 1.0
        operator.builtinOptions = option
        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        """
        Note that Currently, Softmax operator is supported only when `dim` is last dimension and `half_to_float` is False.
        """
        if node.target == torch.ops.aten._softmax.default:
            # aten._softmax
            args = SoftmaxArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            half_to_float: bool = args.half_to_float
            if half_to_float:
                raise NotYetSupportedError(
                    "softmax with half to float conversion is not supported on circle."
                )
        elif node.target == torch.ops.aten._safe_softmax.default:
            # aten._safe_softmax
            args = SafeSoftmaxArgs(*node.args, **node.kwargs)  # type: ignore[arg-type, assignment]

        input: torch.fx.Node = args.input
        dim: int = args.dim

        input_tid: int = self.graph.get_tid_registered(input)
        input_tensor: circle.Tensor.TensorT = self.graph.tensors[input_tid]
        input_shape: List[int] = input_tensor.shape

        if dim < 0:
            dim = dim % len(input_shape)

        if dim == len(input_shape) - 1:
            inputs = [input]
            outputs = [node]
            operator = self.define_softmax_node(inputs, outputs)
        else:
            raise NotYetSupportedError("softmax only supports last dimension for now.")

        return operator
