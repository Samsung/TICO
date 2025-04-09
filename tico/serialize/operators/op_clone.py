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
from tico.utils.validate_args_kwargs import CloneArgs


@register_node_visitor
class CloneVisitor(NodeVisitor):
    """
    Clone tensor
    TODO: Support dim_order and memory_format
    Tranpose may be required if 'memory_format' differs from input tensor's 'memory_format'
    """

    target: List[torch._ops.OpOverload] = [torch.ops.aten.clone.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        if "memory_format" in node.kwargs:
            # TODO: Support dim_order and memory_format
            pass

        args = CloneArgs(*node.args, **node.kwargs)
        input = args.input

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.TRANSPOSE, self._op_codes
        )

        permute = torch.IntTensor(list(range(len(input.meta["val"].shape))))

        inputs = [input, permute]
        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.TransposeOptions
        )
        option = circle.TransposeOptions.TransposeOptionsT()
        operator.builtinOptions = option

        return operator
