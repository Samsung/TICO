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
from tico.serialize.circle_mapping import extract_circle_dtype, extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import InvalidArgumentError, NotYetSupportedError
from tico.utils.validate_args_kwargs import RepeatArgs


@register_node_visitor
class RepeatVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.repeat.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = RepeatArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        repeats = args.repeats

        for r in repeats:
            if r == 0:
                # TODO: Support r == 0 case
                raise NotYetSupportedError("Only support positive repeat value")
            elif r < 0:
                raise InvalidArgumentError("Only support positive repeat value")

        tensor_shape = extract_shape(input)
        assert len(tensor_shape) <= len(repeats)
        if len(tensor_shape) != len(repeats):
            # TODO Support len(tensor_shape) < len(repeats)
            raise NotYetSupportedError(
                "Length of both input tensor and repeats vector should be same."
            )
        repeat_dim_cnt = len(repeats) - repeats.count(1)
        tensor_dtype = extract_circle_dtype(input)
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.CONCATENATION, self._op_codes
        )
        concat_input: torch.fx.Node | circle.Tensor.TensorT = input
        concat_output: torch.fx.node.Node | circle.Tensor.TensorT = node
        for idx, r in enumerate(repeats):
            # concat along idx dimension
            if r > 1:
                # Except last created concat, a tensor should be created.
                if repeat_dim_cnt > 1:
                    repeated_shape = list(tensor_shape)
                    repeated_shape[idx] = repeated_shape[idx] * r
                    concat_output = self.graph.add_tensor_from_scratch(
                        prefix=f"{node.name}_concat_{idx}",
                        shape=repeated_shape,
                        dtype=tensor_dtype,
                        source_node=node,
                    )
                inputs = [concat_input] * r
                if repeat_dim_cnt == 1:
                    outputs: List[torch.fx.node.Node | circle.Tensor.TensorT] = [node]
                else:
                    outputs = [concat_output]
                operator = create_builtin_operator(
                    self.graph, op_index, inputs, outputs
                )
                operator.builtinOptionsType = (
                    circle.BuiltinOptions.BuiltinOptions.ConcatenationOptions
                )
                option = circle.ConcatenationOptions.ConcatenationOptionsT()
                option.axis = idx
                operator.builtinOptions = option
                if repeat_dim_cnt > 1:
                    self.graph.add_operator(operator)
                    concat_input = concat_output
                repeat_dim_cnt -= 1

        return operator
