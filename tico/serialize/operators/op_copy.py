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

from typing import Dict, List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import CopyArgs


@register_node_visitor
class CopyVisitor(NodeVisitor):
    """
    NOTE `torch.Tensor.copy_`'s behavior matches with `Reshape` of CIRCLE.
    - because `torch.Tensor.copy_` is a in-place operator, so `dst` is converted to `Shape` of CIRCLE.
    - after that, `dst` converted to `Shape` is connected to shape of `Reshape`.
    - `src` is connected to tensor of `Reshape`.
    - if `dst` is not converted to `Shape`.
      [dst]      [src]
                   |
                [Reshape]
    - if `dst` is converted to `Shape`.
      [dst]      [src]
        |          |
      [Shape]      |
        \         /
         [Reshape]
    """

    target: List[torch._ops.OpOverload] = [torch.ops.aten.copy.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def check_to_do_broadcast(self, dst: List[int], src: List[int]) -> bool:
        return dst != src

    def define_broadcast_to_node(
        self,
        inputs: List[Union[circle.Tensor.TensorT, torch.Tensor]],
        outputs: List[circle.Tensor.TensorT],
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.BROADCAST_TO, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.BroadcastToOptions
        )

        option = circle.BroadcastToOptions.BroadcastToOptionsT()
        operator.builtinOptions = option
        return operator

    def define_shape_node(
        self, inputs: List[torch.fx.Node], outputs: List[circle.Tensor.TensorT]
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SHAPE, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.ShapeOptions

        option = circle.ShapeOptions.ShapeOptionsT()
        option.outType = circle.TensorType.TensorType.INT32
        operator.builtinOptions = option
        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        if len(node.args) == 3:
            raise NotYetSupportedError("'non_blocking' is not supported yet.")

        assert len(node.args) == 2, len(node.args)

        args = CopyArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        dst = args.dst
        src = args.src

        # To connect 'dst' to Reshape node in the graph, 'dst' must be converted to Shape op.
        dst_tensor: circle.Tensor.TensorT = self.graph.get_tensor(dst)
        dst_shape: List[int] = dst_tensor.shape
        dst_shape_tensor = torch.as_tensor(dst_shape, dtype=torch.int32)

        dst_shape_shape = [len(dst_shape)]
        dst_name: str = dst.name

        shape_output = self.graph.add_tensor_from_scratch(
            prefix=f"{dst_name}_shape_output",
            shape=dst_shape_shape,
            dtype=circle.TensorType.TensorType.INT32,
            source_node=node,
        )

        shape_operator = self.define_shape_node([dst], [shape_output])
        self.graph.add_operator(shape_operator)

        src_tensor: circle.Tensor.TensorT = self.graph.get_tensor(src)
        src_shape: List[int] = src_tensor.shape

        # The src tensor must be broadcastable with the dst tensor.
        do_broadcast = self.check_to_do_broadcast(dst_shape, src_shape)
        if do_broadcast:
            # create braodcastTo output tensor
            src_name: str = src.name
            src_type: int = src_tensor.type

            broadcast_to_output: circle.Tensor.TensorT = (
                self.graph.add_tensor_from_scratch(
                    prefix=f"{src_name}_broadcast_to_output",
                    shape=dst_shape,
                    dtype=src_type,
                    source_node=node,
                )
            )

            broadcast_to_operator: circle.Operator.OperatorT = (
                self.define_broadcast_to_node(
                    [src_tensor, dst_shape_tensor], [broadcast_to_output]
                )
            )
            self.graph.add_operator(broadcast_to_operator)
            inputs: List = [broadcast_to_output, shape_output]
        else:
            inputs = [src, shape_output]

        outputs = [node]
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.RESHAPE, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.ReshapeOptions
        )
        option = circle.ReshapeOptions.ReshapeOptionsT()
        option.newShape = dst_shape

        operator.builtinOptions = option
        return operator
