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

from tico.serialize.circle_mapping import circle_legalize_dtype_to
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import ExpandArgs

def is_dynamic(size: List[int] | torch.fx.immutable_collections.immutable_list):
    for s in size:
        if isinstance(s, int):
            continue
        if isinstance(s, torch.fx.Node):
            if s.target == torch.ops.aten.sym_size.int:
                return True
        
        raise RuntimeError(f"Invalid type for size: {type(s)}")

    return False

@register_node_visitor
class ExpandVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.expand.default,
        torch.ops.aten.expand_copy.default,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_expand_copy_node(self, inputs, outputs) -> circle.Operator.OperatorT:
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
    
    def define_pack_node(
        self, inputs: List[torch.fx.Node], outputs: List[circle.Tensor.TensorT]
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.PACK, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.PackOptions

        option = circle.PackOptions.PackOptionsT()
        option.axis = 0
        option.values_count = len(inputs)
        operator.builtinOptions = option
        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = ExpandArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        size = args.size

        dynamic = is_dynamic(size)
        if dynamic:
            # pack
            pack_shape = [len(size)]
            pack_tensor = self.graph.add_tensor_from_scratch(
                prefix=f"{node.name}_pack_shape",
                shape=pack_shape,
                dtype=circle.TensorType.TensorType.INT32,
            )
            pack_operator = self.define_pack_node(size, [pack_tensor])
            self.graph.add_operator(pack_operator)
            # broadcast_to
            inputs = [input, pack_tensor]
            outputs = [node]
            operator = self.define_expand_copy_node(inputs, outputs)

            return operator
        else:
            input_tid: int = self.graph.get_tid_registered(input)
            input_tensor: circle.Tensor.TensorT = self.graph.tensors[input_tid]
            input_shape: List[int] = input_tensor.shape

            extending_rank = len(size) - len(input_shape)

            size_i32 = circle_legalize_dtype_to(size, dtype=torch.int32)
            for idx, dim in enumerate(size_i32):
                if idx < extending_rank:
                    assert (
                        dim >= 1
                    ), "A dim value(less than 1) isn't allowed in the extending_rank."

                """
                In pytorch, passing -1 as the size for a dimension means that the size of that dimension won't be changed.
                But, circle in ONE does not support this. 
                So, dim value(-1) in the non-extending_rank is supported to convert to the size for the dimension.
                """
                if dim == -1:
                    size_i32[idx] = input_shape[idx - extending_rank]

            for idx, dim in enumerate(input_shape):
                assert (
                    dim == 1 or dim == size_i32[extending_rank + idx]
                ), f"The size of dimension to be expanded ({dim}) must be 1 or the expanded size ({size_i32[extending_rank + idx]})."

            inputs = [input, size_i32]
            outputs = [node]
            operator = self.define_expand_copy_node(inputs, outputs)

            return operator
