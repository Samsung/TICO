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
from torch._subclasses.fake_tensor import FakeTensor

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.circle_mapping import circle_legalize_dtype_to, to_circle_dtype
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import SplitWithSizesArgs


@register_node_visitor
class SplitWithSizesVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.split_with_sizes.default,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SPLIT_V, self._op_codes
        )
        args = SplitWithSizesArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        split_sizes = args.split_sizes
        axis = args.dim

        split_sizes_i32 = [
            circle_legalize_dtype_to(split_size, dtype=torch.int32)
            for split_size in split_sizes
        ]
        axis_i32 = circle_legalize_dtype_to(axis, dtype=torch.int32)
        inputs = [input, split_sizes_i32, axis_i32]

        """
        `split_with_sizes` has multiple output tensors and they are represented as `getitem`.
        Therefore, unlike other ops, node itself doesn't become a circle tensor. Instead, each `getitem` will be
        a circle tensor.
        Further, torch module having `split_with_sizes` may somtimes return selected outputs. At that time, `getitem`
        nodes are generated only for the ouptut selected. Since one-compiler assumes that `CircleSplitV` always has 
        all the outputs, let's add unused output tensors to compensate this restriction.
        """
        outputs: List[Union[circle.Tensor.TensorT, torch.fx.node.Node]] = []
        sorted_users = sorted(node.users.keys(), key=lambda x: x.args[1])  # type: ignore[arg-type, return-value]
        users_indices = list(usrnode.args[1] for usrnode in sorted_users)
        user_it = iter(sorted_users)
        for idx, _ in enumerate(split_sizes):
            if idx in users_indices:
                user_node = next(user_it)
                outputs.append(user_node)
            else:
                # Let's add unused output tensor to satisfy circle split_v operator scheme
                node_val = node.meta.get("val")
                assert isinstance(node_val, list)
                fake_tensor = node_val[idx]
                assert isinstance(fake_tensor, FakeTensor)
                shape = list(fake_tensor.size())
                dtype = to_circle_dtype(fake_tensor.dtype)
                tensor = self.graph.add_tensor_from_scratch(
                    f"{node.name}_unused_{idx}",
                    shape,
                    dtype,
                    source_node=node,
                )
                outputs.append(tensor)

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.SplitVOptions
        option = circle.SplitVOptions.SplitVOptionsT()
        option.numSplits = len(split_sizes)
        operator.builtinOptions = option

        return operator
