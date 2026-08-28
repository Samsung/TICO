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

import copy
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.passes import ops
from tico.serialize.circle_graph import CircleSubgraph

from tico.serialize.circle_mapping import extract_circle_dtype, extract_circle_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.validate_args_kwargs import ClampArgs


@register_node_visitor
class ClampVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = ops.aten.clamp

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_minimum_node(
        self,
        inputs: List[torch.fx.Node | circle.Tensor.TensorT | int | float],
        outputs: List[torch.fx.Node | circle.Tensor.TensorT],
    ) -> circle.Operator.OperatorT:

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.MINIMUM, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.MaximumMinimumOptions
        )
        option = circle.MaximumMinimumOptions.MaximumMinimumOptionsT()

        operator.builtinOptions = option
        return operator

    def define_maximum_node(
        self,
        inputs: List[torch.fx.Node | circle.Tensor.TensorT | int | float],
        outputs: List[torch.fx.Node | circle.Tensor.TensorT],
    ) -> circle.Operator.OperatorT:

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.MAXIMUM, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.MaximumMinimumOptions
        )
        option = circle.MaximumMinimumOptions.MaximumMinimumOptionsT()

        operator.builtinOptions = option

        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        min_val = args.min
        max_val = args.max

        if min_val is None and max_val is None:
            raise ValueError("Both min and max cannot be None")

        elif min_val is not None and max_val is None:
            # min only
            return self.define_maximum_node([input, min_val], [node])

        elif min_val is None and max_val is not None:
            # max only
            return self.define_minimum_node([input, max_val], [node])

        elif min_val is not None and max_val is not None:
            input_shape, input_shape_signature = extract_circle_shape(input)
            input_dtype = extract_circle_dtype(input)
            qparam = (
                copy.deepcopy(node.meta[QPARAM_KEY])
                if QPARAM_KEY in node.meta
                else None
            )
            minimum_tensor = self.graph.add_tensor_from_scratch(
                prefix=f"{input.name}_min",
                dtype=input_dtype,
                shape=input_shape,
                shape_signature=input_shape_signature,
                qparam=qparam,
                source_node=node,
            )
            minimum_opertor = self.define_minimum_node(
                [input, max_val], [minimum_tensor]
            )
            self.graph.add_operator(minimum_opertor)

            maximum_operator = self.define_maximum_node(
                [minimum_tensor, min_val], [node]
            )
            return maximum_operator

        else:
            raise RuntimeError("Cannot reach here")
