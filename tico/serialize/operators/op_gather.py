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
from tico.utils.validate_args_kwargs import GatherArgs


@register_node_visitor
class GatherVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.gather.default,
        torch.ops.aten.gather.out,
    ]

    def define_node(self, node: torch.fx.Node) -> circle.Operator.OperatorT:
        """
        Convert PyTorch gather to Circle GATHER_ND.

        PyTorch: out[i,j,k] = input[i,j,indices[i,j,k],k] for dim=2

        We construct indices that specify full coordinates:
        - For each output position, create coordinates [i0, i1, ..., index_value, ..., iN]
        - The index tensor provides the gather dimension values
        - Other dimensions use arange-based coordinate grids

        Note: The ConvertGatherToGatherNdPass preprocesses the graph to construct
        multi-dimensional indices. This serializer then uses those pre-constructed
        indices.
        """
        args = GatherArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        input_tensor = args.input
        index_tensor = args.index

        # Verify that the pass has already transformed the graph to provide
        # multi-dimensional indices for GATHER_ND.
        if not node.meta.get("gather-to-gather-nd-passed", False):
            raise ValueError(
                f"GatherVisitor expected node {node} to have been preprocessed by "
                f"ConvertGatherToGatherNdPass, but it was not. Ensure that the pass "
                f"is enabled and correctly modifies the graph before serialization."
            )

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.GATHER_ND,
            self._op_codes,
        )

        inputs = [input_tensor, index_tensor]
        outputs = [node]

        operator = create_builtin_operator(
            self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.GatherNdOptions
        operator.builtinOptions = circle.GatherNdOptions.GatherNdOptionsT()

        return operator
