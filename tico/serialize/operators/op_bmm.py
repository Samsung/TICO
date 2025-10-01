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
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import BmmArgs


@register_node_visitor
class BatchMatmulVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.bmm.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_fc_node(self, inputs, outputs) -> circle.Operator.OperatorT:
        def set_fc_option(operator):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.FullyConnectedOptions
            )
            option = circle.FullyConnectedOptions.FullyConnectedOptionsT()

            option.fusedActivationFunction = (
                circle.ActivationFunctionType.ActivationFunctionType.NONE
            )
            option.weightsFormat = (
                circle.FullyConnectedOptionsWeightsFormat.FullyConnectedOptionsWeightsFormat.DEFAULT
            )
            option.keepNumDims = False
            option.asymmetricQuantizeInputs = False
            option.quantizedBiasType = circle.TensorType.TensorType.FLOAT32

            operator.builtinOptions = option

        fc_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED, self._op_codes
        )
        operator = create_builtin_operator(self.graph, fc_op_index, inputs, outputs)
        set_fc_option(operator)
        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = BmmArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        mat2 = args.mat2

        is_const_tensor = lambda n: (
            n.op == "get_attr"
            or (
                n.op == "placeholder"
                and isinstance(n.meta.get("val", None), torch.Tensor)
                and not n.meta["val"].requires_grad
            )
        )

        lhs, rhs = input, mat2
        is_const_lhs = is_const_tensor(lhs)

        if is_const_lhs:
            fc_index = get_op_index(
                circle.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED,
                self._op_codes,
            )

            rhs_tid = self.graph.get_tid_registered(rhs)
            rhs_tensor: circle.Tensor.TensorT = self.graph.tensors[rhs_tid]
            rhs_shape = list(rhs_tensor.shape)  # [..., batch, in_features]
            rhs_dtype = rhs_tensor.type

            # lhs : weight, shape = [..., out_features, in_features]
            lhs_tid = self.graph.get_tid_registered(lhs)
            lhs_tensor: circle.Tensor.TensorT = self.graph.tensors[lhs_tid]
            lhs_shape = list(lhs_tensor.shape)
            out_features = lhs_shape[-2]
            fc_out_shape = rhs_shape[:-1] + [out_features]
            fc_bias = self.graph.add_const_tensor(data=[0.0], source_node=node)
            fc_out = self.graph.add_tensor_from_scratch(
                prefix=f"{node.name}_fc_out",
                shape=fc_out_shape,
                shape_signature=fc_out_shape,
                dtype=rhs_dtype,
            )

            fc_inputs = [rhs, lhs, fc_bias]  # order: [input, weight]
            fc_outputs = [fc_out]
            fc_op = self.define_fc_node(fc_inputs, fc_outputs)
            self.graph.add_operator(fc_op)

            trs_index = get_op_index(
                circle.BuiltinOperator.BuiltinOperator.TRANSPOSE,
                self._op_codes,
            )

            perm = list(range(len(fc_out.shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            perm_tensor = self.graph.add_const_tensor(
                data=torch.tensor(perm, dtype=torch.int32), # to prevent int64
            )

            trs_inputs = [fc_out, perm_tensor]
            trs_outputs = [node]
            trs_op = create_builtin_operator(
                self.graph, trs_index, trs_inputs, trs_outputs
            )

            return trs_op

        bmm_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.BATCH_MATMUL,
            self._op_codes,
        )
        inputs = [lhs, rhs]
        outputs = [node]
        op = create_builtin_operator(self.graph, bmm_index, inputs, outputs)
        return op
