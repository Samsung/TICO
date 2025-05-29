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

from tico.serialize.circle_graph import CircleSubgraph, is_const
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import MatmulArgs


@register_node_visitor
class MatmulDefaultVisitor(NodeVisitor):
    """
    Convert matmul to equavalent BatchMatMul or FullyConnected with Transpose.
    """

    target: List[torch._ops.OpOverload] = [torch.ops.aten.mm.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    # NOTE: Matmul is equivalent to Batch MatMul (batch=1)
    def define_bmm_node(self, inputs, outputs) -> circle.Operator.OperatorT:
        def set_bmm_option(operator):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.BatchMatMulOptions
            )
            option = circle.BatchMatMulOptions.BatchMatMulOptionsT()
            option.adjointLhs, option.adjointRhs = False, False
            option.asymmetricQuantizeInputs = False
            operator.builtinOptions = option

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.BATCH_MATMUL, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)
        set_bmm_option(operator)

        return operator

    def define_transpose_node(self, inputs, outputs) -> circle.Operator.OperatorT:
        def set_transpose_option(operator):
            operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.TransposeOptions
            )
            option = circle.TransposeOptions.TransposeOptionsT()
            operator.builtinOptions = option

        transpose_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.TRANSPOSE, self._op_codes
        )
        operator = create_builtin_operator(
            self.graph, transpose_op_index, inputs, outputs
        )
        set_transpose_option(operator)
        return operator

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

    """
    Define FullyConnnected with Tranpose operator.
    Note that those sets of operators are equivalent.
    (1) Matmul
    matmul( lhs[H, K], rhs[K, W'] ) -> output(H, W')
    
    (2) Transpose + FullyConneccted
    transpose( rhs[K, W'] ) -> trs_output[W', K]
    fullyconnected( lhs[H, K], trs_output[W', K] ) -> output(H, W')
    """

    def define_fc_with_transpose(
        self, node, inputs, outputs
    ) -> circle.Operator.OperatorT:
        lhs, rhs = inputs

        # get transpose shape
        rhs_tid: int = self.graph.get_tid_registered(rhs)
        rhs_tensor: circle.Tensor.TensorT = self.graph.tensors[rhs_tid]
        rhs_name: str = rhs.name
        rhs_type: int = rhs_tensor.type
        rhs_shape: List[int] = rhs_tensor.shape
        assert len(rhs_shape) == 2, len(rhs_shape)
        rhs_shape_transpose = [rhs_shape[1], rhs_shape[0]]

        # create transpose output tensor
        trs_output = self.graph.add_tensor_from_scratch(
            prefix=f"{rhs_name}_transposed_output",
            shape=rhs_shape_transpose,
            dtype=rhs_type,
            source_node=node,
        )
        trs_perm = self.graph.add_const_tensor(data=[1, 0], source_node=node)
        trs_operator = self.define_transpose_node([rhs, trs_perm], [trs_output])
        self.graph.add_operator(trs_operator)

        # define fc node
        fc_input = lhs
        fc_weight = trs_output
        fc_shape = [fc_weight.shape[0]]
        fc_bias = self.graph.add_const_tensor(
            data=[0.0] * fc_shape[0], source_node=node
        )

        operator = self.define_fc_node([fc_input, fc_weight, fc_bias], outputs)

        return operator

    def define_node(
        self, node: torch.fx.Node, prior_latency=True
    ) -> circle.Operator.OperatorT:
        """
        NOTE: Possibility of accuracy-latency trade-off
        From ONE compiler's perspective:
        - BMM uses per-tensor quantization for both rhs and lhs.
        - FC uses per-channel quantization for weight and per-tensor for input.
        Thus, FC is better in terms of accuracy.
        FC necessarily involves an additional transpose operation to be identical with mm.
        If transposed operand is const, it can be optimized by constant folding.
        Thus, convert FC only if tranpose can be folded.
        TODO set prior_latency outside
        """
        args = MatmulArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        other = args.other

        inputs = [input, other]
        outputs = [node]

        if not is_const(other) and prior_latency:
            operator = self.define_bmm_node(inputs, outputs)
        else:
            operator = self.define_fc_with_transpose(node, inputs, outputs)

        return operator
