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

from typing import Dict, List, Optional, TYPE_CHECKING, Union

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

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = ExpandArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        size = args.size

        input_tensor: circle.Tensor.TensorT = self.graph.get_tensor(input)
        input_shape: List[int] = input_tensor.shape
        input_signature: Optional[List[int]] = input_tensor.shapeSignature

        extending_rank = len(size) - len(input_shape)
        assert extending_rank >= 0, "expand cannot reduce the rank."

        # Resolved target size per output dimension. `None` marks a dimension
        # whose size is only known at runtime (a dynamic input dimension that
        # is kept as is); it is read from the input shape when the model runs.
        resolved: List[Optional[int]] = []
        for idx, dim in enumerate(size):
            dim = int(dim)
            if idx < extending_rank:
                assert (
                    dim >= 1
                ), "A dim value(less than 1) isn't allowed in the extending_rank."
                resolved.append(dim)
                continue

            input_idx = idx - extending_rank
            input_dim = input_shape[input_idx]
            input_is_dynamic = (
                input_signature is not None and input_signature[input_idx] == -1
            )
            """
            In pytorch, passing -1 as the size for a dimension means that the size of that dimension won't be changed.
            But, circle in ONE does not support this.
            So, dim value(-1) in the non-extending_rank is converted to the size for the dimension: a constant for
            static dimensions and a runtime SHAPE lookup for dynamic dimensions.
            """
            if dim == -1:
                resolved.append(None if input_is_dynamic else input_dim)
                continue
            if input_is_dynamic:
                # The actual input size is unknown here; BROADCAST_TO validates
                # it at runtime (it must be 1 or equal to `dim`).
                resolved.append(dim)
                continue
            assert (
                input_dim == 1 or input_dim == dim
            ), f"The size of dimension to be expanded ({input_dim}) must be 1 or the expanded size ({dim})."
            resolved.append(dim)

        if all(value is not None for value in resolved):
            size_i32 = circle_legalize_dtype_to(resolved, dtype=torch.int32)
            return self.define_expand_copy_node([input, size_i32], [node])

        shape_tensor = self.define_runtime_shape(node, input, input_shape, resolved)
        return self.define_expand_copy_node([input, shape_tensor], [node])

    def define_runtime_shape(
        self,
        node: torch.fx.Node,
        input: torch.fx.Node,
        input_shape: List[int],
        resolved: List[Optional[int]],
    ) -> circle.Tensor.TensorT:
        """
        Build an INT32 shape tensor at runtime for BROADCAST_TO.

        SHAPE(input) -> STRIDED_SLICE per dynamic dimension -> CONCATENATION with
        the constant dimensions, in output dimension order.
        """
        input_rank = len(input_shape)
        shape_tensor = self.graph.add_tensor_from_scratch(
            prefix=f"{node.name}_input_shape",
            shape=[input_rank],
            shape_signature=None,
            dtype=circle.TensorType.TensorType.INT32,
            source_node=node,
        )
        shape_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.SHAPE, self._op_codes
        )
        shape_operator = create_builtin_operator(
            self.graph, shape_op_index, [input], [shape_tensor]
        )
        shape_operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.ShapeOptions
        )
        shape_option = circle.ShapeOptions.ShapeOptionsT()
        shape_option.outType = circle.TensorType.TensorType.INT32
        shape_operator.builtinOptions = shape_option
        self.graph.add_operator(shape_operator)

        extending_rank = len(resolved) - input_rank
        pieces: List[Union[torch.Tensor, circle.Tensor.TensorT]] = []
        static_run: List[int] = []

        def flush_static_run() -> None:
            if static_run:
                pieces.append(torch.as_tensor(list(static_run), dtype=torch.int32))
                static_run.clear()

        slice_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.STRIDED_SLICE, self._op_codes
        )
        for idx, value in enumerate(resolved):
            if value is not None:
                static_run.append(value)
                continue
            flush_static_run()
            input_idx = idx - extending_rank
            piece = self.graph.add_tensor_from_scratch(
                prefix=f"{node.name}_dim{idx}",
                shape=[1],
                shape_signature=None,
                dtype=circle.TensorType.TensorType.INT32,
                source_node=node,
            )
            slice_operator = create_builtin_operator(
                self.graph,
                slice_op_index,
                [
                    shape_tensor,
                    torch.as_tensor([input_idx], dtype=torch.int32),
                    torch.as_tensor([input_idx + 1], dtype=torch.int32),
                    torch.as_tensor([1], dtype=torch.int32),
                ],
                [piece],
            )
            slice_operator.builtinOptionsType = (
                circle.BuiltinOptions.BuiltinOptions.StridedSliceOptions
            )
            slice_operator.builtinOptions = (
                circle.StridedSliceOptions.StridedSliceOptionsT()
            )
            self.graph.add_operator(slice_operator)
            pieces.append(piece)
        flush_static_run()

        target_shape = self.graph.add_tensor_from_scratch(
            prefix=f"{node.name}_target_shape",
            shape=[len(resolved)],
            shape_signature=None,
            dtype=circle.TensorType.TensorType.INT32,
            source_node=node,
        )
        concat_op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.CONCATENATION, self._op_codes
        )
        concat_operator = create_builtin_operator(
            self.graph, concat_op_index, pieces, [target_shape]
        )
        concat_operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.ConcatenationOptions
        )
        concat_option = circle.ConcatenationOptions.ConcatenationOptionsT()
        concat_option.axis = 0
        concat_option.fusedActivationFunction = (
            circle.ActivationFunctionType.ActivationFunctionType.NONE
        )
        concat_operator.builtinOptions = concat_option
        self.graph.add_operator(concat_operator)
        return target_shape
