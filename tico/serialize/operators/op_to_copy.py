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

from tico.passes import ops

from tico.serialize.circle_mapping import (
    extract_circle_dtype,
    extract_torch_dtype,
    to_circle_dtype,
)
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import ToCopyArgs, ToDtypeArgs, ToDtypeLayoutArgs


@register_node_visitor
class ToCopyVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = ops.aten.to_copy

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_cast_node(
        self,
        inputs: List[torch.fx.Node],
        outputs: List[torch.fx.Node],
        in_type: int,
        out_type: int,
    ):
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.CAST, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.CastOptions
        option = circle.CastOptions.CastOptionsT()
        option.inDataType = in_type
        option.outDataType = out_type
        operator.builtinOptions = option

        return operator

    def parse_args(self, op: torch._ops.OpOverload, args, kwargs):
        ret: Union[ToCopyArgs, ToDtypeArgs, ToDtypeLayoutArgs]
        if op is torch.ops.aten._to_copy.default:
            ret = ToCopyArgs(*args, **kwargs)
        elif op is torch.ops.aten.to.dtype:
            ret = ToDtypeArgs(*args, **kwargs)
        elif op is torch.ops.aten.to.dtype_layout:
            ret = ToDtypeLayoutArgs(*args, **kwargs)
        else:
            raise NotImplementedError(f"Unsupported to_copy/to operator: {op}")

        return ret

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = ToCopyArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        dtype = args.dtype
        layout = args.layout
        # device is meaningless in circle

        pin_memory = args.pin_memory
        non_blocking = args.non_blocking
        memory_format = args.memory_format

        if pin_memory is not None:
            raise NotYetSupportedError("Do not support pin_memory yet")
        if non_blocking is True:
            raise NotYetSupportedError("Do not support non_blocking yet")
        if memory_format is not None:
            raise NotYetSupportedError("Do not support memory_format yet")

        input_meta = input.meta["val"]
        # https://pytorch.org/docs/stable/tensor_attributes.html#torch-layout
        # layout is two types: torch.strided(dense Tensors), torch.sparse_coo(sparse COO Tensors)
        if "layout" in input.kwargs and input.kwargs["layout"] != input_meta:
            raise NotYetSupportedError(
                f"Only support when node and its input have same layout: (input layout: {input_meta}), (node layout: {layout})."
            )

        if dtype is None:
            dtype = extract_torch_dtype(node)
        assert isinstance(dtype, torch.dtype), type(dtype)

        # define cast node
        in_type: int = extract_circle_dtype(input)
        out_type: int = to_circle_dtype(dtype)
        inputs = [input]
        outputs = [node]
        operator = self.define_cast_node(inputs, outputs, in_type, out_type)

        # TODO Support layout conversion

        return operator
