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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export import ExportedProgram

from tico.passes import ops
from tico.serialize.circle_graph import extract_shape
from tico.utils import logging
from tico.utils.graph import create_node, is_single_value_tensor
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_const_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import IndexSelectArgs, SelectCopyIntArgs


def passes():
    """
    This pass lowers aten.ops.select/selct_copy.int to aten.ops.slice.
    We support only when it is index in args, which is a constant tensor.
    Since the index in node'args isn't constant tensor, we can't support converting the below op list yet.

    TODO Support below with const indices
    - torch.ops.aten.embedding.default
    - torch.ops.aten.index.Tensor
    """
    return [
        LowerSelectCopyToSlice(),
        LowerIndexSelectToSlice(),
    ]


@trace_const_diff_on_pass
class LowerSelectCopyToSlice(PassBase):
    """
    [before]
            input
                |
            select (tensor, dim, *index)
                |
            output

    [after]

            input
                |
            slice (input=tensor, dim=dim, start=index, end=index+1, step=1)
                |
            reshape (input=slice_copy, size=select_shape)
                |
            output
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, ops.aten.select):
                continue

            args = SelectCopyIntArgs(*node.args, **node.kwargs)
            input = args.input
            dim = args.dim
            index = args.index

            input_shape = extract_shape(input)
            if dim < 0:
                dim = dim % len(input_shape)

            start = index
            end = index + 1
            step = 1
            slice_copy_args = (input, dim, start, end, step)

            with graph.inserting_after(node):
                # slice
                slice_node = create_node(
                    graph,
                    torch.ops.aten.slice.Tensor,
                    args=slice_copy_args,
                    origin=node,
                )
                node_shape = extract_shape(node)
            with graph.inserting_after(slice_node):
                # reshape
                reshape_args = (slice_node, list(node_shape))
                reshape_node = create_node(
                    graph,
                    torch.ops.aten.reshape.default,
                    args=reshape_args,
                )
                node.replace_all_uses_with(reshape_node, propagate_meta=True)

            modified = True
            logger.debug(
                f"{node.name} is replaced with {slice_node.name} and {reshape_node.name} operators"
            )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)


@trace_const_diff_on_pass
class LowerIndexSelectToSlice(PassBase):
    """

    [before]
            input
                |
            index_select.default  (tensor, dim, *index)
                |
            output

    [after]

            input
                |
            slice (input=tensor, dim=dim, start=index, end=index+1, step=1)
                |
            reshape (input=slice_copy, size=select_shape)
                |
            output
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, ops.aten.index_select):
                continue

            args = IndexSelectArgs(*node.args, **node.kwargs)
            input = args.input
            dim = args.dim
            index = args.index

            input_shape = extract_shape(input)
            if dim < 0:
                dim = dim % len(input_shape)

            if isinstance(index, torch.fx.Node):
                if is_lifted_tensor_constant(exported_program, index):
                    index = get_lifted_tensor_constant(exported_program, index)  # type: ignore[assignment]
                elif is_param(exported_program, index):
                    index = get_param(exported_program, index)  # type: ignore[assignment]
                elif is_buffer(exported_program, index):
                    index = get_buffer(exported_program, index)  # type: ignore[assignment]
                else:
                    continue

            if not isinstance(index, torch.Tensor):
                continue

            if not is_single_value_tensor(index):
                # need to be lowered by LowerIndexSelect pass
                continue
            index_int = index.item()  # convert scalar tensor to int

            start = index_int
            end = index_int + 1
            step = 1
            slice_copy_args = (input, dim, start, end, step)

            with graph.inserting_after(node):
                # slice
                slice_node = create_node(
                    graph,
                    torch.ops.aten.slice.Tensor,
                    args=slice_copy_args,
                    origin=node,
                )
                node_shape = extract_shape(node)
            with graph.inserting_after(slice_node):
                # reshape
                reshape_args = (slice_node, list(node_shape))
                reshape_node = create_node(
                    graph,
                    torch.ops.aten.reshape.default,
                    args=reshape_args,
                )
                node.replace_all_uses_with(reshape_node, propagate_meta=True)

            modified = True
            logger.debug(
                f"{node.name} is replaced with {slice_node.name} and {reshape_node.name} operators"
            )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
