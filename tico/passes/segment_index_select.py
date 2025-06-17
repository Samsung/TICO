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
from tico.utils.graph import add_placeholder, create_node, is_single_value_tensor
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_const_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import IndexSelectArgs


@trace_const_diff_on_pass
class SegmentIndexSelectConst(PassBase):
    """
    Let's segment index_select with multiple const indices to index_select operators with one index.
    WHY?
      Gather(index, index_select, select, embedding, ...) operation with const indices can be lowered to slice by LowerToSlice pass.
      For that, we need to split 'a index_select operator with multiple indice' to 'multiple index_select operators with one index'.
      Note that NPU is not fully compatible with gather operation.

    [before]
            input
                |
            index_select.default, len(index) > 1
                |
            output

    [after]

            input
                |
                -------------------------------------------------
                |                                                |
            index_select.default, len(index) == 1  , ... , index_select.default, len(index) == 1
                |                                                |
                -------------------------------------------------
                |
            torch.concat (input=[index_select0, index_select1, ...], axis = dim)
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

            if is_single_value_tensor(index):
                continue

            if len(index) < 2:
                continue

            input_shape = extract_shape(input)
            if dim < 0:
                dim = dim % len(input_shape)

            index_select_node_list = []
            for i in index:
                index_node = add_placeholder(
                    exported_program, torch.tensor([i]), prefix="segm_index"
                )
                with graph.inserting_before(node):
                    index_select_node = create_node(
                        graph,
                        torch.ops.aten.index_select.default,
                        args=(input, dim, index_node),
                        origin=node,
                    )
                    index_select_node_list.append(index_select_node)

            with graph.inserting_before(node):
                concat_node = create_node(
                    graph,
                    torch.ops.aten.cat.default,
                    args=(index_select_node_list, dim),
                )

                node.replace_all_uses_with(concat_node, propagate_meta=True)

            modified = True
            logger.debug(
                f"{node.name} is replaced with {concat_node.name} and {index_select_node_list}"
            )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
