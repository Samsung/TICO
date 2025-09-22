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
from torch._export.utils import is_lifted_tensor_constant
from torch.export import ExportedProgram

from tico.passes import ops
from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import AddTensorArgs, ExpandArgs, PermuteArgs


@trace_graph_diff_on_pass
class LegalizeExpand(PassBase):
    """
    This pass replaces `aten.reshape` + `aten.expand` pattern with `aten.cat`
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if not is_target_node(node, ops.aten.expand):
                continue

            args = ExpandArgs(*node.args, **node.kwargs)
            expand_input = args.input
            expand_shape = args.size
            print(f"expand shape: {expand_shape}")

            if not isinstance(expand_input, torch.fx.Node) or not is_target_node(
                expand_input, ops.aten.reshape
            ):
                continue

            permute_args = PermuteArgs(*expand_input.args, **expand_input.kwargs)
            permute_input = permute_args.input
            permute_dims = permute_args.dims

            permute_input_shape = extract_shape(permute_input)

            print(f"permute dims: {permute_dims}")

            cat_nodes = []

            for i in range(permute_input_shape[1]):
                # [1, 8, 5, 64] -> [1, 1, 5, 64]
                with graph.inserting_after(expand_input):
                    slice_copy_args = (permute_input, 1, i, i + 1, 1)
                    slice_node = create_node(
                        graph,
                        torch.ops.aten.slice.Tensor,
                        args=slice_copy_args,
                        origin=node,
                    )
                # [1, 1, 5, 64] -> [1, 4, 5, 64]
                with graph.inserting_after(slice_node):
                    cat_args = ([slice_node] * 4, 1)
                    cat_node = create_node(
                        graph,
                        torch.ops.aten.cat.default,
                        args=cat_args,
                        origin=node,
                    )
                    cat_nodes.append(cat_node)

            # [1, 4, 5, 64]*8 -> [1, 32, 5, 64]
            with graph.inserting_after(node):
                cat_args = (cat_nodes, 1)
                cat_node = create_node(
                    graph,
                    torch.ops.aten.cat.default,
                    args=cat_args,
                    origin=node,
                )
                node.replace_all_uses_with(cat_node)

            modified = True
            logger.debug(f"{node.name} is replaced with {cat_node.name} operators")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
