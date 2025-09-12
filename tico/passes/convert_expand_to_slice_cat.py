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
class ConvertExpandToSliceCat(PassBase):
    """
    This pass replaces `aten.reshape` + `aten.expand` pattern by rewriting it using
    a series of `aten.slice` and `aten.cat` operations.

    This pass is specialized for expand of KVCache.
    - Expects (batch, num_key_value_heads, seq_len, head_dim) as input shape of reshape
    """

    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = enabled

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        # This pass handles expand on EXPAND_DIM only
        CAT_DIM = 1
        EXPAND_DIM = 2

        for node in graph.nodes:
            if not self.enabled:
                return PassResult(False)

            if not is_target_node(node, ops.aten.expand):
                continue

            expand_node = node
            expand_args = ExpandArgs(*expand_node.args, **expand_node.kwargs)
            expand_input = expand_args.input
            expand_shape = extract_shape(expand_node)

            if not isinstance(expand_input, torch.fx.Node) or not is_target_node(
                expand_input, ops.aten.reshape
            ):
                continue

            permute_node = expand_input
            permute_args = PermuteArgs(*permute_node.args, **permute_node.kwargs)
            permute_input = permute_args.input
            permute_shape = extract_shape(permute_node)

            if permute_shape[EXPAND_DIM] != 1:
                continue

            permute_input_shape = extract_shape(permute_input)

            if len(expand_shape) != len(permute_shape):
                continue

            # Ensure all dimensions *except* at EXPAND_DIM are identical.
            if not (
                expand_shape[:EXPAND_DIM] == permute_shape[:EXPAND_DIM]
                and expand_shape[EXPAND_DIM + 1 :] == permute_shape[EXPAND_DIM + 1 :]
            ):
                continue

            # Ensure the expansion dimension is a clean multiple.
            if expand_shape[EXPAND_DIM] % permute_shape[EXPAND_DIM] != 0:
                continue

            expand_ratio = expand_shape[EXPAND_DIM] // permute_shape[EXPAND_DIM]

            if expand_ratio <= 1:
                continue

            cat_nodes = []

            for i in range(permute_input_shape[CAT_DIM]):
                with graph.inserting_before(node):
                    slice_copy_args = (permute_input, CAT_DIM, i, i + 1, 1)
                    slice_node = create_node(
                        graph,
                        torch.ops.aten.slice.Tensor,
                        args=slice_copy_args,
                        origin=node,
                    )
                with graph.inserting_after(slice_node):
                    cat_args = ([slice_node] * expand_ratio, CAT_DIM)
                    cat_node = create_node(
                        graph,
                        torch.ops.aten.cat.default,
                        args=cat_args,
                        origin=node,
                    )
                    cat_nodes.append(cat_node)

            with graph.inserting_after(node):
                cat_args = (cat_nodes, CAT_DIM)
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
