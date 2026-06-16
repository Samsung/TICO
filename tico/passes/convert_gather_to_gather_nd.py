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
# See the License for the specific language governing permissions of
# limitations under the License.

import torch
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import GatherArgs


@trace_graph_diff_on_pass
class ConvertGatherToGatherNd(PassBase):
    """
    This pass transforms torch.gather to operations compatible with Circle GATHER_ND.

    This pass constructs multi-dimensional indices by:
    1. Creating coordinate grids for all non-gather dimensions
    2. Stacking them with the gather indices to form full coordinates
    3. Replacing the original gather with an index operation using these coordinates
    """

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph

        modified = False
        for node in graph.nodes:
            if not is_target_node(node, torch.ops.aten.gather.default):
                continue
            # Skip if this node was already processed.
            if node.meta.get("gather-to-gather-nd-passed", False):
                continue

            args = GatherArgs(*node.args, **node.kwargs)  # type: ignore
            args_input_shape = args.input.meta["val"].shape
            args_index_shape = args.index.meta["val"].shape

            # Normalize dimension index to be positive.
            if args.dim < 0:
                args.dim = len(args_input_shape) + args.dim

            logger.debug(
                "%s: Lowering to GATHER_ND: input=%r dim=%d",
                node,
                args_input_shape,
                args.dim,
            )

            with graph.inserting_before(node):
                # Create coordinate grids for each dimension.

                indices = []
                for i in range(len(args_input_shape)):
                    if i == args.dim:
                        # Use the original index tensor for the gather dimension.
                        indices.append(args.index)
                        continue

                    # For dimensions other than the gather dimension, create a
                    # coordinate grid with shape [1, 1, ..., index_shape[i], ..., 1].

                    # Create a range from 0 to the size of the n-th dimension.
                    arange_node = create_node(
                        graph,
                        torch.ops.aten.arange.start_step,
                        args=(0, args_index_shape[i], 1),
                        kwargs={"dtype": torch.int32},
                        origin=node,
                    )

                    # Use INDEX shape for the range.
                    shape = [1] * len(args_index_shape)
                    shape[i] = args_index_shape[i]
                    reshape_node = create_node(
                        graph,
                        torch.ops.aten.reshape.default,
                        args=(arange_node, shape),
                        origin=node,
                    )

                    # Expand to match the output shape.
                    indices.append(
                        create_node(
                            graph,
                            torch.ops.aten.expand.default,
                            args=(reshape_node, list(args_index_shape)),
                            origin=node,
                        )
                    )

                # Stack all indices along the last dimension. We have to do it
                # manually because torch.stack is not supported for our case.
                indices = [
                    create_node(
                        graph,
                        torch.ops.aten.unsqueeze.default,
                        args=(index, -1),
                        origin=node,
                    )
                    for index in indices
                ]
                stacked_indices = create_node(
                    graph,
                    torch.ops.aten.cat.default,
                    args=(indices, -1),
                    origin=node,
                )

            # Replace the original index tensor with the stacked indices.
            # The gather node will now use GATHER_ND with these indices

            if len(node.args) > 2:
                new_args = list(node.args)
                new_args[2] = stacked_indices
                node.args = tuple(new_args)

            node.meta["gather-to-gather-nd-passed"] = True
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(modified)
