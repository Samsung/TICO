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

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_shape

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import enforce_type, is_target_node


@trace_graph_diff_on_pass
class DecomposeSliceScatter(PassBase):
    """
    Let's decompose slice_scatter.default to cat.

    slice_scatter with step=1 embeds src tensor to input tensor
    We can replace it with (1) slicing input tensors and (2) concatenating all tensors

    [1] When step = 1,

        (1) Split input to input_0 and input_1 (either of them can be zero-size)
        (2) Concatenate input_0, src, input_1

        Before)

            input                  src
            |                      |
            |                      |
            |                      |
            +--> slice_scatter <---+

        After)

            input
            |-------------------------
            |                         |
            |                         |
            |                         |
            slice_copy                slice_copy
            |                         |
            |                         |
            |                         |
            slice_0*      src          slice_1*
            |            |            |
            |            |            |
            |            |            |
            +---------> cat <---------+

            *Either of slice_0 or slice_1 could be empty. Then it's ignored.

    [2] When step > 1, not supported yet. (TBD)
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if not is_target_node(node, torch.ops.aten.slice_scatter.default):
                continue

            @enforce_type
            @dataclass
            class Args:
                """
                input (Tensor) the input tensor.
                src (Tensor) The tensor to embed into input
                dim (int) the dimension to insert the slice into
                start (Optional[int]) the start index of where to insert the slice
                end (Optional[int]) the end index of where to insert the slice
                step (int) the how many elements to skip in
                """

                input: torch.fx.Node
                src: torch.fx.Node
                dim: int = 0
                start: Optional[int] = None
                end: Optional[int] = None
                step: int = 1

            args = Args(*node.args, **node.kwargs)  # type: ignore[arg-type]

            input = args.input
            src = args.src
            dim = args.dim
            s = args.start
            e = args.end
            step = args.step

            # TODO Support step > 1 cases
            if step > 1:
                raise RuntimeError(
                    f"slice_scatter with step > 1 is not yet supported. Node: {node}"
                )

            start: int = 0 if s is None else s
            end: int = (
                extract_shape(src)[dim]
                if e is None
                else min(extract_shape(src)[dim], e)
            )

            with graph.inserting_before(node):
                slices = []

                if 0 < start:
                    slice_0 = create_node(
                        graph,
                        torch.ops.aten.slice_copy.Tensor,
                        args=(input, dim, 0, start, 1),
                        origin=node,
                    )
                    slices.append(slice_0)

                slices.append(src)

                if start + end < extract_shape(input)[dim]:
                    slice_1 = create_node(
                        graph,
                        torch.ops.aten.slice_copy.Tensor,
                        args=(
                            input,
                            dim,
                            start + end,
                            extract_shape(input)[dim],
                            1,
                        ),
                        origin=node,
                    )
                    slices.append(slice_1)

                concat = create_node(
                    graph, torch.ops.aten.cat.default, args=(slices, dim)
                )
                node.replace_all_uses_with(concat, propagate_meta=True)

            modified = True
            logger.debug(f"{node.name} is replaced with slice_copy + concat")

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(modified)
