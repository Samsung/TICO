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
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx

import torch
from torch.export import ExportedProgram

from tico.passes.convert_gather_to_gather_nd import ConvertGatherToGatherNd
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node, set_new_meta_val
from tico.utils.validate_args_kwargs import IndexArgs


@trace_graph_diff_on_pass
class LowerMultiIndexToGatherNd(PassBase):
    """
    Lower ``aten.index.Tensor`` with several index tensors to
    ``circle_custom.gather_nd``.

    PyTorch advanced indexing with ``k`` leading index tensors broadcasts the
    indices to a common shape ``B`` and reads one coordinate per element::

        out[b...] = input[i0[b...], i1[b...], ..., i(k-1)[b...], ...]

    Circle ``GATHER_ND`` reads full coordinates from the last dimension of its
    ``indices`` input, so the lowering stacks the broadcast indices::

        aten.index(input, [i0, ..., i(k-1)])
          -> circle_custom.gather_nd(
                 input, cat([unsqueeze(expand(int32(i), B), -1) ...], -1))

    The resulting ``indices`` tensor has shape ``B + [k]`` and the output keeps
    PyTorch's shape ``B + input.shape[k:]``.

    Preconditions
        - The node is ``aten.index.Tensor`` whose ``indices`` list contains at
          least two non-``None`` entries.
        - The non-``None`` entries are FX nodes that occupy the leading positions
          ``0 .. k-1``; any remaining entries are ``None``. Advanced indices that
          start after a sliced dimension or are separated by one are left
          untouched (the serializer reports them as unsupported).
        - Every index tensor has an int32 or int64 dtype and a static shape, and
          the input has a static shape.

    Transformation
        - Index tensors are cast to int32 when needed, broadcast to the common
          shape with ``aten.expand`` when their shape differs from it, unsqueezed
          and concatenated along a new trailing dimension.
        - The ``aten.index`` node is replaced by ``circle_custom.gather_nd`` with
          recomputed metadata; its users are redirected.

    Postconditions
        No ``aten.index.Tensor`` with two or more leading index tensors remains.

    Semantic assumptions
        Index values are non-negative, as for the single-index ``GATHER``
        lowering in the serializer. ``GATHER_ND`` does not wrap negative
        coordinates.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if not is_target_node(node, torch.ops.aten.index.Tensor):
                continue

            args = IndexArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            input_node = args.input
            indices = list(args.indices)

            index_nodes: List["torch.fx.Node"] = []
            for position, index in enumerate(indices):
                if index is None:
                    continue
                if not isinstance(index, torch.fx.Node):
                    # Non-node indices are outside this lowering.
                    index_nodes = []
                    break
                if position != len(index_nodes):
                    # Advanced indices must be leading and contiguous.
                    index_nodes = []
                    break
                index_nodes.append(index)

            if len(index_nodes) < 2:
                continue

            for index in index_nodes:
                if index.meta["val"].dtype not in (torch.int32, torch.int64):
                    raise NotYetSupportedError(
                        "aten.index.Tensor indices must have int32 or int64 dtype "
                        "for GatherNd lowering."
                    )

            get_static_shape = ConvertGatherToGatherNd._get_static_shape
            get_static_shape(input_node, "aten.index input")
            index_shapes = [
                get_static_shape(index, "aten.index index") for index in index_nodes
            ]
            broadcast_shape = list(torch.broadcast_shapes(*index_shapes))

            logger.debug(
                "%s: lowering aten.index with %d indices to circle_custom.gather_nd "
                "(broadcast shape %r)",
                node,
                len(index_nodes),
                broadcast_shape,
            )

            with graph.inserting_before(node):
                coordinates = []
                for index, index_shape in zip(index_nodes, index_shapes):
                    coordinate = ConvertGatherToGatherNd._cast_index_to_int32(
                        graph, index
                    )
                    if index_shape != broadcast_shape:
                        coordinate = create_node(
                            graph,
                            torch.ops.aten.expand.default,
                            args=(coordinate, broadcast_shape),
                            origin=node,
                        )
                        set_new_meta_val(coordinate)
                    unsqueezed = create_node(
                        graph,
                        torch.ops.aten.unsqueeze.default,
                        args=(coordinate, -1),
                        origin=node,
                    )
                    set_new_meta_val(unsqueezed)
                    coordinates.append(unsqueezed)

                full_indices = create_node(
                    graph,
                    torch.ops.aten.cat.default,
                    args=(coordinates, -1),
                    origin=node,
                )
                set_new_meta_val(full_indices)

                gather_nd = create_node(
                    graph,
                    torch.ops.circle_custom.gather_nd,
                    args=(input_node, full_indices),
                    origin=node,
                )
                set_new_meta_val(gather_nd)

            assert gather_nd.meta["val"].shape == node.meta["val"].shape, (
                f"{node.name}: GatherNd output shape {gather_nd.meta['val'].shape} "
                f"differs from aten.index output shape {node.meta['val'].shape}"
            )
            node.replace_all_uses_with(gather_nd, propagate_meta=False)
            modified = True
            logger.debug("%s is replaced with %s", node.name, gather_nd.name)

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(modified)
