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

import operator
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx

import torch
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_shape
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import set_new_meta_val


@dataclass
class RankMapping:
    """
    Mapping from a 3D tensor view to its original 4D tensor view.

    Example:
        4D: [B, 1, T, D]
        3D: [B, T, D]

    Then:
        axis_3d_to_4d = {0: 0, 1: 2, 2: 3}
        inserted_axis_4d = 1
    """

    axis_3d_to_4d: dict[int, int]
    inserted_axis_4d: int

    def remap_axis(self, axis: int) -> int:
        """
        Remap one 3D axis into the corresponding 4D axis.
        """
        if axis < 0:
            axis += 3
        if axis not in self.axis_3d_to_4d:
            raise ValueError(f"Cannot remap 3D axis {axis}")
        return self.axis_3d_to_4d[axis]

    def remap_axes(self, axes: list[int]) -> list[int]:
        """
        Remap a list of 3D axes into 4D axes.
        """
        return [self.remap_axis(axis) for axis in axes]

    def remap_permute(self, perm_3d: list[int]) -> list[int]:
        """
        Convert a 3D permutation into the equivalent 4D permutation by
        keeping the inserted singleton axis at its original 4D position.
        """
        mapped = [self.remap_axis(axis) for axis in perm_3d]

        perm_4d = []
        mapped_idx = 0
        for out_axis_4d in range(4):
            if out_axis_4d == self.inserted_axis_4d:
                perm_4d.append(self.inserted_axis_4d)
            else:
                perm_4d.append(mapped[mapped_idx])
                mapped_idx += 1

        return perm_4d

    def remap_transpose(self, dim0: int, dim1: int) -> tuple[int, int]:
        """
        Convert a 3D transpose pair into the equivalent 4D transpose pair.
        """
        return self.remap_axis(dim0), self.remap_axis(dim1)


@dataclass
class Region:
    """
    A supported subgraph region between:
      - a source reshape that removes one singleton axis from 4D to 3D
      - one or more sink reshapes that lift a 3D value to 4D

    Internal nodes are supported ops that can be rewritten from 3D semantics
    into 4D semantics.
    """

    source_reshape: torch.fx.Node
    original_4d: torch.fx.Node
    mapping: RankMapping
    internal_nodes: list[torch.fx.Node]
    sink_reshapes: list[torch.fx.Node]


@trace_graph_diff_on_pass
class EliminateRankRoundTripRegion(PassBase):
    """
    Eliminate a source rank-reducing reshape and rewrite the reachable supported
    3D subgraph directly on the original 4D tensor.

    Pattern:
        reshape(4D -> 3D)
          -> supported small DAG region
          -> one or more reshape(3D -> 4D)

    Important behavior:
      - The source reshape must remove exactly one singleton axis.
      - Sink reshapes are treated as weak boundaries. They do not need to be the
        exact inverse of the source reshape.
      - After region rewrite, each sink reshape is updated to consume the new 4D
        producer. If the sink becomes an identity/no-op later, another peephole
        pass may remove it.
    """

    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = enabled

    @staticmethod
    def _is_call_function(node: torch.fx.Node, target) -> bool:
        """
        Check whether a node is a call_function node targeting the given op.
        """
        return node.op == "call_function" and node.target == target

    @staticmethod
    def _normalize_shape(shape) -> list[int]:
        """
        Normalize a shape-like object into a plain Python list.
        """
        return list(shape)

    @staticmethod
    def _tensor_node_inputs(node: torch.fx.Node) -> list[torch.fx.Node]:
        """
        Collect tensor-like node inputs from node.args.

        This function intentionally inspects only positional arguments because
        the supported operators in this pass use positional tensor inputs.
        """
        return [arg for arg in node.args if isinstance(arg, torch.fx.Node)]

    def _match_reshape_4d_to_3d(
        self, node: torch.fx.Node
    ) -> tuple[torch.fx.Node, RankMapping] | None:
        """
        Match a reshape that removes exactly one singleton dimension.

        Examples:
            [B, 1, T, D] -> [B, T, D]
            [B, T, 1, D] -> [B, T, D]
            [B, T, D, 1] -> [B, T, D]

        If multiple singleton axes could be removed to produce the same 3D shape,
        prefer removing a non-leading singleton axis first. This avoids
        accidentally interpreting the batch axis as the removed singleton axis
        in shapes such as [1, 1, T, D] -> [1, T, D].
        """
        if not self._is_call_function(node, torch.ops.aten.reshape.default):
            return None

        src = node.args[0]
        dst_shape = node.args[1]

        if not isinstance(src, torch.fx.Node):
            return None

        src_shape = self._normalize_shape(extract_shape(src))
        dst_shape = self._normalize_shape(dst_shape)

        if len(src_shape) != 4 or len(dst_shape) != 3:
            return None

        candidate_axes = [i for i, dim in enumerate(src_shape) if dim == 1]

        # Prefer removing a non-leading singleton axis first.
        candidate_axes.sort(key=lambda i: (i == 0, i))

        for inserted_axis in candidate_axes:
            squeezed = src_shape[:inserted_axis] + src_shape[inserted_axis + 1 :]
            if squeezed == dst_shape:
                axis_3d_to_4d = {}
                j = 0
                for i in range(4):
                    if i == inserted_axis:
                        continue
                    axis_3d_to_4d[j] = i
                    j += 1
                return src, RankMapping(
                    axis_3d_to_4d=axis_3d_to_4d,
                    inserted_axis_4d=inserted_axis,
                )

        return None

    def _is_supported_unary(self, node: torch.fx.Node) -> bool:
        """
        Check whether a unary op is supported.
        """
        return node.op == "call_function" and node.target in {
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.relu.default,
            torch.ops.aten.rsqrt.default,
            torch.ops.aten.silu.default,
            torch.ops.aten.neg.default,
        }

    def _is_supported_binary(self, node: torch.fx.Node) -> bool:
        """
        Check whether a binary elementwise op is supported.
        """
        return node.op == "call_function" and node.target in {
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor,
        }

    def _is_supported_pow(self, node: torch.fx.Node) -> bool:
        """
        Check whether a scalar power op is supported.
        """
        return (
            node.op == "call_function"
            and node.target == torch.ops.aten.pow.Tensor_Scalar
            and len(node.args) == 2
            and isinstance(node.args[1], (int, float))
        )

    def _is_supported_mean(self, node: torch.fx.Node) -> bool:
        """
        Check whether a mean reduction is supported.
        """
        return (
            node.op == "call_function"
            and node.target == torch.ops.aten.mean.dim
            and len(node.args) >= 2
            and isinstance(node.args[1], (list, tuple))
        )

    def _is_supported_permute(self, node: torch.fx.Node) -> bool:
        """
        Check whether a 3D permute is supported.
        """
        return (
            node.op == "call_function"
            and node.target == torch.ops.aten.permute.default
            and len(node.args) == 2
            and isinstance(node.args[1], (list, tuple))
            and len(node.args[1]) == 3
        )

    def _is_supported_transpose(self, node: torch.fx.Node) -> bool:
        """
        Check whether a 3D transpose is supported.
        """
        return (
            node.op == "call_function"
            and node.target == torch.ops.aten.transpose.int
            and len(node.args) == 3
            and isinstance(node.args[1], int)
            and isinstance(node.args[2], int)
        )

    def _is_supported_split(self, node: torch.fx.Node) -> bool:
        """
        Check whether split_with_sizes is supported.
        """
        return (
            node.op == "call_function"
            and node.target == torch.ops.aten.split_with_sizes.default
            and len(node.args) == 3
            and isinstance(node.args[1], (list, tuple))
            and isinstance(node.args[2], int)
        )

    def _is_supported_chunk(self, node: torch.fx.Node) -> bool:
        """
        Check whether chunk is supported.
        """
        return (
            node.op == "call_function"
            and node.target == torch.ops.aten.chunk.default
            and len(node.args) == 3
            and isinstance(node.args[1], int)
            and isinstance(node.args[2], int)
        )

    def _is_supported_getitem(self, node: torch.fx.Node) -> bool:
        """
        Check whether tuple indexing is supported.
        """
        return (
            node.op == "call_function"
            and node.target == operator.getitem
            and len(node.args) == 2
            and isinstance(node.args[1], int)
        )

    def _is_supported_internal_node(self, node: torch.fx.Node) -> bool:
        """
        Check whether a node is supported inside the region.
        """
        return (
            self._is_supported_unary(node)
            or self._is_supported_binary(node)
            or self._is_supported_pow(node)
            or self._is_supported_mean(node)
            or self._is_supported_permute(node)
            or self._is_supported_transpose(node)
            or self._is_supported_split(node)
            or self._is_supported_chunk(node)
            or self._is_supported_getitem(node)
        )

    def _is_weak_sink_reshape(self, node: torch.fx.Node) -> bool:
        """
        Check whether a node is a weak sink reshape from rank 3 to rank 4.

        Unlike a strict restore matcher, this only checks rank change.
        It does not require the output shape to be the exact inverse of the
        source reshape.
        """
        if not self._is_call_function(node, torch.ops.aten.reshape.default):
            return False

        src = node.args[0]
        dst_shape = node.args[1]

        if not isinstance(src, torch.fx.Node):
            return False

        try:
            src_shape = self._normalize_shape(extract_shape(src))
        except Exception:
            return False

        dst_shape = self._normalize_shape(dst_shape)

        return len(src_shape) == 3 and len(dst_shape) == 4

    def _is_allowed_input_dependency(
        self,
        arg,
        region_nodes: set[torch.fx.Node],
        source_reshape: torch.fx.Node,
    ) -> bool:
        """
        Check whether an input argument is allowed for a region internal node.

        Allowed inputs include:
        - the source reshape (region entry)
        - nodes already collected inside the region
        - non-node constants
        - external immutable inputs such as get_attr / placeholder

        Any other dependency is considered invalid and breaks region closure.
        """
        if not isinstance(arg, torch.fx.Node):
            return True

        if arg is source_reshape:
            return True

        if arg in region_nodes:
            return True

        if arg.op in {"get_attr", "placeholder"}:
            return True

        return False

    def _collect_region(
        self,
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
        mapping: RankMapping,
    ) -> Region | None:
        """
        Collect a supported region reachable from the source reshape.

        The collection is performed in two phases:
          1. Collect candidate internal nodes and weak sink reshapes by forward
             traversal over supported nodes.
          2. Validate closure: every internal node must depend only on
             - the source reshape,
             - other internal nodes,
             - or allowed external inputs.

        Weak sink reshapes are treated as region boundaries and are not required
        to be the exact inverse of the source reshape.
        """
        visited: set[torch.fx.Node] = {source_reshape}
        internal_nodes: list[torch.fx.Node] = []
        sink_reshapes: list[torch.fx.Node] = []

        q = deque([source_reshape])

        # Phase 1: collect candidates without requiring all inputs to already be
        # present in region (visited).
        while q:
            producer = q.popleft()

            for user in list(producer.users):
                if self._is_weak_sink_reshape(user):
                    if user not in sink_reshapes:
                        sink_reshapes.append(user)
                    continue

                if not self._is_supported_internal_node(user):
                    continue

                if user not in visited:
                    visited.add(user)
                    internal_nodes.append(user)
                    q.append(user)

        if not sink_reshapes:
            return None

        internal_set = set(internal_nodes)

        # Phase 2: validate input closure.
        for node in internal_nodes:
            tensor_inputs = self._tensor_node_inputs(node)
            for arg in tensor_inputs:
                if arg is source_reshape:
                    continue
                if arg in internal_set:
                    continue
                if self._is_allowed_input_dependency(
                    arg, internal_set | {source_reshape}, source_reshape
                ):
                    continue
                return None

        # Phase 3: ensure region does not leak to unsupported users except sinks.
        for node in internal_nodes:
            for user in node.users:
                if user in internal_set:
                    continue
                if user in sink_reshapes:
                    continue
                return None

        return Region(
            source_reshape=source_reshape,
            original_4d=original_4d,
            mapping=mapping,
            internal_nodes=internal_nodes,
            sink_reshapes=sink_reshapes,
        )

    def _topo_sort_region(self, region: Region) -> list[torch.fx.Node]:
        """
        Topologically sort internal region nodes according to graph order.
        """
        internal_set = set(region.internal_nodes)
        return [
            node for node in region.source_reshape.graph.nodes if node in internal_set
        ]

    def _map_arg(
        self,
        arg,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
    ):
        """
        Map one old argument to its rewritten equivalent.
        """
        if not isinstance(arg, torch.fx.Node):
            return arg

        if arg is source_reshape:
            return original_4d

        if arg in env:
            return env[arg]

        return arg

    def _rewrite_unary(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
    ) -> torch.fx.Node:
        """
        Rewrite a unary operator.
        """
        inp = self._map_arg(node.args[0], env, source_reshape, original_4d)
        assert isinstance(node.target, torch._ops.OpOverload), type(node.target)
        return create_node(
            graph,
            node.target,
            args=(inp,),
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_binary(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
    ) -> torch.fx.Node:
        """
        Rewrite a binary elementwise operator.
        """
        lhs = self._map_arg(node.args[0], env, source_reshape, original_4d)
        rhs = self._map_arg(node.args[1], env, source_reshape, original_4d)
        assert isinstance(node.target, torch._ops.OpOverload), type(node.target)
        return create_node(
            graph,
            node.target,
            args=(lhs, rhs),
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_pow(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
    ) -> torch.fx.Node:
        """
        Rewrite a scalar power operator.
        """
        inp = self._map_arg(node.args[0], env, source_reshape, original_4d)
        exponent = node.args[1]
        assert isinstance(node.target, torch._ops.OpOverload), type(node.target)
        return create_node(
            graph,
            node.target,
            args=(inp, exponent),
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_mean(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
        mapping: RankMapping,
    ) -> torch.fx.Node:
        """
        Rewrite a mean reduction by remapping 3D dims into 4D dims.
        """
        inp = self._map_arg(node.args[0], env, source_reshape, original_4d)
        dims_3d = list(node.args[1])  # type: ignore[arg-type]
        dims_4d = mapping.remap_axes(dims_3d)
        args = (inp, dims_4d)
        if len(node.args) == 3:
            keep_dim = node.args[2]
            args = (inp, dims_4d, keep_dim)  # type: ignore[assignment]
        assert isinstance(node.target, torch._ops.OpOverload), type(node.target)
        return create_node(
            graph,
            node.target,
            args=args,
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_permute(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
        mapping: RankMapping,
    ) -> torch.fx.Node:
        """
        Rewrite a 3D permute into the equivalent 4D permute.
        """
        inp = self._map_arg(node.args[0], env, source_reshape, original_4d)
        perm_3d = list(node.args[1])  # type: ignore[arg-type]
        perm_4d = mapping.remap_permute(perm_3d)
        assert isinstance(node.target, torch._ops.OpOverload), type(node.target)
        return create_node(
            graph,
            node.target,
            args=(inp, perm_4d),
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_transpose(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
        mapping: RankMapping,
    ) -> torch.fx.Node:
        """
        Rewrite a 3D transpose into the equivalent 4D transpose.
        """
        inp = self._map_arg(node.args[0], env, source_reshape, original_4d)
        dim0_3d = node.args[1]
        dim1_3d = node.args[2]
        assert isinstance(dim0_3d, int) and isinstance(dim1_3d, int)
        dim0_4d, dim1_4d = mapping.remap_transpose(dim0_3d, dim1_3d)
        assert isinstance(node.target, torch._ops.OpOverload), type(node.target)
        return create_node(
            graph,
            node.target,
            args=(inp, dim0_4d, dim1_4d),
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_split(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
        mapping: RankMapping,
    ) -> torch.fx.Node:
        """
        Rewrite split_with_sizes by remapping its split dimension.
        """
        inp = self._map_arg(node.args[0], env, source_reshape, original_4d)
        split_sizes = list(node.args[1])  # type: ignore[arg-type]
        dim_3d = node.args[2]
        assert isinstance(dim_3d, int)
        dim_4d = mapping.remap_axis(dim_3d)
        assert isinstance(node.target, torch._ops.OpOverload), type(node.target)
        return create_node(
            graph,
            node.target,
            args=(inp, split_sizes, dim_4d),
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_chunk(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
        mapping: RankMapping,
    ) -> torch.fx.Node:
        """
        Rewrite chunk by remapping its chunk dimension.
        """
        inp = self._map_arg(node.args[0], env, source_reshape, original_4d)
        chunks = node.args[1]
        dim_3d = node.args[2]
        assert isinstance(dim_3d, int)
        dim_4d = mapping.remap_axis(dim_3d)
        assert isinstance(node.target, torch._ops.OpOverload), type(node.target)
        return create_node(
            graph,
            node.target,
            args=(inp, chunks, dim_4d),
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_getitem(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
    ) -> torch.fx.Node:
        """
        Rewrite a tuple getitem.

        The producer is expected to be a rewritten split or chunk node.
        """
        inp = self._map_arg(node.args[0], env, source_reshape, original_4d)
        idx = node.args[1]
        return create_node(
            graph,
            operator.getitem,  # type: ignore[arg-type]
            args=(inp, idx),
            kwargs=node.kwargs,
            origin=node,
        )

    def _rewrite_node(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        env: dict[torch.fx.Node, torch.fx.Node],
        source_reshape: torch.fx.Node,
        original_4d: torch.fx.Node,
        mapping: RankMapping,
    ) -> torch.fx.Node:
        """
        Rewrite one supported region-internal node.
        """
        if self._is_supported_unary(node):
            return self._rewrite_unary(graph, node, env, source_reshape, original_4d)

        if self._is_supported_binary(node):
            return self._rewrite_binary(graph, node, env, source_reshape, original_4d)

        if self._is_supported_pow(node):
            return self._rewrite_pow(graph, node, env, source_reshape, original_4d)

        if self._is_supported_mean(node):
            return self._rewrite_mean(
                graph, node, env, source_reshape, original_4d, mapping
            )

        if self._is_supported_permute(node):
            return self._rewrite_permute(
                graph, node, env, source_reshape, original_4d, mapping
            )

        if self._is_supported_transpose(node):
            return self._rewrite_transpose(
                graph, node, env, source_reshape, original_4d, mapping
            )

        if self._is_supported_split(node):
            return self._rewrite_split(
                graph, node, env, source_reshape, original_4d, mapping
            )

        if self._is_supported_chunk(node):
            return self._rewrite_chunk(
                graph, node, env, source_reshape, original_4d, mapping
            )

        if self._is_supported_getitem(node):
            return self._rewrite_getitem(graph, node, env, source_reshape, original_4d)

        raise RuntimeError(f"Unsupported node in region rewrite: {node.target}")

    def _rewrite_region(
        self,
        exported_program: ExportedProgram,
        region: Region,
    ) -> bool:
        """
        Rewrite all internal region nodes in topological order and update each
        sink reshape to consume the rewritten 4D producer.

        The sink reshape is not required to be removed here. This pass focuses
        on eliminating the source rank-reducing reshape and lifting the region
        into 4D semantics.
        """
        gm = exported_program.graph_module
        graph = gm.graph

        env: dict[torch.fx.Node, torch.fx.Node] = {}
        topo_nodes = self._topo_sort_region(region)

        with graph.inserting_before(region.source_reshape):
            for node in topo_nodes:
                rewritten = self._rewrite_node(
                    graph=graph,
                    node=node,
                    env=env,
                    source_reshape=region.source_reshape,
                    original_4d=region.original_4d,
                    mapping=region.mapping,
                )
                set_new_meta_val(rewritten)
                env[node] = rewritten

        for sink in region.sink_reshapes:
            old_input = sink.args[0]
            if not isinstance(old_input, torch.fx.Node):
                return False

            if old_input is region.source_reshape:
                new_input = region.original_4d
            elif old_input in env:
                new_input = env[old_input]
            else:
                return False

            sink.update_arg(0, new_input)

        return True

    def call(self, exported_program: ExportedProgram) -> PassResult:
        """
        Run the pass over the graph.
        """
        if not self.enabled:
            return PassResult(False)

        gm = exported_program.graph_module
        graph = gm.graph
        modified = False

        for node in list(graph.nodes):
            matched = self._match_reshape_4d_to_3d(node)
            if matched is None:
                continue

            original_4d, mapping = matched
            region = self._collect_region(
                source_reshape=node,
                original_4d=original_4d,
                mapping=mapping,
            )
            if region is None:
                continue

            if not region.internal_nodes:
                continue

            modified |= self._rewrite_region(
                exported_program=exported_program,
                region=region,
            )

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            gm.recompile()

        return PassResult(modified)
