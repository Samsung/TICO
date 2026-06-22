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

from typing import Any, List, Mapping, Tuple

import torch
from torch.export import ExportedProgram
from torch.export.exported_program import InputKind, InputSpec, TensorArgument

from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node, get_first_user_input
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import get_fake_mode, is_target_node, set_new_meta_val


@trace_graph_diff_on_pass
class ConvertGatherToGatherNd(PassBase):
    """Lower torch.gather to an explicit internal GatherNd IR node.

    PyTorch gather uses one index value per output element and replaces only the
    gather dimension with that value:

        out[p0, ..., pN] = input[p0, ..., index[p], ..., pN]

    Circle GATHER_ND expects the indices input to contain full coordinates in
    the params tensor. This pass constructs those full coordinates and replaces
    aten.gather with circle_custom.gather_nd:

        aten.gather(input, dim, index)
          -> circle_custom.gather_nd(input, full_indices)

    The resulting custom op has GatherNd semantics at the FX level and is
    serialized directly as Circle GATHER_ND.
    """

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph

        modified = False
        for node in list(graph.nodes):
            if not is_target_node(node, torch.ops.aten.gather.default):
                continue

            input_node, dim, index_node, sparse_grad = self._parse_gather_args(node)
            if sparse_grad:
                raise NotYetSupportedError(
                    "torch.gather with sparse_grad=True is not supported."
                )

            input_shape = self._get_static_shape(input_node, "gather input")
            index_shape = self._get_static_shape(index_node, "gather index")

            if len(input_shape) != len(index_shape):
                raise NotYetSupportedError(
                    "torch.gather lowering requires input and index to have the same rank."
                )

            normalized_dim = self._normalize_dim(dim, len(input_shape))
            self._validate_non_gather_shapes(input_shape, index_shape, normalized_dim)

            logger.debug(
                "%s: lowering aten.gather to circle_custom.gather_nd: "
                "input=%r dim=%d index=%r",
                node,
                input_shape,
                normalized_dim,
                index_shape,
            )

            with graph.inserting_before(node):
                int32_index = self._cast_index_to_int32(graph, index_node)
                full_indices = self._build_full_indices(
                    exported_program,
                    graph,
                    node,
                    int32_index,
                    index_shape,
                    normalized_dim,
                )
                gather_nd = create_node(
                    graph,
                    torch.ops.circle_custom.gather_nd,
                    args=(input_node, full_indices),
                    origin=node,
                )
                self._copy_non_value_meta(node, gather_nd)
                set_new_meta_val(gather_nd)

            # gather_nd already owns copied metadata and a recomputed meta["val"].
            # Propagating metadata again would assert because the replacement node
            # is not metadata-empty.
            node.replace_all_uses_with(gather_nd, propagate_meta=False)
            modified = True
            logger.debug("%s is replaced with %s", node.name, gather_nd.name)

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(modified)

    @staticmethod
    def _parse_gather_args(
        node: torch.fx.Node,
    ) -> Tuple[torch.fx.Node, int, torch.fx.Node, bool]:
        """Parse aten.gather arguments from an FX node."""
        args = list(node.args)
        kwargs: Mapping[str, Any] = node.kwargs

        input_node = (
            args[0] if len(args) > 0 else kwargs.get("input", kwargs.get("self"))
        )
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        index_node = args[2] if len(args) > 2 else kwargs.get("index")
        sparse_grad = kwargs.get("sparse_grad", False)

        if not isinstance(input_node, torch.fx.Node):
            raise NotYetSupportedError("torch.gather input must be an FX node.")
        if not isinstance(dim, int):
            raise NotYetSupportedError("torch.gather dim must be a static integer.")
        if not isinstance(index_node, torch.fx.Node):
            raise NotYetSupportedError("torch.gather index must be an FX node.")
        if not isinstance(sparse_grad, bool):
            raise NotYetSupportedError(
                "torch.gather sparse_grad must be a static boolean."
            )

        return input_node, dim, index_node, sparse_grad

    @staticmethod
    def _normalize_dim(dim: int, rank: int) -> int:
        """Return a non-negative dimension index."""
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise NotYetSupportedError("torch.gather dimension is out of range.")
        return dim

    @staticmethod
    def _get_static_shape(node: torch.fx.Node, tensor_name: str) -> List[int]:
        """Return a static tensor shape or reject dynamic shapes."""
        shape = node.meta["val"].shape
        static_shape = []
        for dim in shape:
            if not isinstance(dim, int):
                raise NotYetSupportedError(
                    f"{tensor_name} must have a static shape for GatherNd lowering."
                )
            static_shape.append(dim)
        return static_shape

    @staticmethod
    def _validate_non_gather_shapes(
        input_shape: List[int], index_shape: List[int], gather_dim: int
    ) -> None:
        """Validate PyTorch gather shape constraints used by this lowering."""
        for axis, (input_dim, index_dim) in enumerate(zip(input_shape, index_shape)):
            if axis == gather_dim:
                continue
            if index_dim > input_dim:
                raise NotYetSupportedError(
                    "torch.gather index shape must not exceed input shape for "
                    "non-gather dimensions."
                )

    @staticmethod
    def _cast_index_to_int32(
        graph: torch.fx.Graph, index_node: torch.fx.Node
    ) -> torch.fx.Node:
        """Return an int32 index tensor for Circle GATHER_ND."""
        index_dtype = index_node.meta["val"].dtype
        if index_dtype == torch.int32:
            return index_node

        if index_dtype not in (torch.int64, torch.long):
            raise NotYetSupportedError(
                "torch.gather index must have int32 or int64 dtype."
            )

        int32_index = create_node(
            graph,
            torch.ops.aten._to_copy.default,
            args=(index_node,),
            kwargs={"dtype": torch.int32},
            origin=index_node,
        )
        set_new_meta_val(int32_index)
        return int32_index

    def _build_full_indices(
        self,
        exported_program: ExportedProgram,
        graph: torch.fx.Graph,
        origin: torch.fx.Node,
        index_node: torch.fx.Node,
        index_shape: List[int],
        gather_dim: int,
    ) -> torch.fx.Node:
        """Build a tensor whose last dimension stores full GatherNd coordinates."""
        rank = len(index_shape)
        coordinates = []
        for axis in range(rank):
            if axis == gather_dim:
                coordinate = index_node
            else:
                coordinate = self._add_coordinate_grid(
                    exported_program, index_shape, axis
                )

            unsqueezed = create_node(
                graph,
                torch.ops.aten.unsqueeze.default,
                args=(coordinate, -1),
                origin=origin,
            )
            set_new_meta_val(unsqueezed)
            coordinates.append(unsqueezed)

        full_indices = create_node(
            graph,
            torch.ops.aten.cat.default,
            args=(coordinates, -1),
            origin=origin,
        )
        set_new_meta_val(full_indices)
        return full_indices

    @classmethod
    def _add_coordinate_grid(
        cls, exported_program: ExportedProgram, index_shape: List[int], axis: int
    ) -> torch.fx.Node:
        """Add a constant coordinate grid for one non-gather dimension."""
        rank = len(index_shape)
        view_shape = [1] * rank
        view_shape[axis] = index_shape[axis]

        coordinate = torch.arange(index_shape[axis], dtype=torch.int32)
        coordinate = coordinate.reshape(view_shape).expand(index_shape).clone()
        return cls._add_non_persistent_buffer(
            exported_program, coordinate, "_gather_nd_grid"
        )

    @classmethod
    def _add_non_persistent_buffer(
        cls,
        exported_program: ExportedProgram,
        tensor: torch.Tensor,
        prefix: str,
    ) -> torch.fx.Node:
        """Add a generated tensor as a lifted non-persistent buffer.

        A CONSTANT_TENSOR input is unlifted to a plain tensor attribute by
        ExportedProgram.module(). FX warns when a get_attr node targets such an
        attribute because it is not a module, parameter, or registered buffer.
        Representing compiler-generated coordinate grids as non-persistent
        buffers preserves ExportedProgram invariants and avoids those warnings.
        """
        fake_mode = get_fake_mode(exported_program)
        graph = exported_program.graph
        target = cls._generate_buffer_target(exported_program, prefix)

        first_user_input = get_first_user_input(exported_program)
        if first_user_input is None:
            first_user_input = next(iter(graph.nodes))

        with graph.inserting_before(first_user_input):
            buffer_node = graph.placeholder(target)

        buffer_node.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
        buffer_node.meta["val"].constant = tensor

        exported_program.constants[target] = tensor
        buffer_spec = InputSpec(
            kind=InputKind.BUFFER,
            arg=TensorArgument(name=buffer_node.name),
            target=target,
            persistent=False,
        )

        name_to_spec = {
            spec.arg.name: spec for spec in exported_program.graph_signature.input_specs
        }
        if buffer_node.name in name_to_spec:
            raise RuntimeError(
                f"Input spec already exists for generated buffer {buffer_node.name}."
            )
        name_to_spec[buffer_node.name] = buffer_spec

        new_input_specs = []
        for graph_node in graph.nodes:
            if graph_node.op != "placeholder":
                continue
            if graph_node.name not in name_to_spec:
                raise RuntimeError(
                    f"Missing input spec for placeholder {graph_node.name}."
                )
            new_input_specs.append(name_to_spec[graph_node.name])
        exported_program.graph_signature.input_specs = new_input_specs

        return buffer_node

    @staticmethod
    def _generate_buffer_target(exported_program: ExportedProgram, prefix: str) -> str:
        """Return a unique state target for a generated buffer."""
        used_targets = {
            spec.target
            for spec in exported_program.graph_signature.input_specs
            if isinstance(spec.target, str)
        }
        used_targets.update(exported_program.constants.keys())
        used_targets.update(exported_program.state_dict.keys())
        used_targets.update(
            str(node.target)
            for node in exported_program.graph.nodes
            if node.op == "placeholder"
        )

        index = 0
        while True:
            candidate = f"{prefix}{index}"
            if candidate not in used_targets and not hasattr(
                exported_program.graph_module, candidate
            ):
                return candidate
            index += 1

    @staticmethod
    def _copy_non_value_meta(src: torch.fx.Node, dst: torch.fx.Node) -> None:
        """Copy metadata except the value field, which is recomputed."""
        for key, value in src.meta.items():
            if key == "val":
                continue
            dst.meta[key] = value
