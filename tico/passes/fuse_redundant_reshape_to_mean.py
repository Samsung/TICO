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
from torch.export import ExportedProgram
from torch.utils import _pytree as pytree

from tico.passes import ops
from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node


@trace_graph_diff_on_pass
class FuseRedundantReshapeToMean(PassBase):
    """
    This pass removes redundant `aten.reshape` operators that can be fused to `aten.mean` with `keep_dims`.

    Shape(aten.reshape(aten.mean(input))) == Shape(aten.mean(input, keep_dims=True))
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, torch.ops.aten.mean.dim):
                continue

            # If mean is being used in other nodes, do not fuse it.
            if len(node.users) != 1:
                continue

            user_node = next(iter(node.users))
            if not is_target_node(user_node, ops.aten.reshape):
                continue

            mean_args, mean_kwargs = pytree.tree_map_only(
                torch.fx.Node,
                lambda n: n.meta["val"],
                (node.args, node.kwargs),
            )
            # Signature of aten.mean.dim is as follows.
            # mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
            # `keepdim` in `node.kwargs` is moved to `node.args` in `run_decompositions`.
            # `dtype` in `node.kwargs` is not moved
            assert len(mean_args) == 3 or len(mean_args) == 2  # keepdim exists or not
            assert len(mean_kwargs) <= 1  # dtype exists or not
            fused_mean_args = mean_args
            keep_dims = True
            if len(mean_args) == 2:
                fused_mean_args += (keep_dims,)

            fused_val = node.target(*fused_mean_args, **mean_kwargs)

            # Check if both shapes are same
            # 1. Shape(aten.reshape(aten.mean))
            # 2. Shape(aten.mean(keep_dims=True))
            if fused_val.size() != extract_shape(user_node):
                continue

            # update args
            if len(mean_args) == 2:
                updated_args = node.args + (keep_dims,)
            elif len(mean_args) == 3:
                updated_args = node.args
            node.args = updated_args
            node.meta["val"] = fused_val
            user_node.replace_all_uses_with(node, propagate_meta=False)

            modified = True
            logger.debug(f"{user_node.name} is replaced with {node.name}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
