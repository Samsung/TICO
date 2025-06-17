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

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import RepeatArgs


@trace_graph_diff_on_pass
class ConvertRepeatToExpandCopy(PassBase):
    """
    aten.repeat.default is converted to aten.expand_copy.default.
    Why? There isn't CircleNode mapped to repeat.
    so, We convert it using existing aten.expand_copy.default.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, torch.ops.aten.repeat.default):
                continue

            reshape_args = RepeatArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            tensor, repeats = reshape_args.input, reshape_args.repeats

            tensor_shape: List[int] = [int(dim) for dim in tensor.meta["val"].shape]

            # Check if it is possible to convert to aten.expand_copy.default
            cannot_converted = False
            extending_idx = len(repeats) - len(tensor_shape)
            for idx, dim in enumerate(tensor_shape):
                if not (dim == 1 or repeats[extending_idx + idx] == 1):
                    cannot_converted = True
            if cannot_converted:
                continue

            size = []
            for idx, repeats_dim in enumerate(repeats):
                if idx < extending_idx:
                    size.append(repeats_dim)
                else:
                    size.append(repeats_dim * tensor_shape[idx - extending_idx])

            expand_copy_args = (tensor, size)

            with graph.inserting_after(node):
                expand_copy_node = create_node(
                    graph,
                    torch.ops.aten.expand_copy.default,
                    args=expand_copy_args,
                )
                node.replace_all_uses_with(expand_copy_node, propagate_meta=True)

            modified = True
            logger.debug(f"{node.name} is replaced with expand_copy operator")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
