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

from tico.passes import ops
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node


@trace_graph_diff_on_pass
class RemoveNop(PassBase):
    """
    Let's remove noops by propagation.
    """

    target_ops = (
        [
            torch.ops.prims.view_of.default,
        ]
        + ops.aten.alias
        + ops.aten.clone
        + ops.aten.detach
        + [torch.ops.aten.lift_fresh_copy.default]
    )

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, RemoveNop.target_ops):
                continue

            # TODO Consider memory format
            if node.target in ops.aten.clone and "memory_format" in node.kwargs:
                if node.kwargs["memory_format"] not in [
                    torch.preserve_format,
                    # Converting non-contiguous layout to contiguous only updates
                    # strides of tensor. This is not visible on circle, so we can
                    # safely ignore this operation.
                    torch.contiguous_format,
                ]:
                    continue

            assert len(node.args) == 1

            src = node.args[0]
            assert isinstance(src, torch.fx.Node)

            with graph.inserting_after(node):
                node.replace_all_uses_with(src, propagate_meta=False)

            modified = True
            logger.debug(f"{node.name} is replaced with {src}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
