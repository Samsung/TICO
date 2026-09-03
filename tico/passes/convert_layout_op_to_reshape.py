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
from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import SqueezeArgs, UnSqueezeArgs, ViewArgs


@trace_graph_diff_on_pass
class ConvertLayoutOpToReshape(PassBase):
    """
    This pass converts layout transformation Op to reshape if possible.
    This is helpful for further optimization.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        def resolve_static_size(node, explicit_size):
            """Return an all-int reshape size for ``node``.

            Dynamic-shape graphs expose symbolic output dims. Circle RESHAPE
            needs a constant shape tensor, so prefer the explicit ``view`` size
            (which keeps the user's ``-1``) and otherwise infer a single
            symbolic dim as ``-1``. Shapes with several symbolic dims are left
            unchanged and rejected downstream as before.
            """
            out_shape = list(extract_shape(node))
            if all(isinstance(dim, int) for dim in out_shape):
                return out_shape
            if explicit_size is not None and all(
                isinstance(dim, int) for dim in explicit_size
            ):
                return list(explicit_size)
            symbolic = [
                i for i, dim in enumerate(out_shape) if not isinstance(dim, int)
            ]
            if len(symbolic) == 1:
                out_shape[symbolic[0]] = -1
            return out_shape

        def convert(node, input, explicit_size=None):
            out_shape = resolve_static_size(node, explicit_size)

            with graph.inserting_after(node):
                reshape_node = create_node(
                    graph,
                    torch.ops.aten.reshape.default,
                    args=(input, out_shape),
                )
            node.replace_all_uses_with(reshape_node, propagate_meta=True)

            logger.debug(f"{node.name} is replaced with {reshape_node.name}")

        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            if node.target in ops.aten.view:
                view_args = ViewArgs(*node.args, **node.kwargs)
                convert(node, view_args.input, explicit_size=view_args.size)
                modified = True
                continue
            elif node.target in ops.aten.unsqueeze:
                unsqueeze_args = UnSqueezeArgs(*node.args, **node.kwargs)
                convert(node, unsqueeze_args.input)
                modified = True
                continue
            elif node.target in ops.aten.squeeze:
                squeeze_args = SqueezeArgs(*node.args, **node.kwargs)
                convert(node, squeeze_args.input)
                modified = True
                continue

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
