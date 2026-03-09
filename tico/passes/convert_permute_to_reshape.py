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
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import PermuteArgs


@trace_graph_diff_on_pass
class ConvertPermuteToReshape(PassBase):
    """
    This pass replaces `aten.permute` to `aten.reshape` when
    the order of output data is exactly same as input data.
    """

    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = enabled

    def call(self, exported_program: ExportedProgram) -> PassResult:
        if not self.enabled:
            return PassResult(False)

        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if not isinstance(node, torch.fx.Node) or not is_target_node(
                node, ops.aten.permute
            ):
                continue

            # Extract permute arguments
            args = PermuteArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

            input = args.input
            dims = args.dims

            input_shape = extract_shape(input)

            # When permute dims with non-1 values have same order,
            # we can replace permute to reshape
            #
            # For example, if
            # - input.shape = [1, x, 1, y]
            # - torch.permute(input, [1, 2, 3, 0])
            # then permute dims 2 and 0 keeps same order for 'x' and 'y'.
            is_same_order = True
            last_dim = -1
            for dim in dims:
                if input_shape[dim] == 1:
                    continue

                if last_dim < dim:
                    last_dim = dim
                else:
                    is_same_order = False
                    break

            if is_same_order == True:
                with graph.inserting_before(node):
                    reshape = create_node(
                        graph,
                        torch.ops.aten.reshape.default,
                        args=(input, [input_shape[dim] for dim in dims]),
                        origin=node,
                    )

                    node.replace_all_uses_with(reshape, propagate_meta=False)
                    modified = True
                    logger.debug(
                        f"{node.name} is replaced with {reshape.name} operators"
                    )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
