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

from torch.export import ExportedProgram

from tico.passes import ops
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import CatArgs


@trace_graph_diff_on_pass
class MergeConsecutiveCat(PassBase):
    """
    This pass merges consecutive `aten.cat` operators when they can be merged into single operator.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for cat in graph.nodes:
            if not is_target_node(cat, ops.aten.cat):
                continue

            args = CatArgs(*cat.args, **cat.kwargs)  # type: ignore[arg-type]
            inputs = args.tensors
            dim = args.dim

            new_inputs = []
            for prev_cat in inputs:
                new_inputs.append(prev_cat)
                if not prev_cat.op == "call_function":
                    continue

                if not prev_cat.target in ops.aten.cat:
                    continue

                prev_args = CatArgs(*prev_cat.args, **prev_cat.kwargs)  # type: ignore[arg-type]
                prev_inputs = prev_args.tensors
                prev_dim = prev_args.dim

                if not prev_dim == dim:
                    continue

                new_inputs.pop()
                for prev_input in prev_inputs:
                    new_inputs.append(prev_input)

            if len(new_inputs) > len(inputs):
                cat.args = (new_inputs, dim)

                modified = True
                logger.debug(
                    f"Consecutive cat nodes before {cat.name} are merged into {cat.name}"
                )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
