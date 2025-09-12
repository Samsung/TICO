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

from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import set_new_meta_val


@trace_graph_diff_on_pass
class FillMetaVal(PassBase):
    """
    Let's set new meta['val'] for nodes which don't have meta['val']
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram, graph_module) -> PassResult:
        logger = logging.getLogger(__name__)
        graph = graph_module.graph
        modified = False
        # To make sure graph is topologically sorted
        graph.lint()
        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            if hasattr(node, "meta") and "val" in node.meta:
                continue

            set_new_meta_val(node)

            modified = True

            logger.debug(f"{node.name} has new meta values.")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
