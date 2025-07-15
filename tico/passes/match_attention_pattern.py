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
import operator

import torch
from torch.export import export, ExportedProgram
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

from tico.config import CompileConfigBase

from tico.passes.patterns.attention import LlamaAttentionWithKVCacheWithOutputs
from tico.utils import logging
from tico.utils.canonicalize import canonicalize, remove_dead_placeholders
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_const_diff_on_pass
from tico.utils.utils import set_new_meta_val


@trace_const_diff_on_pass
class MatchAttentionPattern(PassBase):
    """A pass that matches and replaces attention patterns in the graph.

    This pass identifies attention patterns in the graph and replaces them with
    optimized implementations using custom operators.
    """

    def __init__(self, config: CompileConfigBase):
        super().__init__()
        self.config = config

    def get_pattern_graph(self):
        match_llama_attention_config = self.config.get("match_llama_attention_config")
        mod = LlamaAttentionWithKVCacheWithOutputs(config=match_llama_attention_config)
        attention_ep = export(mod, mod.get_example_inputs())
        attention_ep = canonicalize(attention_ep, self.config)
        attention_ep = remove_dead_placeholders(attention_ep)
        return attention_ep.graph

    def call(self, exported_program: ExportedProgram) -> PassResult:
        if not self.config.match_llama_attention:
            return PassResult(False)

        logger = logging.getLogger(__name__)
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        subgraph_matcher = SubgraphMatcher(
            self.get_pattern_graph(), ignore_literals=True
        )
        match_result = subgraph_matcher.match(graph)
        for match in match_result:

            def find_last_node(random_nodes):
                # Map node names to their indices in the ordered list
                ordered_list = [i for i in graph.nodes]
                index_map = {name: idx for idx, name in enumerate(ordered_list)}
                # Filter nodes that actually exist in the graph
                filtered_nodes = [node for node in random_nodes if node in index_map]
                if not filtered_nodes:
                    return None  # Return None if no matching nodes found
                # Return the node with the highest index
                return max(filtered_nodes, key=lambda x: index_map[x])

            last_placeholder = find_last_node(match.placeholder_nodes)
            with graph.inserting_after(last_placeholder):
                attn_node = create_node(
                    graph,
                    torch.ops.circle_custom.llama_attention_with_kvcache,
                    args=tuple(match.placeholder_nodes),
                )
                set_new_meta_val(attn_node)

            attn_res, key_out, value_out = match.returning_nodes

            with graph.inserting_after(attn_node):
                with graph.inserting_before(attn_res):
                    attn_node_out0 = create_node(
                        graph,
                        operator.getitem,
                        args=(attn_node, 0),
                        origin=attn_res,
                    )
                with graph.inserting_before(key_out):
                    attn_node_out1 = create_node(
                        graph,
                        operator.getitem,
                        args=(attn_node, 1),
                        origin=key_out,
                    )
                with graph.inserting_before(value_out):
                    attn_node_out2 = create_node(
                        graph,
                        operator.getitem,
                        args=(attn_node, 2),
                        origin=value_out,
                    )

            attn_res.replace_all_uses_with(attn_node_out0, propagate_meta=False)
            key_out.replace_all_uses_with(attn_node_out1, propagate_meta=False)
            value_out.replace_all_uses_with(attn_node_out2, propagate_meta=False)

            logger.debug(f"Replaced attention pattern in the graph. {attn_node}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
