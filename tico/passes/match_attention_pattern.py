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

from torch.export import export, ExportedProgram
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_const_diff_on_pass
from tico.utils.canonicalize import canonicalize
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from tico.passes.patterns.attention import LlamaAttentionWithOutputs

def get_llama_attention_ep():
    mod = LlamaAttentionWithOutputs()
    llama_attention_ep = export(mod, mod.get_example_inputs())
    return llama_attention_ep


def remove_dead_placeholders(exported):
    # 그래프에서 dead placeholder 찾기
    graph = exported.graph_module.graph
    dead_placeholders = []
    for node in graph.nodes:
        if node.op == "placeholder" and len(node.users) == 0:
            dead_placeholders.append(node)

    # Dead placeholder 노드 제거
    for node in dead_placeholders:
        graph.erase_node(node)

    # 그래프 서명 업데이트
    new_input_specs = [
        spec for spec in exported.graph_signature.input_specs
        if spec.arg.name not in {node.name for node in dead_placeholders}
    ]
    exported.graph_signature.input_specs = new_input_specs
    
    graph.eliminate_dead_code()
    exported.graph_module.recompile()
    
    return exported


### Match Attention Pattern

from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from tico.passes.patterns.attention import LlamaAttention

@trace_const_diff_on_pass
class MatchAttentionPattern(PassBase):
    """
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        
        
        pattern_ep = get_llama_attention_ep()
        from tico.config import get_default_config
        pattern_ep = canonicalize(pattern_ep, get_default_config())
        pattern_ep = remove_dead_placeholders(pattern_ep)
        pattern_graph = pattern_ep.graph
        
        subgraph_matcher = SubgraphMatcher(pattern_graph, ignore_literals = True)
        match_result = subgraph_matcher.match(graph)
        print(match_result)
        
        breakpoint()
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
