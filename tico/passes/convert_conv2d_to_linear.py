# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
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

from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node

@trace_graph_diff_on_pass
class ConvertConv2dToLinear(PassBase):
    """
    Conv2D를 Linear(addmm/mm)로 변환하는 패스입니다.
    조건: Kernel Size == Input Size (결과가 1x1 인 경우)
    """

    def convert(self, exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
        logger = logging.getLogger(__name__)
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        # aten.conv2d.default(input, weight, bias, stride, padding, dilation, groups)
        input_node = node.args[0]
        weight_node = node.args[1]
        bias_node = node.args[2] if len(node.args) > 2 else None
        
        input_shape = extract_shape(input_node)   # [N, C_in, H, W]
        weight_shape = extract_shape(weight_node) # [C_out, C_in, kH, kW]

        if len(input_shape) != 4 or len(weight_shape) != 4:
            return False

        N, C_in, H, W = input_shape
        C_out, C_in_w, kH, kW = weight_shape

        # 변환 조건: stride나 padding과 무관하게 H == kH 이고 W == kW 이면 
        # 결과가 1x1이 되므로 Linear로 치환 가능
        if H != kH or W != kW:
            logger.debug(f"Skip {node.name}: Kernel size {kH}x{kW} does not match input {H}x{W}")
            return False

        with graph.inserting_before(node):
            # 1. Weight를 [C_out, C_in * H * W]로 변형 후 Transpose
            # Linear 표준 가중치: [In_features, Out_features]
            weight_flattened = create_node(
                graph,
                torch.ops.aten.reshape.default,
                args=(weight_node, [C_out, -1]),
                origin=weight_node,
            )
            weight_t = create_node(
                graph,
                torch.ops.aten.t.default,
                args=(weight_flattened,),
                origin=weight_node,
            )

            # 2. Input을 [N, C_in * H * W]로 변형
            in_features = C_in * H * W
            input_flattened = create_node(
                graph,
                torch.ops.aten.reshape.default,
                args=(input_node, [N, in_features]),
                origin=input_node,
            )

            # 3. Matrix Multiplication 수행
            if bias_node is not None:
                # Linear 결과: [N, C_out]
                mm_result = create_node(
                    graph,
                    torch.ops.aten.addmm.default,
                    args=(bias_node, input_flattened, weight_t),
                    origin=node,
                )
            else:
                mm_result = create_node(
                    graph,
                    torch.ops.aten.mm.default,
                    args=(input_flattened, weight_t),
                    origin=node,
                )

            # 4. 결과 차원 복구: [N, C_out] -> [N, C_out, 1, 1]
            # 원래 Conv2D의 출력이 4D이므로 후속 노드와의 호환성을 위해 unsqueeze 수행
            final_output = create_node(
                graph,
                torch.ops.aten.reshape.default,
                args=(mm_result, [N, C_out, 1, 1]),
                origin=node,
            )

        node.replace_all_uses_with(final_output, propagate_meta=False)
        return True

    def call(self, exported_program: ExportedProgram) -> PassResult:
        target_ops = [torch.ops.aten.conv2d.default]
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        modified = False
        for node in list(graph.nodes):
            if is_target_node(node, target_ops):
                modified |= self.convert(exported_program, node)

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(modified)
