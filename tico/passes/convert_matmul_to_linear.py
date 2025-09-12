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

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import MatmulArgs


class Converter:  # type: ignore[empty-body]
    def __init__(self):
        super().__init__()

    def match(self, exported_program, node) -> bool:  # type: ignore[empty-body]
        return False

    def convert(self, exported_program, node) -> torch.fx.Node:  # type: ignore[empty-body]
        pass


class ConvertRhsConstMatmulToLinear(Converter):
    def __init__(self):
        super().__init__()

    def match(self, exported_program, node) -> bool:
        if not node.target == torch.ops.aten.mm.default:
            return False

        mm_args = MatmulArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        rhs = mm_args.other
        if isinstance(rhs, torch.fx.Node):
            if is_lifted_tensor_constant(exported_program, rhs):
                return True
            elif is_param(exported_program, rhs):
                return True
            elif is_buffer(exported_program, rhs):
                return True
            else:
                return False
        return False

    def convert(self, exported_program, node) -> torch.fx.Node:
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        mm_args = MatmulArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        lhs = mm_args.input
        rhs = mm_args.other

        with graph.inserting_before(node):
            transpose_node = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(rhs, [1, 0]),
            )
            fc_node = create_node(
                graph,
                torch.ops.aten.linear.default,
                args=(lhs, transpose_node),
            )
            node.replace_all_uses_with(fc_node, propagate_meta=True)

        return fc_node


class ConvertLhsConstMatmulToLinear(Converter):
    def __init__(self):
        super().__init__()

    def match(self, exported_program, node) -> bool:
        if not node.target == torch.ops.aten.mm.default:
            return False

        mm_args = MatmulArgs(*node.args, **node.kwargs)
        lhs = mm_args.input
        if isinstance(lhs, torch.fx.Node):
            if is_lifted_tensor_constant(exported_program, lhs):
                return True
            elif is_param(exported_program, lhs):
                return True
            elif is_buffer(exported_program, lhs):
                return True
            else:
                return False

    def convert(self, exported_program, node) -> torch.fx.Node:
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        mm_args = MatmulArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        lhs = mm_args.input
        rhs = mm_args.other

        with graph.inserting_before(node):
            transpose_node = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(rhs, [1, 0]),
            )
            fc_node = create_node(
                graph,
                torch.ops.aten.linear.default,
                args=(lhs, transpose_node),
            )
            node.replace_all_uses_with(fc_node, propagate_meta=True)

        return fc_node


@trace_graph_diff_on_pass
class ConvertMatmulToLinear(PassBase):
    """
    This pass converts matmul to linear selectively

    How to select between `matmul` and `linear`?

    * Linear has better quantization accuracy (NPU backend)
        Due to ONE compiler's quantization policy;
        FullyConnected(=Linear) uses per-channel quantization for weight and per-tensor for input.
        BatchMatmul(=matmul) uses per-tensor quantization for both rhs and lhs.

    * Matmul to Linear requires Transpose, which may harm latency
        When RHS is constant, addtional transpose can be folded.

    [RHS non-const case]
    Constant folding cannot be performed.

    lhs         rhs (non-const)
    |           |
    |           transpose
    |           |
     -- linear --
         |
         out

    [RHS const case]
    Constant folding can be performed to

    lhs         rhs (const)         lh          rhs (folded const)
    |           |                   |           |
    |           transpose           |           |
    |           |                   |           |
     -- linear --         -->        -- linear --
         |                                |
         out                              out


    enable_lhs_const: If true, also convert matmul where LHS is constant tensor. Default is False.
    enable_rhs_const: If true, also convert matmul where RHS is constant tensor. Default is True.
    """

    def __init__(
        self,
        enable_lhs_const: Optional[bool] = False,
        enable_rhs_const: Optional[bool] = True,
    ):
        super().__init__()
        self.converters: List[Converter] = []
        if enable_lhs_const:
            self.converters.append(ConvertLhsConstMatmulToLinear())
        if enable_rhs_const:
            self.converters.append(ConvertRhsConstMatmulToLinear())

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            for converter in self.converters:
                if not converter.match(exported_program, node):
                    continue

                new_node = converter.convert(exported_program, node)
                modified = True
                logger.debug(
                    f"{node.name} is replaced with {new_node.name} operator (permute + linear)"
                )
                continue

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
