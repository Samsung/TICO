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

from tico.serialize.circle_mapping import extract_shape

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import set_new_meta_val
from tico.utils.validate_args_kwargs import BmmArgs, MatmulArgs


class Converter:  # type: ignore[empty-body]
    def __init__(self):
        super().__init__()

    def match(self, exported_program, node) -> bool:  # type: ignore[empty-body]
        return False

    def convert(self, exported_program, node) -> torch.fx.Node:  # type: ignore[empty-body]
        pass


class MatmulToLinearConverter(Converter):
    def __init__(self):
        super().__init__()

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


class RhsConstMatmulToLinearConverter(MatmulToLinearConverter):
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
        return super().convert(exported_program, node)


class LhsConstSingleBatchBmmToMmConverter(Converter):
    """
    PREPROCESS FOR `LhsConstMatmulToLinearConverter`

    Convert `single-batch lhs-const bmm` to `reshape + matmul`

    lhs(const)  rhs                lhs           rhs
    |[1,s0,s1]  |[1,s1,s2]          |[1,s0,s1]   |[1,s1,s2]
    |           |                reshape         reshape
    |           |                   |[s0,s1]     |[s1,s2]
     --- bmm  ---         -->        ---- mm ----
         |                                |[s0,s2]
         out[1,s0,s2]                     reshape
                                          |[1,s0,s2]
                                          out
    """

    def __init__(self):
        super().__init__()

    def match(self, exported_program, node) -> bool:
        if not node.target == torch.ops.aten.bmm.default:
            return False

        bmm_args = BmmArgs(*node.args, **node.kwargs)
        lhs = bmm_args.input
        rhs = bmm_args.mat2

        # [Cond1] lhs is constant
        if not isinstance(lhs, torch.fx.Node):
            return False
        if not (
            is_lifted_tensor_constant(exported_program, lhs)
            or is_param(exported_program, lhs)
            or is_buffer(exported_program, lhs)
        ):
            return False

        # [Cond2] Both lhs/rhs has batch 1
        lhs_shape = extract_shape(lhs)
        rhs_shape = extract_shape(rhs)

        assert len(lhs_shape) == len(
            rhs_shape
        ), f"Bmm input's ranks must be the same but got {lhs_shape} and {rhs_shape}"

        if not (lhs_shape[0] == rhs_shape[0] == 1):
            return False

        return True

    def convert(self, exported_program, node) -> torch.fx.Node:
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        bmm_args = BmmArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        lhs = bmm_args.input
        rhs = bmm_args.mat2

        with graph.inserting_before(node):
            lhs_reshape = create_node(
                graph,
                torch.ops.aten.reshape.default,
                args=(lhs, list(extract_shape(lhs))[1:]),
            )
            # NOTE: Call a series of `set_new_meta_val` in order to extract shape of `mm_node` below
            set_new_meta_val(lhs_reshape)
            rhs_reshape = create_node(
                graph,
                torch.ops.aten.reshape.default,
                args=(rhs, list(extract_shape(rhs))[1:]),
            )
            set_new_meta_val(rhs_reshape)
            mm_node = create_node(
                graph,
                torch.ops.aten.mm.default,
                args=(lhs_reshape, rhs_reshape),
            )
            set_new_meta_val(mm_node)
            mm_reshape = create_node(
                graph,
                torch.ops.aten.reshape.default,
                args=(mm_node, [1] + list(extract_shape(mm_node))),
            )
            set_new_meta_val(mm_reshape)

            node.replace_all_uses_with(mm_reshape, propagate_meta=False)

        return mm_reshape


class LhsConstMatmulToLinearConverter(MatmulToLinearConverter):
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
        return False

    def convert(self, exported_program, node) -> torch.fx.Node:
        return super().convert(exported_program, node)


@trace_graph_diff_on_pass
class ConvertMatmulToLinear(PassBase):
    """
    This pass converts matmul(partially includes single-batch bmm) to linear selectively

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


    enable_lhs_const: If true, convert matmul where LHS is constant tensor. Default is False.
    enable_rhs_const: If true, convert matmul where RHS is constant tensor. Default is True.
    """

    def __init__(
        self,
        enable_lhs_const: Optional[bool] = False,
        enable_rhs_const: Optional[bool] = True,
    ):
        super().__init__()
        self.converters: List[Converter] = []
        if enable_lhs_const:
            self.converters.append(
                LhsConstSingleBatchBmmToMmConverter()
            )  # Preprocess to support single-batched bmm
            self.converters.append(LhsConstMatmulToLinearConverter())
        if enable_rhs_const:
            self.converters.append(RhsConstMatmulToLinearConverter())

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
