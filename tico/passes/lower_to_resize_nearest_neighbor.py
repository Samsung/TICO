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

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import IndexArgs, UpsampleNearest2DVecArgs


@trace_graph_diff_on_pass
class LowerToResizeNearestNeighbor(PassBase):
    """
    This pass lowers `aten.index` and `aten.upsample_nearest2d.vec` to `circle_custom.resize_nearest_neighbor` when it is possible.

    Until torch 2.7, `torch.nn.functional.interpolate` is converted to `aten.index` op.
        [BEFORE PASS]
        input - aten.index - output

        [AFTER PASS]
        input - aten.permute(NCHW_to_NHWC) - circle_custom.resize_nearest_neighbor - aten.permute(NHWC_to_NCHW) - output

    Since torch 2.8, `torch.nn.functional.interpolate` is converted to aten.upsample_nearest2d.vec` op.
        [BEFORE PASS]
        input - aten.upsample_nearest2d.vec - output

        [AFTER PASS]
        input - aten.permute(NCHW_to_NHWC) - circle_custom.resize_nearest_neighbor - aten.permute(NHWC_to_NCHW) - output
    """

    def __init__(self):
        super().__init__()

    def convert_index_to_resize_nearest_neighbor(
        self, exported_program, node
    ) -> Optional[torch.fx.Node]:
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        args = IndexArgs(*node.args, **node.kwargs)
        input_tensor = args.input
        indices = args.indices

        # Only support 4-D tensor
        if len(indices) != 4:
            return None
        # indices = [None, None, H index, W index]
        N, C, H, W = indices
        if N != None or C != None:
            return None
        if not isinstance(H, torch.fx.Node):
            return None
        if not isinstance(W, torch.fx.Node):
            return None
        constants_dict = exported_program.constants
        if (H.name not in constants_dict) or (W.name not in constants_dict):
            return None
        H_index, W_index = constants_dict[H.name], constants_dict[W.name]
        input_tensor_shape = extract_shape(input_tensor)
        input_tensor_H, input_tensor_W = (
            input_tensor_shape[2],
            input_tensor_shape[3],
        )
        if H_index.size()[0] % input_tensor_H != 0:
            return None
        scale_factor = int(H_index.size()[0] / input_tensor_H)
        # H and W should be resized with same ratio.
        if scale_factor != W_index.size()[0] / input_tensor_W:
            return None
        expected_H_index = []
        expected_W_index = []
        # Please refer to above `_prop_tensor_constant1` constant in the example.
        for i in range(input_tensor_H):
            expected_H_index += [[i]] * scale_factor
        # Please refer to above `_prop_tensor_constant0` constant in the example.
        for i in range(input_tensor_W):
            expected_W_index += [i] * scale_factor
        if not torch.all(
            torch.eq(H_index, torch.tensor(expected_H_index))
        ) or not torch.all(torch.eq(W_index, torch.tensor(expected_W_index))):
            return None
        expected_shape = [
            input_tensor_shape[0],
            input_tensor_shape[1],
            len(expected_H_index),
            len(expected_W_index),
        ]
        assert expected_shape == list(extract_shape(node))

        with graph.inserting_before(node):
            nchw_to_nhwc = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(input_tensor, [0, 2, 3, 1]),
                origin=input_tensor,
            )
            resize_nearest_neighbor = create_node(
                graph,
                torch.ops.circle_custom.resize_nearest_neighbor,
                args=(nchw_to_nhwc, [len(expected_H_index), len(expected_W_index)]),
                origin=node,
            )
            nhwc_to_nchw = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(resize_nearest_neighbor, [0, 3, 1, 2]),
            )
            node.replace_all_uses_with(nhwc_to_nchw, propagate_meta=True)

        return resize_nearest_neighbor

    def convert_upsample_nearest2d_to_resize_nearest_neighbor(
        self, exported_program, node
    ) -> Optional[torch.fx.Node]:
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        args = UpsampleNearest2DVecArgs(*node.args, **node.kwargs)
        input_tensor = args.input
        output_size = args.output_size
        scale_factors = args.scale_factors

        input_tensor_shape = extract_shape(input_tensor)
        input_tensor_H, input_tensor_W = (
            input_tensor_shape[2],
            input_tensor_shape[3],
        )

        if output_size is not None:
            raise NotYetSupportedError("output_size is not supported yet")

        if scale_factors is None:
            raise NotYetSupportedError("scale_factors is None")
        # TODO Support output_size case. Currently only scale_factors case is supported.

        assert (
            isinstance(scale_factors[0], float)
            and isinstance(scale_factors[1], float)
            and scale_factors[0] > 0
            and scale_factors[1] > 0
        )

        def close_enough(x, y, epsilon=1e-10):
            return abs(x - y) < epsilon

        expected_H = int(input_tensor_H * scale_factors[0])
        if not close_enough(expected_H, input_tensor_H * scale_factors[0]):
            raise NotYetSupportedError(
                f"Cannot support input_tensor_H ({input_tensor_H}) with scaling factor ({scale_factors[0]})"
            )

        expected_W = int(input_tensor_W * scale_factors[1])
        if not close_enough(expected_W, input_tensor_W * scale_factors[1]):
            raise NotYetSupportedError(
                f"Cannot support input_tensor_W ({input_tensor_W}) with scaling factor ({scale_factors[1]})"
            )

        with graph.inserting_before(node):
            nchw_to_nhwc = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(input_tensor, [0, 2, 3, 1]),
                origin=input_tensor,
            )
            resize_nearest_neighbor = create_node(
                graph,
                torch.ops.circle_custom.resize_nearest_neighbor,
                args=(nchw_to_nhwc, [expected_H, expected_W]),
                origin=node,
            )
            nhwc_to_nchw = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(resize_nearest_neighbor, [0, 3, 1, 2]),
            )
            node.replace_all_uses_with(nhwc_to_nchw, propagate_meta=True)
            return resize_nearest_neighbor

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        modified = False
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        for node in graph.nodes:
            if not is_target_node(
                node,
                [torch.ops.aten.index.Tensor, torch.ops.aten.upsample_nearest2d.vec],
            ):
                continue

            resize_nearest_neighbor = None
            if node.target == torch.ops.aten.index.Tensor:
                resize_nearest_neighbor = self.convert_index_to_resize_nearest_neighbor(
                    exported_program, node
                )
            elif node.target == torch.ops.aten.upsample_nearest2d.vec:
                resize_nearest_neighbor = (
                    self.convert_upsample_nearest2d_to_resize_nearest_neighbor(
                        exported_program, node
                    )
                )

            if resize_nearest_neighbor:
                modified = True
                logger.debug(
                    f"{node.name} is replaced with {resize_nearest_neighbor.name} operator"
                )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
