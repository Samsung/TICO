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
from tico.utils.errors import InvalidArgumentError, NotYetSupportedError
from tico.utils.graph import add_placeholder, create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import Conv2DArgs


@trace_graph_diff_on_pass
class DecomposeGroupedConv2d(PassBase):
    """
    This pass decomposes grouped Conv2d operator as multiple Conv2d operator whose groups=1.

    Grouped Conv2d denotes a Conv2d operator whose `groups` argument is not equal to input channels nor 1.

    [before]

        input       weight       bias
          |           |           |
          +-----------+-----------+
                      |
                    Conv2d (groups != IN_CHANNEL && groups != 1)
                      |
                    output

    [after]

    The below `slice` operators slice the input tensor, weight and bias along the channel axis by the number of `groups`.
    In addition, the numbered input, weight and bias denotes sliced input tensor, weight and bias respectively.

        input
          |       weight
        slice       |        bias
          |       slice       |
          |         |       slice
          |         |         |
          +---------------------------+---------------------------+
          |         |         |       |                           |
          |         +---------------------------+---------------------------+
          |         |         |       |         |                 |         |
          |         |         +---------------------------+---------------------------+
          |         |         |       |         |         |       |         |         |
        input_1     |         |      ...        |         |     input_N     |         |
          |      weight_1     |       |        ...        |       |      weight_N     |
          |         |       bias_1    |         |        ...      |         |       bias_N
          +---------+---------+       +---------+---------+       +---------+---------+
                    |                           |                           |
                Conv2d_1                       ...                      Conv2d_N
                    |                           |                           |
                    +---------------------------+---------------------------+
                                                |
                                              concat
                                                |
                                              output
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        gm = exported_program.graph_module
        graph: torch.fx.Graph = gm.graph
        modified = False

        for node in graph.nodes:
            if not is_target_node(node, ops.aten.conv2d):
                continue

            args = Conv2DArgs(*node.args)
            input_ = args.input
            weight = args.weight
            bias = args.bias
            stride = args.stride
            padding = args.padding
            dilation = args.dilation
            groups = args.groups

            input_shape = extract_shape(input_)
            if not len(input_shape) == 4:
                raise NotYetSupportedError(
                    f"Only support 4D input tensor: node's input shape: {input_shape}"
                )

            in_channels = input_shape[1]
            if groups == 1 or groups == in_channels:
                continue
            assert (
                in_channels % groups == 0
            ), f"in_channels should be divisible by groups: in_channels: {in_channels}, groups: {groups}"

            output_shape = extract_shape(node)
            assert len(output_shape) == 4, len(output_shape)

            out_channels = output_shape[1]
            assert (
                out_channels % groups == 0
            ), f"out_channels should be divisible by groups: out_channels: {out_channels}, groups: {groups}"

            weight_shape = extract_shape(weight)
            assert len(weight_shape) == 4, len(weight_shape)
            assert (
                weight_shape[0] == out_channels
            ), f"weight shape[0]: {weight_shape[0]}, out channels: {out_channels}"
            assert (
                weight_shape[1] == in_channels // groups
            ), f"weight shape[1]: {weight_shape[1]}, in channels: {in_channels}"

            if bias is not None:
                bias_shape = extract_shape(bias)
                assert (
                    bias_shape[0] == out_channels
                ), f"bias shape[0]: {bias_shape[0]}, out channels: {out_channels}"
            else:  # Make dummy bias tensor
                bias = add_placeholder(
                    exported_program, torch.zeros(out_channels), "bias"
                )

            group_size = in_channels // groups
            out_group_size = out_channels // groups

            with gm.graph.inserting_before(node):
                conv2d_op = None
                if isinstance(padding, list) and all(
                    isinstance(element, int) for element in padding
                ):
                    conv2d_op = torch.ops.aten.conv2d.default
                elif isinstance(padding, str):
                    conv2d_op = torch.ops.aten.conv2d.padding
                else:
                    raise InvalidArgumentError(
                        f"Unsupported padding type: {padding}"
                    )  # Unreachable to here

                conv2d_tensors = []
                for i in range(groups):
                    sliced_input = create_node(
                        graph,
                        torch.ops.aten.slice.Tensor,
                        (input_, 1, group_size * i, group_size * (i + 1), 1),
                        origin=node,
                    )
                    sliced_weight = create_node(
                        graph,
                        torch.ops.aten.slice.Tensor,
                        (weight, 0, out_group_size * i, out_group_size * (i + 1), 1),
                        origin=node,
                    )
                    sliced_bias = create_node(
                        graph,
                        torch.ops.aten.slice.Tensor,
                        (bias, 0, out_group_size * i, out_group_size * (i + 1), 1),
                        origin=node,
                    )
                    conv2d_tensor = create_node(
                        graph,
                        conv2d_op,
                        (
                            sliced_input,
                            sliced_weight,
                            sliced_bias,
                            stride,
                            padding,
                            dilation,
                            1,
                        ),
                        origin=node,
                    )
                    conv2d_tensors.append(conv2d_tensor)

                concat_output = create_node(
                    graph, torch.ops.aten.cat.default, (conv2d_tensors, 1)
                )

                node.replace_all_uses_with(concat_output, propagate_meta=True)

            modified = True
            logger.debug(
                f"{node.name} is replaced with groups of conv2d: The number of groups: {groups}, groups size: {group_size}"
            )

        graph.eliminate_dead_code()
        gm.recompile()
        return PassResult(modified)
