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

from tico.serialize.circle_graph import extract_shape
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import Conv1DArgs


@trace_graph_diff_on_pass
class ConvertConv1dToConv2d(PassBase):
    """
    This pass converts `torch.ops.aten.conv1d` to `torch.ops.aten.conv2d`
    because Circle does not support `conv1d`.

    [before]

            input               weight
           (tensor,dim=3)     (tensor,dim=3)
              |                    |
            conv1d<----------------+
              |
            output
           (tensor,dim=3)

    [after]

            input               weight
           (tensor,dim=3)     (tensor,dim=3)
              |                  |
            unsqueeze         unsqueeze
             (dim=4)            (dim=4)
              |                  |
            conv2d<--------------+
              |
            squeeze
            (dim=3)
              |
            output
           (tensor,dim=3)
    """

    def __init__(self):
        super().__init__()

    def convert(self, exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
        logger = logging.getLogger(__name__)
        modified = False

        graph_module = exported_program.graph_module
        graph = graph_module.graph

        # conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] dilation=1, SymInt groups=1) -> Tensor
        # conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, str padding="valid", SymInt[1] dilation=1, SymInt groups=1) -> Tensor
        args = Conv1DArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        weight = args.weight
        bias = args.bias
        stride = args.stride
        padding = args.padding
        dilation = args.dilation
        groups = args.groups

        input_shape = extract_shape(input)
        if not (len(input_shape) == 3):
            raise NotYetSupportedError(
                f"Only support 3D input tensor: node's input shape: {input_shape}"
            )

        with graph.inserting_after(input):
            input_unsqueeze = create_node(
                graph,
                torch.ops.aten.unsqueeze.default,
                args=(input, 3),
                origin=input,
            )

        with graph.inserting_after(weight):
            weight_unsqueeze = create_node(
                graph,
                torch.ops.aten.unsqueeze.default,
                args=(weight, 3),
                origin=weight,
            )

        with graph.inserting_before(node):
            if isinstance(padding, list):
                conv2d_op = torch.ops.aten.conv2d.default
            elif isinstance(padding, str):
                conv2d_op = torch.ops.aten.conv2d.padding

            conv2d = create_node(
                graph,
                conv2d_op,
                args=(
                    input_unsqueeze,
                    weight_unsqueeze,
                    bias,
                    [*stride, 1],
                    [*padding, 0] if isinstance(padding, list) else padding,
                    [*dilation, 1],
                    groups,
                ),
                kwargs=node.kwargs,
                origin=node,
            )

            conv_out_squeeze = create_node(
                graph,
                torch.ops.aten.squeeze.dims,
                args=(conv2d, [3]),
            )

        node.replace_all_uses_with(conv_out_squeeze, propagate_meta=True)

        logger.debug(f"{node.name} is replaced with {conv2d.name}")
        modified = True
        return modified

    def call(self, exported_program: ExportedProgram) -> PassResult:
        target_conv_op = [torch.ops.aten.conv1d.default, torch.ops.aten.conv1d.padding]

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, target_conv_op):
                continue

            modified |= self.convert(exported_program, node)

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
