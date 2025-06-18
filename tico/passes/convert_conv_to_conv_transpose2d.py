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

import torch
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import ConvArgs


@trace_graph_diff_on_pass
class ConvertConvToConvTranspose2D(PassBase):
    """ """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, torch.ops.aten.convolution.default):
                continue

            args = ConvArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            input = args.input
            padding = args.padding
            groups = args.groups

            if len(padding) != 2:
                continue

            input_shape = extract_shape(input)
            if not (len(input_shape) == 4):
                raise NotYetSupportedError(
                    f"Only support 4D input tensor: node's input shape: {input_shape}"
                )

            if not (groups == 1 or groups == input_shape[1]):
                raise NotYetSupportedError(
                    f"Only support groups=1 or groups=input_channels: node's groups: {groups}, input channels: {input_shape[1]}"
                )

            with graph.inserting_after(node):
                conv2d = create_node(
                    graph,
                    torch.ops.aten.conv_transpose2d.input,
                    args=(
                        args.input,
                        args.weight,
                        args.bias,
                        args.stride,
                        args.padding,
                        args.output_padding,
                        args.groups,
                        args.dilation,
                    ),
                )

            node.replace_all_uses_with(conv2d, propagate_meta=True)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(modified)
