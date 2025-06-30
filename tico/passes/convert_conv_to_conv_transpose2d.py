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
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import ConvArgs


@trace_graph_diff_on_pass
class ConvertConvToConvTranspose2D(PassBase):
    """
    Convert 2-D transposed convolutions that were exported as the generic
    `aten.convolution.default` into `aten.conv_transpose2d.input` calls.

    A node N is rewritten if all the following hold
    ------------------------------------------------------
    [1] The seventh argument/kwarg `transposed` is *True*.
    [2] Both input and weight tensors are 4-D and all spatial-parameter
       lists (`stride`, `padding`, `dilation`, `output_padding`)
       have length 2 - i.e. we are in the 2-D case.
    """

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
            stride = args.stride
            padding = args.padding
            dilation = args.dilation
            transposed = args.transposed
            output_padding = args.output_padding

            # condition checks
            # [1]
            if not transposed:
                continue
            # [2]
            two_d = all(
                len(a) == 2 for a in (stride, padding, dilation, output_padding)
            )
            if not two_d:
                continue
            input_shape = extract_shape(input)
            if len(input_shape) != 4:
                continue

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
