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

from tico.utils import logging
from tico.utils.graph import add_placeholder
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import AvgPool2dArgs


@trace_graph_diff_on_pass
class LegalizeAvgpool2D(PassBase):
    """
    Let's legalize avg_pool2d with various options.

    Now it supports avg_pool2d (count_include_pad=False)


    [BEFORE]

    input
    |
    avgpool2d (padding = padding, count_include_pad=False)
    |
    out

    [AFTER]

    input
    |
    pad (padding = padding)
    |
    avgpool2d (padding = 'VALID')
    |

    ...

    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        gm = exported_program.graph_module
        graph: torch.fx.Graph = gm.graph
        modified = False

        for node in graph.nodes:
            if node.op != "call_function":
                continue

            if node.target in [
                torch.ops.aten.avg_pool2d.default,
            ]:
                args = AvgPool2dArgs(*node.args, **node.kwargs)
                input = args.input
                kernel_size = args.kernel_size
                stride = args.stride
                padding = args.padding
                count_include_pad = args.count_include_pad

                if args.count_include_pad == True:
                    continue

                assert args.count_include_pad == False

                if isinstance(padding, int):
                    padding = [padding, padding]

                with graph.inserting_before(node):
                    # padding_tensor = add_placeholder(
                    #     exported_program,
                    #     torch.tensor([padding, padding, padding, padding]),
                    #     f"{node.name}_padding",
                    # )

                    # 1. Pad the input tensor
                    x_padded = graph.call_function(
                        torch.ops.aten.constant_pad_nd.default,
                        (input, [padding[0], padding[0], padding[1], padding[1]], 0),
                    )

                    # 2. Perform average pooling (with padding included)
                    pooled = graph.call_function(
                        torch.ops.aten.avg_pool2d.default,
                        (x_padded, kernel_size, stride, [0, 0], count_include_pad),
                    )

                    # 3. Calculate mask with valid pixel count ratio
                    #
                    # ones_padded -> mask
                    # 0 0 0 0     .  .   .   .
                    # 0 1 1 1 ->  .  4/9 6/9 6/9
                    # 0 1 1 1     .  6/9  1   1
                    ones = graph.call_function(
                        torch.ops.aten.full_like.default, (pooled, 1.0)
                    )
                    ones_padded = graph.call_function(
                        torch.ops.aten.constant_pad_nd.default,
                        (ones, [padding[0], padding[0], padding[1], padding[1]], 0),
                    )
                    mask = graph.call_function(
                        torch.ops.aten.avg_pool2d.default,
                        (
                            ones_padded,
                            kernel_size,
                            stride,
                            [0, 0],
                        ),
                    )  # Already padded

                    result = graph.call_function(
                        torch.ops.aten.div.Tensor, (pooled, mask)
                    )

                node.replace_all_uses_with(result, propagate_meta=True)

                modified = True

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

        return PassResult(modified)
