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
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import AddTensorArgs


@trace_graph_diff_on_pass
class LegalizeCausalMaskValue(PassBase):
    """
    This pass replaces occurrences of -inf in attention masks with a large negative finite value (e.g., -120) to ensure numerical stability in computations, particularly in softmax operations.

    This pass can be turned enable only when
        1. The model will be quantized later (e.g., by circle-quantizer).
        2. Softmax kernel of our backend does not support masking.
        3. `Add with -inf` is used only for masking.
    """

    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = enabled

    def call(self, exported_program: ExportedProgram) -> PassResult:
        if not self.enabled:
            return PassResult(False)

        new_mask = -120  # Make it configurable
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, ops.aten.add):
                continue

            args = AddTensorArgs(*node.args, **node.kwargs)
            input = args.input
            other = args.other

            if (
                isinstance(input, torch.fx.Node)
                and input.name
                in exported_program.graph_signature.lifted_tensor_constants
            ):
                mask_node = input
            elif (
                isinstance(other, torch.fx.Node)
                and other.name
                in exported_program.graph_signature.lifted_tensor_constants
            ):
                mask_node = other
            else:
                continue

            mask_node_name = (
                exported_program.graph_signature.inputs_to_lifted_tensor_constants[
                    mask_node.name
                ]
            )
            mask_data = exported_program.constants[mask_node_name]

            # WHY Use -1.e+38, not -float('inf') or torch.finfo(torch.float32).min?
            #
            # torch.finfo(torch.float32).min is -3.4028234663852886e+38 but it changes while processed in const prop or other passes.
            # Therefore, use a rounded value and compare to know it's very large negative number.
            fp32_minus_inf_rounded = -1.0e38
            if torch.all(
                torch.logical_or(mask_data == 0, mask_data < fp32_minus_inf_rounded)
            ):
                exported_program.constants[mask_node_name] = torch.where(
                    mask_data < fp32_minus_inf_rounded,
                    torch.tensor(new_mask, dtype=mask_data.dtype),
                    mask_data,
                )

            modified = False  # To run only once
            logger.debug(
                f"{mask_node.name}'s mask data are changed from '-inf' to {new_mask}"
            )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
