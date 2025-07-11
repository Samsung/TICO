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
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node, set_new_meta_val
from tico.utils.validate_args_kwargs import ClampArgs


@trace_graph_diff_on_pass
class CastClampMixedTypeArgs(PassBase):
    """
    This pass ensures consistent dtypes for clamp operations by:
    1. Converting min/max arguments to match output dtype when provided
    2. Inserting cast operations when input dtype differs from output dtype

    Behavior Examples:
    - When input dtype differs from output:
        Inserts _to_copy operation to convert input
    - When min/max dtype differs from output:
        Converts min/max values to output dtype

    [before, if input dtype is different from output dtype]

            input               min(or max)
           (dtype=int)         (dtype=float)
              |                    |
            clamp <----------------+
              |
            output
           (dtype=float)

    [after]

            input             min(or max)
           (dtype=int)       (dtype=float)
              |                  |
            cast                 |
          (in=int, out=float)    |
              |                  |
            clamp <--------------+
              |
            output
           (dtype=float)
    """

    def __init__(self):
        super().__init__()

    def convert(self, exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
        logger = logging.getLogger(__name__)
        modified = False

        graph_module = exported_program.graph_module
        graph = graph_module.graph

        # clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
        args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        input = args.input
        min = args.min
        max = args.max

        if "val" not in input.meta or "val" not in node.meta:
            logger.warning(f"Missing meta['val'] for node {node.name}")
            return False

        input_dtype = input.meta["val"].dtype
        output_dtype = node.meta["val"].dtype

        def _convert_arg(arg, arg_name: str):
            if arg is None:
                return False

            arg_dtype = torch.tensor(arg).dtype
            arg_idx = node.args.index(arg)
            if arg_dtype != output_dtype:
                if output_dtype == torch.float32 or output_dtype == torch.float:
                    arg = float(arg)
                else:
                    arg = int(arg)
                node.update_arg(arg_idx, arg)
                logger.debug(
                    f"Converted {arg_name} value from {arg_dtype} to {output_dtype} for clamp operation at {node.name}"
                )
                return True
            return False

        modified |= _convert_arg(min, "min")
        modified |= _convert_arg(max, "max")

        if input_dtype != output_dtype:
            logger.debug(
                f"Inserting cast from {input_dtype} to {output_dtype} for input {input.name}"
            )
            with graph.inserting_after(input):
                to_copy = create_node(
                    graph,
                    torch.ops.aten._to_copy.default,
                    (input,),
                    {"dtype": output_dtype},
                    origin=input,
                )
                set_new_meta_val(to_copy)
                node.update_arg(node.args.index(input), to_copy)

            modified = True

        return modified

    def call(self, exported_program: ExportedProgram) -> PassResult:
        target_op = [torch.ops.aten.clamp.default, torch.ops.aten.clamp.Tensor]

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, target_op):
                continue

            modified |= self.convert(exported_program, node)

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
