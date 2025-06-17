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

from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import (
    add_placeholder,
    create_node,
    get_first_user_input,
    get_torch_buffer_value,
    get_torch_param_value,
    is_torch_buffer,
    is_torch_param,
)
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import NativeBatchNormLegitNoTrainingArgs


@trace_graph_diff_on_pass
class DecomposeBatchNorm(PassBase):
    """
    [BatchNorm]

    The op can be decomposed to a single aten.mul and a single aten.add because mean and
      var are fixed during evaluation.

      W = (weight / sqrt(var + eps))
      B = bias - (mean * weight) / sqrt(var + eps)
      Y = X * W + B

    [before]

        input (tensor, weight, bias, running_mean, running_var, momentum, eps)
          |
      BatchNorm
          |
        output

    [after]

        input
       (tensor)
          |    W
          |   /
         mul
          |   B
          |  /
         add
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
            if not is_target_node(
                node, torch.ops.aten._native_batch_norm_legit_no_training.default
            ):
                continue

            args = NativeBatchNormLegitNoTrainingArgs(*node.args)
            input_ = args.input
            weight = args.weight
            bias = args.bias
            running_mean = args.running_mean
            running_var = args.running_var
            eps = args.eps

            if not running_mean:
                raise NotYetSupportedError(f"running_mean=None is not supported yet")
            if not running_var:
                raise NotYetSupportedError(f"running_var=None is not supported yet")

            """
            Only support the cases generated from torch.nn.BatchNorm2d module,
             for which, let's checks if weight and bias are parameters and 
             running_mean and running_var are buffers.
            """
            if weight and not is_torch_param(weight, exported_program):
                continue
            if bias and not is_torch_param(bias, exported_program):
                continue
            if not is_torch_buffer(running_mean, exported_program):
                continue
            if not is_torch_buffer(running_var, exported_program):
                continue

            input_shape = extract_shape(input_)
            assert len(input_shape) == 4
            C = input_shape[1]

            weight_value = (
                get_torch_param_value(weight, exported_program)
                if weight
                else torch.tensor([1] * C)
            )
            bias_value = (
                get_torch_param_value(bias, exported_program)
                if bias
                else torch.tensor([0] * C)
            )
            mean_value = get_torch_buffer_value(running_mean, exported_program)
            var_value = get_torch_buffer_value(running_var, exported_program)

            assert isinstance(weight_value, torch.Tensor)
            assert isinstance(bias_value, torch.Tensor)
            assert isinstance(mean_value, torch.Tensor)
            assert isinstance(var_value, torch.Tensor)

            assert (
                weight_value.shape
                == bias_value.shape
                == mean_value.shape
                == var_value.shape
            )
            # Calculate constants for mul and add
            mul_const = weight_value / torch.sqrt(var_value + eps)
            add_const = bias_value - (mul_const * mean_value)
            # N, C, H, W
            assert len(mul_const) == len(add_const) == C
            # reshape along with channel dimension
            mul_const = mul_const.view(1, mul_const.shape[0], 1, 1)
            add_const = add_const.view(1, add_const.shape[0], 1, 1)

            # Placeholder nodes must be the first N nodes in the nodes list of a graph.
            # Therefore, insert the newly created placeholders at the start of the node list.
            with exported_program.graph.inserting_before(
                get_first_user_input(exported_program)
            ):
                mul_const_node = add_placeholder(
                    exported_program,
                    mul_const,
                    prefix=f"{node.name}_mul_const",
                )
                add_const_node = add_placeholder(
                    exported_program,
                    add_const,
                    prefix=f"{node.name}_add_const",
                )

            with gm.graph.inserting_before(node):
                mul = create_node(
                    graph,
                    torch.ops.aten.mul.Tensor,
                    args=(input_, mul_const_node),
                    origin=node,
                )
                add = create_node(
                    graph,
                    torch.ops.aten.add.Tensor,
                    args=(mul, add_const_node),
                )
            get_item, *_ = node.users.keys()
            get_item.replace_all_uses_with(add, propagate_meta=True)

            logger.debug(f"{node.name} is decomposed to {mul.name} and {add.name}")
            modified = True

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

        return PassResult(modified)
