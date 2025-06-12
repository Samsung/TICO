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
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import LinearArgs, MulScalarArgs, MulTensorArgs

_SCALAR_MUL_TARGETS = [
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
]

_SHAPE_ONLY_TARGETS = [
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
]

_LINEAR_TARGET = torch.ops.aten.linear.default


# TODO Make it a util function
def _get_constant(ep: ExportedProgram, node: torch.fx.Node):
    """
    Get constants from ExportedProgram.
    """
    if is_lifted_tensor_constant(ep, node):
        return get_lifted_tensor_constant(ep, node)  # type: ignore[assignment]
    elif is_param(ep, node):
        return get_param(ep, node)  # type: ignore[assignment]
    elif is_buffer(ep, node):
        return get_buffer(ep, node)  # type: ignore[assignment]
    else:
        return None


@trace_graph_diff_on_pass
class FuseScalarMulIntoLinear(PassBase):
    """
    Fold `aten.mul(scalar)` into a preceding `aten.linear`.

    The pattern handled is:

        x → linear(W, B) → [shape-only ops]* → mul(scalar) → ..

    If the scalar is a constant, we can pre-multiply **W** and **B**
    once during compilation, delete the `mul`, and re-wire all
    uses to the linear's (or last shape-only op's) output.

    Notes
    -----
    * Only single-use, shape-only ops are allowed between the linear and
      the mul.
    * The pass mutates the underlying *parameter* tensors in-place
      (`weight.mul_()` and, if present, `bias.mul_()`).  Downstream code
      must not rely on the original parameter values after the pass.
    """

    def call(self, ep: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        gm = ep.graph_module
        graph = gm.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, _SCALAR_MUL_TARGETS):
                continue

            if node.target == torch.ops.aten.mul.Tensor:
                mul_args = MulTensorArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
                mul_input = mul_args.input
                scalar_node = mul_args.other
            elif node.target == torch.ops.aten.mul.Scalar:
                mul_args = MulScalarArgs(*node.args, **node.kwargs)  # type: ignore[arg-type, assignment]
                mul_input = mul_args.input
                scalar_node = mul_args.other
            else:
                raise RuntimeError("Invalid target.")

            if not isinstance(mul_input, torch.fx.Node) or isinstance(
                scalar_node, torch.fx.Node
            ):
                continue
            if isinstance(scalar_node, torch.Tensor) and len(scalar_node.shape) != 0:
                continue

            # Walk backwards through shape-only single-use ops
            shape_chain = []
            cur = mul_input
            while is_target_node(cur, _SHAPE_ONLY_TARGETS) and len(cur.users) == 1:
                shape_chain.append(cur)
                cur = cur.args[0]  # type: ignore[assignment]

            # Root must be aten.linear and single-use
            if not is_target_node(cur, _LINEAR_TARGET) or len(cur.users) != 1:
                continue

            # Get the scalar value
            assert isinstance(scalar_node, (int, float, torch.Tensor))
            scalar_val = (
                scalar_node.item()
                if isinstance(scalar_node, torch.Tensor)
                else scalar_node
            )
            if scalar_val == 1:
                continue

            # Mutate parameters
            linear_args = LinearArgs(*cur.args, **cur.kwargs)  # type: ignore[arg-type]
            linear_weight, linear_bias = linear_args.weight, linear_args.bias

            weight_data = _get_constant(ep, linear_weight)
            assert weight_data is not None and isinstance(weight_data, torch.Tensor)
            # In-place operation needs no-grad.
            with torch.no_grad():
                weight_data.mul_(scalar_val)
            if linear_bias is not None:
                bias_data = _get_constant(ep, linear_bias)
                assert isinstance(bias_data, torch.Tensor)
                with torch.no_grad():
                    bias_data.mul_(scalar_val)

            # Bypass the mul node entirely
            node.replace_all_uses_with(mul_input)
            logger.debug(f"{node.name} is fused to {cur.name}.")
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            gm.recompile()

        return PassResult(modified)
