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

import math
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import NativeGroupNormArgs, NativeLayerNormArgs


@trace_graph_diff_on_pass
class DecomposeGroupNorm(PassBase):
    """
    This pass decomposes Group normalization operators.

    LayerNorm is group=1 Group normalization.

    [LayerNorm, GroupNorm]

      Two normalzations result in same nodes but have different normalization shapes.

      [before]

              input (tensor, normalized_shape, weight, bias, eps)
                |
     NativeLayerNorm or GroupNorm
                |
              output

      [after]

              input
             (tensor)
                |
             reshape
                |
                +------------+
                |            |
               mean          |
                |            |
             reshape         |
                |            |
                + --->sub<---+
                       |
                       +-------+
                       |       |
                      pow      |
      input            |       |
      (eps)           mean     |
        |              |       |
        +----->add<----+       |
                |              |      input
              rsqrt            |     (weight)
                |              |        |         input
             reshape           |     reshape      (bias)
                |              |        |           |
                +----->mul<----+      expand     reshape
                        |               |           |
                        +----->mul<-----+         expand
                                |                   |
                                +------->add<-------+
                                          |
                                       reshape
                                          |
                                        output
    """

    def __init__(self):
        super().__init__()

    def _insert_norm(self, graph, tensor, eps, origin):
        """
        Insert (tensor - mean) / sqrt(var + eps)) into the graph
          and return the normalized tensor node.
        """
        mean = create_node(
            graph,
            torch.ops.aten.mean.dim,
            (tensor, [-1]),
            {"keepdim": True},
            origin=origin,
        )
        deviation = create_node(
            graph, torch.ops.aten.sub.Tensor, (tensor, mean), origin=origin
        )
        squared = create_node(
            graph, torch.ops.aten.pow.Tensor_Scalar, (deviation, 2), origin=origin
        )
        var = create_node(
            graph,
            torch.ops.aten.mean.dim,
            (squared, [-1]),
            {"keepdim": True},
            origin=origin,
        )
        inverse_std = create_node(
            graph,
            torch.ops.aten.rsqrt.default,
            (create_node(graph, torch.ops.aten.add.Tensor, (var, eps), origin=origin),),
            origin=origin,
        )
        return create_node(
            graph, torch.ops.aten.mul.Tensor, (deviation, inverse_std), origin=origin
        )

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        gm = exported_program.graph_module
        graph: torch.fx.Graph = gm.graph
        modified = False

        for node in graph.nodes:
            if not is_target_node(
                node,
                [
                    torch.ops.aten.native_layer_norm.default,
                    torch.ops.aten.native_group_norm.default,
                ],
            ):
                continue

            if node.target == torch.ops.aten.native_layer_norm.default:
                ln_args = NativeLayerNormArgs(*node.args, **node.kwargs)
                x = ln_args.input
                normalized_shape = ln_args.normalized_shape
                weight = ln_args.weight
                bias = ln_args.bias
                eps = ln_args.eps

                if weight:
                    weight_shape = extract_shape(weight)
                    assert list(weight_shape) == normalized_shape
                if bias:
                    bias_shape = extract_shape(bias)
                    assert list(bias_shape) == normalized_shape

                x_val = x.meta.get("val")
                assert isinstance(x_val, torch.Tensor)
                x_shape = list(x_val.size())
                x_dim = len(x_shape)
                normalized_dim = len(normalized_shape)
                assert x_dim >= normalized_dim
                idx_normalize_start = x_dim - normalized_dim

                norm_size = math.prod(normalized_shape)
                layer_size = math.prod(x_shape[:idx_normalize_start])
            elif node.target == torch.ops.aten.native_group_norm.default:
                gn_args = NativeGroupNormArgs(*node.args, **node.kwargs)
                x = gn_args.input
                weight = gn_args.weight
                bias = gn_args.bias
                N = gn_args.N
                C = gn_args.C
                HW = gn_args.HxW
                group = gn_args.group
                eps = gn_args.eps

                x_shape = list(extract_shape(x))
                assert len(x_shape) == 4 or len(x_shape) == 3
                assert x_shape[0] == N
                assert x_shape[1] == C

                assert C % group == 0
                norm_size = int((C / group) * HW)
                layer_size = N * group
            else:
                assert False, "Unreachable"

            pack_shape = [layer_size, norm_size]

            with gm.graph.inserting_before(node):
                # Branch only on whether a reshape is needed; the normalization is shared.
                if norm_size != x_shape[-1]:
                    # Pack groups so that the last dimension equals norm_size.
                    packed = create_node(
                        graph,
                        torch.ops.aten.reshape.default,
                        (x, pack_shape),
                        origin=node,
                    )
                    normed = self._insert_norm(graph, packed, eps, origin=node)
                    # Restore the original shape after normalization.
                    layer_norm = create_node(
                        graph,
                        torch.ops.aten.reshape.default,
                        (normed, x_shape),
                        origin=node,
                    )
                else:
                    # The input already has norm_size in the last dimension.
                    layer_norm = self._insert_norm(graph, x, eps, origin=node)

                # weight
                if weight:
                    if node.target == torch.ops.aten.native_group_norm.default:
                        weight_shape = extract_shape(weight)
                        assert weight_shape[0] == C
                        reshape_size = [1] * len(x_shape)
                        reshape_size[1] = C
                        weight = create_node(
                            graph,
                            torch.ops.aten.view.default,
                            (weight, reshape_size),
                            origin=node,
                        )
                    layer_norm = create_node(
                        graph,
                        torch.ops.aten.mul.Tensor,
                        (layer_norm, weight),
                        origin=node,
                    )

                # bias
                if bias:
                    if node.target == torch.ops.aten.native_group_norm.default:
                        bias_shape = extract_shape(bias)
                        assert bias_shape[0] == C
                        reshape_size = [1] * len(x_shape)
                        reshape_size[1] = C
                        bias = create_node(
                            graph,
                            torch.ops.aten.view.default,
                            (bias, reshape_size),
                            origin=node,
                        )
                    layer_norm = create_node(
                        graph,
                        torch.ops.aten.add.Tensor,
                        (layer_norm, bias),
                    )
                # Reset last node's meta for propagating replacing node's meta.
                layer_norm.meta = {}

                # NOTE Why select user `getitem` here?
                #   `native_layer_norm` and `native_group_norm` requires `getitem`
                #   to select the first output and discard the rest unused outputs.
                #   To replace those operators, it's necessary to replace the corresponding
                #   `getitem` node as well.
                get_item = next(iter(node.users))
                assert (
                    get_item.target == operator.getitem
                ), "First user of native_group/layer_norm should be getitem"

                get_item.replace_all_uses_with(layer_norm, propagate_meta=True)

            modified = True

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

        return PassResult(modified)
