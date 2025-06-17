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

from typing import Sequence

import torch
from torch.export import ExportedProgram

from tico.passes import ops
from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import PermuteArgs, ReshapeArgs


def _is_leading_unsqueeze(target: Sequence[int], permuted: Sequence[int]) -> bool:
    """
    True if `target` == [1]*k + permuted,  k>=1.
    """
    k = len(target) - len(permuted)
    return (
        k > 0 and all(d == 1 for d in target[:k]) and list(target[k:]) == list(permuted)
    )


@trace_graph_diff_on_pass
class FuseLeadingUnsqueezeReshape(PassBase):
    """
    Fuse reshape → permute → reshape where the second reshape only
    prepends one-sized dims (unsqueeze) to the permuted tensor.

    [BEFORE]
        x - aten.reshape(s1) - aten.permute(p) - aten.reshape([1]*k + p(s1))
    [AFTER]
        x - aten.reshape([1]*k + s1) - aten.permute(list(range(k)) + [d+k for d in p])
    """

    def call(self, ep: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        gm = ep.graph_module
        graph = gm.graph
        modified = False
        for reshape_back in graph.nodes:
            if not is_target_node(reshape_back, ops.aten.reshape):
                continue
            reshape_back_args = ReshapeArgs(*reshape_back.args, **reshape_back.kwargs)  # type: ignore[arg-type]
            permute = reshape_back_args.input

            if not is_target_node(permute, ops.aten.permute):
                continue
            permute_args = PermuteArgs(*permute.args, **permute.kwargs)  # type: ignore[arg-type]
            reshape_front, permute_dims = permute_args.input, permute_args.dims

            if not is_target_node(reshape_front, ops.aten.reshape):
                continue
            reshape_front_args = ReshapeArgs(*reshape_front.args, **reshape_front.kwargs)  # type: ignore[arg-type]
            reshape_front_input, reshape_front_size = (
                reshape_front_args.input,
                reshape_front_args.shape,
            )

            # ---- condition: only leading unsqueeze ------------------------
            back_shape = extract_shape(reshape_back)
            permute_shape = extract_shape(permute)

            if not _is_leading_unsqueeze(back_shape, permute_shape):
                continue

            # ---- create new reshape & new permute -------------------------
            k = len(back_shape) - len(permute_shape)
            with graph.inserting_before(permute):
                new_shape = [1] * k + list(reshape_front_size)
                r_new = create_node(
                    graph,
                    torch.ops.aten.reshape.default,
                    args=(reshape_front_input, new_shape),
                    origin=reshape_back,
                )
                new_p_dims = list(range(k)) + [
                    d + k for d in permute_dims
                ]  # shift by k
                p_new = create_node(
                    graph,
                    torch.ops.aten.permute.default,
                    args=(r_new, new_p_dims),
                )

            reshape_back.replace_all_uses_with(p_new, propagate_meta=True)
            modified = True
            logger.debug(f"{reshape_back.name} is fused to {r_new.name}")

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            gm.recompile()

        return PassResult(modified)
