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
from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import PermuteArgs


def _compose_permutation(dims1: list[int], dims2: list[int]):
    """
    Compose two permutation vectors.

    Given y = x.permute(dims1) and z = y.permute(dims2),
    the overall permutation p = dims2 ∘ dims1 is

        p[i] = dims1[dims2[i]]
    """
    assert len(dims1) == len(
        dims2
    ), f"len(dims1): {len(dims1)}, len(dims2): {len(dims2)}"
    return [dims1[i] for i in dims2]


def passes():
    """
    Return a list of passes that remove redundant `aten.permute` operators.

    NOTE Both shape and stride of input/output should be same.
    """
    return [
        RemoveRedundantPermutePattern1(),
    ]


@trace_graph_diff_on_pass
class RemoveRedundantPermutePattern1(PassBase):
    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        """
        [BEFORE]
            (AxBxC) - aten.permute_1 - aten.permute_2 - (OUT_SHAPE)
        [AFTER]
            if OUT_SHAPE == (AxBxC):
                (AxBxC)
            else:
                (AxBxC) - aten.permute (fused dims) - (OUT_SHAPE)

        """
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for permute2 in graph.nodes:
            if not is_target_node(permute2, ops.aten.permute):
                continue

            if len(permute2.users) != 1:
                continue
            permute2_args = PermuteArgs(*permute2.args, **permute2.kwargs)  # type: ignore[arg-type]
            permute1, permute2_dims = permute2_args.input, permute2_args.dims

            if not is_target_node(permute1, ops.aten.permute):
                continue
            if len(permute1.users) != 1:
                continue
            permute1_args = PermuteArgs(*permute1.args, **permute1.kwargs)  # type: ignore[arg-type]
            permute1_input, permute1_dims = permute1_args.input, permute1_args.dims

            fused_dims = _compose_permutation(permute1_dims, permute2_dims)
            identity = list(range(len(fused_dims)))

            if fused_dims == identity:
                # shape
                permute1_input_shape = extract_shape(permute1_input)
                permute2_shape = extract_shape(permute2)
                assert permute1_input_shape == permute2_shape

                permute2.replace_all_uses_with(permute1_input, propagate_meta=False)
                logger.debug(f"{permute1.name} and {permute2.name} are removed.")
            else:
                with graph.inserting_after(permute2):
                    new_args = (permute1_input, fused_dims)
                    fused_permute = create_node(
                        graph,
                        torch.ops.aten.permute.default,
                        args=new_args,
                    )
                    permute2.replace_all_uses_with(fused_permute, propagate_meta=True)
                    logger.debug(f"{permute1.name} and {permute2.name} are fused.")
            modified = True

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
