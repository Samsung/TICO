# Portions of this file are adapted from code originally authored by
# Meta Platforms, Inc. and affiliates, licensed under the BSD-style
# license found in the LICENSE file in the root directory of their source tree.

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
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_torch_dtype
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node, set_new_meta_val


ops_to_promote = {
    torch.ops.aten.add.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.div.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.eq.Scalar: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.eq.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.ge.Scalar: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.ge.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.gt.Scalar: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.gt.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.mul.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.minimum.default: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.ne.Scalar: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.ne.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.pow.Tensor_Scalar: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    torch.ops.aten.sub.Tensor: ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
}


def has_same_dtype(lhs, rhs):
    if isinstance(lhs, torch.fx.Node):
        lhs_dtype = lhs.meta["val"].dtype
    elif isinstance(lhs, torch.Tensor):
        lhs_dtype = lhs.dtype
    else:
        lhs_dtype = torch.tensor(lhs).dtype
    if isinstance(rhs, torch.fx.Node):
        rhs_dtype = rhs.meta["val"].dtype
    elif isinstance(rhs, torch.Tensor):
        rhs_dtype = rhs.dtype
    else:
        rhs_dtype = torch.tensor(rhs).dtype

    if lhs_dtype == rhs_dtype:
        return True
    return False


def to_numeric_type(torch_dtype: torch.dtype):
    dmap = {
        torch.float32: float,
        torch.float: float,
        torch.int64: int,
        torch.bool: bool,
    }

    if torch_dtype not in dmap:
        return None

    return dmap[torch_dtype]


@trace_graph_diff_on_pass
class CastMixedTypeArgs(PassBase):
    def __init__(self, preserve_ep_invariant=True):
        super().__init__()
        self.preserve_ep_invariant = preserve_ep_invariant

    # TODO Folding float and int values before this pass
    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, list(ops_to_promote.keys())):
                continue

            assert len(node.args) == 2
            lhs, rhs = node.args
            assert isinstance(lhs, (torch.fx.Node, torch.Tensor, float, int)), type(lhs)
            assert isinstance(rhs, (torch.fx.Node, torch.Tensor, float, int)), type(rhs)
            if has_same_dtype(lhs, rhs):
                continue

            lhs_val = (
                lhs.meta["val"] if isinstance(lhs, torch.fx.Node) else torch.tensor(lhs)
            )
            rhs_val = (
                rhs.meta["val"] if isinstance(rhs, torch.fx.Node) else torch.tensor(rhs)
            )
            type_to_promote: torch.dtype = elementwise_dtypes(
                lhs_val, rhs_val, type_promotion_kind=ops_to_promote[node.target]
            )[1]
            arg_to_promote = None
            if lhs_val.dtype == type_to_promote:
                ori_type = rhs_val.dtype
                arg_to_promote = rhs
            if rhs_val.dtype == type_to_promote:
                ori_type = lhs_val.dtype
                arg_to_promote = lhs
            assert arg_to_promote != None

            if isinstance(arg_to_promote, torch.fx.Node):
                with graph.inserting_after(arg_to_promote):
                    to_copy = create_node(
                        graph,
                        torch.ops.aten._to_copy.default,
                        (arg_to_promote,),
                        {"dtype": type_to_promote},
                        origin=arg_to_promote,
                    )
                    # set new meta["val"] in advance because we will use it below for checking if type promotion is valid.
                    set_new_meta_val(to_copy)
                    node.update_arg(node.args.index(arg_to_promote), to_copy)

                modified = True
                logger.debug(
                    f"{arg_to_promote.name}'s dtype was casted from {ori_type} to {type_to_promote}"
                )
            else:
                index_to_promote = node.args.index(arg_to_promote)
                if isinstance(arg_to_promote, torch.Tensor):
                    arg_to_promote = arg_to_promote.to(type_to_promote)
                else:
                    # numerical types
                    numeric_type = to_numeric_type(type_to_promote)
                    if numeric_type is not None:
                        arg_to_promote = numeric_type(arg_to_promote)
                    else:
                        if self.preserve_ep_invariant:
                            # ExportedProgram (EP) requires to add a placeholder when
                            # a tensor is created, which complicates EP structure but
                            # not necessary for circle serialization. We skip this case if
                            # preserve_ep_invariant = True.
                            continue
                        else:
                            # Create tensor without placeholder
                            # NOTE This breaks EP invariant
                            arg_to_promote = torch.tensor(arg_to_promote).to(
                                type_to_promote
                            )
                node.update_arg(index_to_promote, arg_to_promote)

                modified = True
                logger.debug(
                    f"{arg_to_promote}'s dtype was casted from {ori_type} to {type_to_promote}"
                )

            # check if type promotion is valid.
            node_dtype_ori = extract_torch_dtype(node)
            set_new_meta_val(node)
            node_dtype = extract_torch_dtype(node)
            assert (
                node_dtype == node_dtype_ori
            ), f"Type casting doesn't change node's dtype."

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
