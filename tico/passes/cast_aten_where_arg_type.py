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

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_torch_dtype
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import (
    trace_const_diff_on_pass,
    trace_graph_diff_on_pass,
)
from tico.utils.utils import is_target_node, set_new_meta_val
from tico.utils.validate_args_kwargs import WhereSelfArgs


dtype_ranking = {
    torch.int32: 0,
    torch.int64: 1,
    torch.float32: 2,
}


def sort_by_dtype(
    result_true: torch.fx.Node, result_false: torch.fx.Node
) -> Tuple[torch.fx.Node, torch.fx.Node]:
    true_dtype = extract_torch_dtype(result_true)
    false_dtype = extract_torch_dtype(result_false)
    if dtype_ranking[true_dtype] > dtype_ranking[false_dtype]:
        return result_true, result_false
    if dtype_ranking[true_dtype] < dtype_ranking[false_dtype]:
        return result_false, result_true
    assert False, "There is no case that the dtype_ranking of the nodes are the same"


def check_if_covered_by_float(tensor: torch.Tensor) -> bool:
    # About the min/max range, please refer to https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Precision_limitations_on_integer_values
    if tensor.min() < -(2**24) or tensor.max() > 2**24:
        return False
    return True


@trace_graph_diff_on_pass
@trace_const_diff_on_pass
class CastATenWhereArgType(PassBase):
    """
    This pass casts the data type of `aten.where.self` operation's argument.

    This pass is applied when the data type of `aten.where.self` operation's argument is different.
    If the data type of arguments, which are denoted `result_true` and `result_false` in below graph are identical, this pass is not applied.

    In addition, this pass casts the data type as the direction that avoids data loss.
    For example, if the data type of `result_true` is `float32` and the data type of `result_false` is `int32`,
    then the data type of `result_false` will be casted to `float32`.
    Moreover, in this case, it should be checked whether the contents of `result_false` are within the range of `float32`.
    If so, the data type of `result_true` will be casted to `float32`.
    If not, RuntimeError will be raised.

    After this pass, the arguments of `aten.where.self` should have same data type.

    The graph before this pass and the graph after this pass are shown below.
    NOTE Below example denotes the case when the `result_false` was casted.

    (before)

    [condition]   [result_true]   [result_false]
        |               |                |
        |               |                |
        +---------------+----------------+
                        |
                        |
                     [where]
                        |
                        |
                    [output]

    (after)

                                  [result_false]
    [condition]   [result_true]          |
        |               |             [cast]
        |               |                |
        +---------------+----------------+
                        |
                        |
                     [where]
                        |
                        |
                    [output]
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if not is_target_node(node, torch.ops.aten.where.self):
                continue

            where_args = WhereSelfArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            result_true, result_false = where_args.input, where_args.other
            if not isinstance(result_true, torch.fx.Node) or not isinstance(
                result_false, torch.fx.Node
            ):
                continue

            ep = exported_program
            assert isinstance(result_true, torch.fx.Node)
            assert isinstance(result_false, torch.fx.Node)
            if not (
                result_true.name in ep.graph_signature.inputs_to_buffers
                and result_false.name in ep.graph_signature.inputs_to_buffers
            ):
                continue

            # Check if they have different data types
            true_dtype = extract_torch_dtype(result_true)
            false_dtype = extract_torch_dtype(result_false)
            if true_dtype == false_dtype:
                continue

            node_to_dtype = {result_true: true_dtype, result_false: false_dtype}

            not_to_cast, to_cast = sort_by_dtype(result_true, result_false)

            buf_name_to_data = {name: buf for name, buf in ep.named_buffers()}
            buf_name = ep.graph_signature.inputs_to_buffers[to_cast.name]
            buf_data = buf_name_to_data[buf_name]

            assert isinstance(buf_data, torch.Tensor)

            dtype_to_cast = node_to_dtype[not_to_cast]

            if dtype_to_cast == torch.float32:
                if not check_if_covered_by_float(buf_data):
                    raise RuntimeError(
                        f"{to_cast.name}({buf_data.dtype}) data range is out of {dtype_to_cast} range"
                    )
            with graph_module.graph.inserting_after(to_cast):
                cast = create_node(
                    graph,
                    torch.ops.aten._to_copy.default,
                    args=(to_cast,),
                    kwargs={"dtype": dtype_to_cast},
                    origin=to_cast,
                )
            # set new meta["val"] in advance because we will use it below for checking if type promotion is valid.
            set_new_meta_val(cast)
            node.update_arg(node.args.index(to_cast), cast)

            # check if type promotion is valid.
            node_dtype_ori = extract_torch_dtype(node)
            set_new_meta_val(node)
            node_dtype = extract_torch_dtype(node)
            assert (
                node_dtype == node_dtype_ori
            ), f"Type casting doesn't change node's dtype."

            logger.debug(
                f"{to_cast.name}'s dtype was casted from {buf_data.dtype} to {dtype_to_cast}"
            )

            modified = True

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
