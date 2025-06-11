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
from torch.utils import _pytree as pytree

from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node


def _extract_to_output(node: torch.fx.Node, graph: torch.fx.Graph) -> bool:
    """
    This extracts dtype kwargs to node's output direction

    So, op(..., dtype = X) is converted to op(...).to(X)

    Return true if modified

    NOTE

    [1] This function always returns true. Return value is introduced for extension
    [2] This conversion is not safe for some Ops whose inputs should also be casted to X (ex: Mean).

    """
    logger = logging.getLogger(__name__)

    node_kwargs = node.kwargs
    # Remove "dtype" from node's kwargs
    new_kwargs = {}
    for k, v in node_kwargs.items():
        if k == "dtype":
            continue
        new_kwargs[k] = v
    node.kwargs = new_kwargs
    # Create new val for node
    # `node.target()` needs only `Tensor` for its arguments. Therefore, let's retrieve `FakeTensor` if it is `torch.fx.Node`.
    args, kwargs = pytree.tree_map_only(
        torch.fx.Node, lambda x: x.meta["val"], (node.args, node.kwargs)
    )
    new_val = node.target(*args, **kwargs)  # type: ignore[operator]
    # Set args, kwargs of `to_copy`
    to_args = (node,)
    to_kwargs = {"dtype": node_kwargs["dtype"]}
    with graph.inserting_after(node):
        to_copy = graph.call_function(torch.ops.aten._to_copy.default, (), {})
        node.replace_all_uses_with(to_copy, propagate_meta=True)
        # Q) Why lazy-update args, kwargs of the `to_copy`?
        # A) `replace_all_uses_with` replace all the uses of `node`. If `to_copy` args is set to
        #   (node, ) before `replace_all_uses_with`, the function would even replace the args of
        #   `to_copy` with `to_copy`.
        to_copy.args = to_args
        to_copy.kwargs = to_kwargs
        # Update meta["val"] to change dtype
        node.meta["val"] = new_val

    logger.debug(f"{node.name}'s dtype kwargs is extracted into {to_copy.name}")

    return True


@trace_graph_diff_on_pass
class ExtractDtypeKwargsPass(PassBase):
    """
    This pass extracts "dtype" keyword argument from nodes.

    Sometimes, torch api receives "dtype" keyword argument.

    E.g. x_bool = torch.full_like(x, 0, dtype=torch.bool)

    But, this argument makes circle build logic complicated because many operators has
      same type with their inputs'.

    So, this pass changes `op(dtype)` to `op + to(dtype)`.

    NOTE

    [1] There are some ops that are natural to have "dtype" kwargs. The pass is not applied to those ops.
    [2] If node.kwargs["dtype"] is redundant `op(dtype).dtype == op().dtype`, the pass is not applied.

    """

    def __init__(self):
        super().__init__()
        # List of Ops whose "dtype" kwargs is extracted
        self.target_ops = dict()
        self.target_ops[torch.ops.aten.full_like.default] = _extract_to_output

    def call(self, exported_program: ExportedProgram) -> PassResult:
        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, list(self.target_ops.keys())):
                continue
            if "dtype" not in node.kwargs:
                continue

            modified |= self.target_ops[node.target](node, graph)

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
