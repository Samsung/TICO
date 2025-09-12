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

# https://github.com/pytorch/executorch/blob/61ddee5/exir/passes/constant_prop_pass.py

from collections import OrderedDict
from typing import List, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
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
from torch.export.exported_program import InputKind, InputSpec
from torch.utils import _pytree as pytree

from tico.serialize.circle_graph import _PRIMITIVE_TYPES
from tico.utils import logging
from tico.utils.graph import create_input_spec, generate_fqn, get_first_user_input
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import (
    trace_const_diff_on_pass,
    trace_graph_diff_on_pass,
)
from tico.utils.utils import get_fake_mode


def get_constant_placeholder_to_tensor_dict(
    exported_program: ExportedProgram,
    graph_module,
) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    """
    Returns a dictionary of constant placeholder node to constant tensor.
    """
    const_node_to_tensor: OrderedDict[torch.fx.Node, torch.Tensor] = OrderedDict()
    graph: torch.fx.Graph = graph_module.graph
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        tensor: Optional[torch.Tensor] = None
        if is_param(exported_program, node):
            tensor = get_param(exported_program, node)
        elif is_buffer(exported_program, node):
            tensor = get_buffer(exported_program, node)
        elif is_lifted_tensor_constant(exported_program, node):
            tensor = get_lifted_tensor_constant(exported_program, node)

        if tensor is not None:
            assert node not in const_node_to_tensor
            const_node_to_tensor[node] = tensor

    return const_node_to_tensor


def has_constant_data(arg, const_node_to_tensor=None) -> bool:
    """
    Check if `arg` has constant data.

    Assume that `const_node_to_tensor` is retrived from exported program.
    When a node is a placeholder, only method to check if it is constant is to check the exported program.
    """
    if isinstance(arg, (tuple, list)):
        return all(has_constant_data(a, const_node_to_tensor) for a in arg)
    elif isinstance(arg, dict):
        return all(has_constant_data(a, const_node_to_tensor) for a in arg.values())
    elif isinstance(
        arg,
        _PRIMITIVE_TYPES,
    ):
        return True
    elif not isinstance(arg, torch.fx.Node):
        return False
    elif const_node_to_tensor is not None and arg in const_node_to_tensor:
        return True

    return False


def get_data(
    arg,
    exported_program: ExportedProgram,
    const_node_to_tensor: Mapping[torch.fx.Node, torch.Tensor],
):
    if isinstance(arg, (tuple, list)):
        return (get_data(x, exported_program, const_node_to_tensor) for x in arg)
    elif isinstance(arg, _PRIMITIVE_TYPES):
        return arg
    elif arg in const_node_to_tensor:
        return const_node_to_tensor[arg]
    return None


def propagate_constants(
    exported_program: ExportedProgram,
    graph_module
) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    """
    Propagates constants and returns a dictionary of node to constant tensors of the graph.
    """
    const_node_to_tensor = get_constant_placeholder_to_tensor_dict(exported_program, graph_module)

    graph: torch.fx.Graph = graph_module.graph
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]:
            continue
        if not has_constant_data(
            [node.args, node.kwargs],
            const_node_to_tensor,
        ):
            continue

        args_data, kwargs_data = pytree.tree_map(
            lambda x: get_data(x, exported_program, const_node_to_tensor),
            (node.args, node.kwargs),
        )

        # propagate constant because all of its args are constant tensors.
        with torch.no_grad():
            prop_constant_tensor = node.target(*args_data, **kwargs_data)
        const_node_to_tensor[node] = prop_constant_tensor

    return const_node_to_tensor


def erase_constant_node(
    exported_program: ExportedProgram,
    node: torch.fx.Node,
) -> None:
    """
    Remove corresponding tensor from param/constants dict.

    Q) Isn't it necessary to remove a node from `inputs_to_parameters`, `inputs_to_lifted_tensor_constants`
      and `inputs_to_buffers` as well? Why do they just call `get`?
    A) They internally uses `exported_program.graph_signature.input_specs` and the `input_specs` are updated
      at the end of the const_prop_pass.
    """
    signature = exported_program.graph_signature
    if name := signature.inputs_to_parameters.get(node.name, None):
        exported_program.state_dict.pop(name, None)
    elif name := signature.inputs_to_lifted_tensor_constants.get(node.name, None):
        exported_program.constants.pop(name, None)
    elif name := signature.inputs_to_buffers.get(node.name, None):
        exported_program.constants.pop(name, None)
        exported_program.state_dict.pop(name, None)

    # Remove from graph.
    exported_program.graph.erase_node(node)


def create_constant_placeholder(
    const_node_to_tensor: Mapping[torch.fx.Node, torch.Tensor],
    exported_program: ExportedProgram,
    graph_module
) -> List[torch.fx.Node]:
    """
    This function creates constant placeholder nodes according to the given constant nodes (`const_node_to_tensor`) and replace it with the original node.
    """
    placeholders = []

    fake_mode = get_fake_mode(exported_program)
    first_user_input = get_first_user_input(exported_program)
    if not first_user_input:
        # Placeholder nodes must be the first N nodes in the nodes list of a graph.
        # Therefore, insert the newly created placeholders at the start of the node list.
        assert exported_program.graph.nodes
        first_node = list(exported_program.graph.nodes)[0]
        first_user_input = first_node

    # Iterate over nodes in reverse order to insert created placeholder before the `first_user_input`.
    for node, prop_constant_tensor in reversed(const_node_to_tensor.items()):
        if all(x in const_node_to_tensor for x in node.users):
            # All users of this constant node are also constant, so we don't need to create a new constant node.
            erase_constant_node(exported_program, node)
            continue

        if node.op == "placeholder":
            continue

        # Add `prop_constant_tensor` to program.state_dict.
        prop_constant_tensor_fqn = generate_fqn(
            "_prop_tensor_constant", exported_program
        )

        # Insert a new placeholder node for the propagated constant tensor.
        with exported_program.graph.inserting_before(first_user_input):
            const_placeholder_node = exported_program.graph.placeholder(
                prop_constant_tensor_fqn
            )

        # The key here should be same with "target" arg of InputSpec when creating input specs.
        exported_program.constants[prop_constant_tensor_fqn] = prop_constant_tensor

        # Replace the original node with the new constant node.
        node.replace_all_uses_with(const_placeholder_node, propagate_meta=True)
        exported_program.graph.erase_node(node)

        # Update the meta data of the new placeholder node.
        const_placeholder_node.meta["val"] = fake_mode.from_tensor(
            prop_constant_tensor, static_shapes=True
        )
        const_placeholder_node.meta["val"].constant = prop_constant_tensor

        placeholders.append(const_placeholder_node)

    return placeholders


def create_input_specs(
    placeholders: List[torch.fx.Node],
) -> dict[str, InputSpec]:
    name_to_spec: dict[str, InputSpec] = {}

    # https://pytorch.org/docs/stable/export.ir_spec.html#placeholder
    # %name = placeholder[target = name](args = ())
    for node in placeholders:
        name_to_spec[node.name] = create_input_spec(node, InputKind.CONSTANT_TENSOR)

    return name_to_spec


@trace_graph_diff_on_pass
@trace_const_diff_on_pass
class ConstPropPass(PassBase):
    """
    Performs constant folding and constant propagation.

    NOTE The exported program gurantees that parameters, buffers, and constant tensors are lifted out of the graph as inputs.
    It means that the pass need to update input specs after folding the constant nodes.
    # ref: https://pytorch.org/docs/stable/export.html#torch.export.ExportGraphSignature

    [WHAT IT DOES]
    [1] Propagate the constants.
    [2] Get propagated data from constant nodes.
    [3] Create the constant placeholder nodes according to the propagated data.
    [4] Create input specs according to the created placeholders.
    [5] Update the input specs.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, exported_program: ExportedProgram, graph_module) -> PassResult:
        logger = logging.getLogger(__name__)
        graph: torch.fx.Graph = graph_module.graph

        # [1], [2]
        const_node_to_tensor: OrderedDict[
            torch.fx.Node, torch.Tensor
        ] = propagate_constants(exported_program, graph_module)
        # [3]
        placeholders = create_constant_placeholder(
            const_node_to_tensor, exported_program, graph_module
        )
        # [4]
        new_name_to_spec = create_input_specs(placeholders)

        # [5]
        # Get existing input specs.
        existing_name_to_spec = {
            s.arg.name: s for s in exported_program.graph_signature.input_specs
        }
        # Add the new constants to existing input specs dict.
        existing_name_to_spec.update(new_name_to_spec)
        # Generate new input spec.
        new_input_specs = []
        for node in exported_program.graph.nodes:
            if node.op != "placeholder":
                continue
            assert node.name in existing_name_to_spec, node.name
            new_input_specs.append(existing_name_to_spec[node.name])
        exported_program.graph_signature.input_specs = new_input_specs

        graph.eliminate_dead_code()
        graph_module.recompile()

        logger.debug("Constant nodes are propagated")
        # Constant folding can be done with only one time run. Let's set `modified` to False.
        modified = False
        return PassResult(modified)
