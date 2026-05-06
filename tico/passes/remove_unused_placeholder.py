# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import (
    trace_const_diff_on_pass,
    trace_graph_diff_on_pass,
)


def _is_constant_placeholder(
    exported_program: ExportedProgram,
    node: "torch.fx.Node",
) -> bool:
    """
    Return whether the given placeholder represents a lifted constant.

    Parameters, buffers, and lifted tensor constants are treated as constant
    placeholders because they are backed by ExportedProgram state instead of
    runtime user inputs.
    """

    if node.op != "placeholder":
        return False

    return (
        is_param(exported_program, node)
        or is_buffer(exported_program, node)
        or is_lifted_tensor_constant(exported_program, node)
    )


def _remove_constant_placeholder(
    exported_program: ExportedProgram,
    node: "torch.fx.Node",
) -> None:
    """
    Remove an unused constant placeholder from the graph and ExportedProgram state.

    The graph signature is updated by the caller after all unused placeholders are
    removed.
    """

    signature = exported_program.graph_signature

    if name := signature.inputs_to_parameters.get(node.name, None):
        exported_program.state_dict.pop(name, None)
    elif name := signature.inputs_to_lifted_tensor_constants.get(node.name, None):
        exported_program.constants.pop(name, None)
    elif name := signature.inputs_to_buffers.get(node.name, None):
        exported_program.constants.pop(name, None)
        exported_program.state_dict.pop(name, None)

    exported_program.graph.erase_node(node)


@trace_graph_diff_on_pass
@trace_const_diff_on_pass
class RemoveUnusedPlaceholder(PassBase):
    """
    Remove unused constant placeholders from an exported graph.

    FX dead-code elimination does not remove placeholder nodes even when they have
    no users. This pass removes unused placeholders that correspond to parameters,
    buffers, or lifted tensor constants, and then updates the ExportedProgram graph
    signature accordingly.

    Runtime user input placeholders are never removed by this pass.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph

        unused_placeholders = [
            node
            for node in graph.nodes
            if _is_constant_placeholder(exported_program, node) and len(node.users) == 0
        ]

        if not unused_placeholders:
            return PassResult(False)

        removed_names = [node.name for node in unused_placeholders]

        for node in unused_placeholders:
            _remove_constant_placeholder(exported_program, node)

        existing_name_to_spec = {
            spec.arg.name: spec for spec in exported_program.graph_signature.input_specs
        }
        exported_program.graph_signature.input_specs = [
            existing_name_to_spec[node.name]
            for node in graph.nodes
            if node.op == "placeholder"
        ]

        graph.lint()
        graph_module.recompile()

        logger.debug(f"Unused constant placeholders are removed: {removed_names}")

        # Run only once.
        return PassResult(False)
