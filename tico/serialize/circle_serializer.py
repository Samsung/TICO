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

import operator
from typing import Dict

import flatbuffers
import torch
from circle_schema import circle
from torch.export.exported_program import ConstantArgument, ExportedProgram, InputKind

from tico.config import CompileConfigBase, get_default_config
from tico.serialize.circle_mapping import to_circle_dtype, to_circle_shape
from tico.serialize.operators import *
from tico.serialize.circle_graph import CircleModel, CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import get_node_visitors
from tico.utils import logging
from tico.utils.serialize import finalise_tensor_names, validate_tensor_shapes


multiple_output_ops = [
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.max.dim,
]


def _initialize_model() -> tuple[CircleModel, CircleSubgraph]:
    """Initialize a new Circle model and subgraph.

    Returns:
        Tuple containing the model and subgraph
    """
    model = CircleModel()
    model.add_buffer(circle.Buffer.BufferT())  # Add empty buffer at the front
    graph = CircleSubgraph(model)
    return model, graph


def build_circle(
    ep: ExportedProgram, config: CompileConfigBase = get_default_config()
) -> bytes:
    """Convert ExportedProgram to Circle format.

    Args:
        ep: The exported PyTorch program to convert

    Returns:
        bytes: Raw bytes of the Circle model
    """
    logger = logging.getLogger(__name__)
    builder = flatbuffers.Builder()
    model, graph = _initialize_model()

    # Export tensors
    _export_tensors(graph, ep)

    # Register inputs
    logger.debug("---------------Register inputs--------------")
    for in_spec in ep.graph_signature.input_specs:
        if in_spec.kind != InputKind.USER_INPUT:
            continue
        if isinstance(in_spec.arg, ConstantArgument):
            # ConstantArgument is ignored when option is given
            if config.get("remove_constant_input"):
                continue
            # NoneType ConstantArgument is ignored.
            if in_spec.arg.value == None:
                continue
        arg_name = in_spec.arg.name
        graph.add_input(arg_name)
        logger.debug(f"Registered input: {arg_name}")

    # Register outputs
    logger.debug("---------------Register outputs--------------")
    for user_output in ep.graph_signature.user_outputs:
        if user_output == None:
            logger.debug("Ignore 'None' output")
            continue

        graph.add_output(user_output)
        logger.debug(f"Registered output: {user_output}")

    # Export operators
    logger.debug("---------------Export operators--------------")
    op_codes: Dict[OpCode, int] = {}
    visitors = get_node_visitors(op_codes, graph)
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue

        opcode = node.target
        if opcode == operator.getitem:
            continue
        if opcode not in visitors:
            raise RuntimeError(f"{opcode} is not yet supported")
        circle_op = visitors[opcode].define_node(node)

        if circle_op:
            graph.add_operator(circle_op)
            logger.debug(f"call_function: {node.name} ({opcode}) Op exported.")

    finalise_tensor_names(graph)
    validate_tensor_shapes(graph)

    # Register subgraph
    model.subgraphs.append(graph)

    # Encode operator codes
    model.operatorCodes = [
        code for code, _ in sorted(op_codes.items(), key=lambda x: x[1])
    ]

    # Final model settings
    model.description = "circle"
    model.version = 0

    # Finish model
    builder.Finish(model.Pack(builder), "CIR0".encode("utf8"))
    buf = builder.Output()

    return bytes(buf)


def _export_tensors(graph: CircleSubgraph, ep: ExportedProgram) -> None:
    """Export all tensors from the exported program to the circle graph.

    Args:
        graph: The CircleSubgraph to add tensors to
        ep: The exported PyTorch program
    """
    logger = logging.getLogger(__name__)
    logger.debug("---------------Export tensors--------------")
    buf_name_to_data = {name: buf for name, buf in ep.named_buffers()}

    for node in ep.graph.nodes:
        if node.op == "call_function":
            if node.target in multiple_output_ops:
                continue
            node_val = node.meta["val"]
            if node_val.layout != torch.strided:
                raise RuntimeError(
                    f"Only support dense tensors (node layout: {node_val.layout})"
                )
            graph.add_tensor_from_node(node)
            logger.debug(f"call_function: {node.name} tensor exported.")

        elif node.op == "placeholder":
            _handle_placeholder_node(graph, node, ep, buf_name_to_data)

        elif node.op == "get_attr":
            _handle_get_attr_node(graph, node)

        elif node.op == "output":
            for output in node.args[0]:
                if isinstance(output, torch.fx.Node):
                    assert graph.has_tensor(output.name)
            continue

        elif node.op == "call_method":
            raise AssertionError("Not yet implemented")

        elif node.op == "call_module":
            raise AssertionError("Not yet implemented")

        else:
            raise AssertionError(f"Unknown fx.Node op {node.op}")


def _handle_placeholder_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    ep: ExportedProgram,
    buf_name_to_data: dict,
) -> None:
    """Handle a placeholder node during tensor export."""
    # placeholder invariants
    assert node.args is None or len(node.args) == 0  # Not support default param

    if node.name in ep.graph_signature.inputs_to_parameters:
        _handle_parameter_node(graph, node, ep)
    elif node.name in ep.graph_signature.inputs_to_buffers:
        _handle_buffer_node(graph, node, ep, buf_name_to_data)
    elif node.name in ep.graph_signature.inputs_to_lifted_tensor_constants:
        _handle_constant_tensor_node(graph, node, ep)
    else:
        _handle_user_input_node(graph, node, ep)


def _handle_parameter_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    ep: ExportedProgram,
) -> None:
    """Handle a parameter placeholder node by exporting its tensor data.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The parameter node to process
        ep: ExportedProgram containing parameter data
    """
    param_name = ep.graph_signature.inputs_to_parameters[node.name]
    param_data = ep.state_dict[param_name]

    if not isinstance(param_data, torch.Tensor):
        raise ValueError(f"Parameter {param_name} is not a tensor")

    tensor_value = param_data.cpu().detach().numpy()
    graph.add_tensor_from_node(node, tensor_value)

    logger = logging.getLogger(__name__)
    logger.debug(f"Exported parameter tensor: {node.name}")


def _handle_buffer_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    ep: ExportedProgram,
    buf_name_to_data: dict,
) -> None:
    """Handle a buffer placeholder node by exporting its tensor data.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The buffer node to process
        ep: ExportedProgram containing buffer info
        buf_name_to_data: Mapping of buffer names to data
    """
    buffer_name = ep.graph_signature.inputs_to_buffers[node.name]

    if buffer_name not in buf_name_to_data:
        raise ValueError(f"Buffer {buffer_name} not found in buffer data")

    buffer_data = buf_name_to_data[buffer_name]

    if not isinstance(buffer_data, torch.Tensor):
        raise ValueError(f"Buffer {buffer_name} is not a tensor")

    tensor_value = buffer_data.cpu().detach().numpy()
    graph.add_tensor_from_node(node, tensor_value)

    logger = logging.getLogger(__name__)
    logger.debug(f"Exported buffer tensor: {node.name}")


def _handle_constant_tensor_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    ep: ExportedProgram,
) -> None:
    """Handle a constant tensor placeholder node by exporting its tensor data.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The constant tensor node to process
        ep: ExportedProgram containing constant data
    """
    ctensor_name = ep.graph_signature.inputs_to_lifted_tensor_constants[node.name]

    if ctensor_name not in ep.constants:
        raise ValueError(f"Constant tensor {ctensor_name} not found")

    ctensor_data = ep.constants[ctensor_name]

    if not isinstance(ctensor_data, torch.Tensor):
        raise ValueError(f"Constant tensor {ctensor_name} is not a tensor")

    tensor_value = ctensor_data.cpu().detach().numpy()
    graph.add_tensor_from_node(node, tensor_value)

    logger = logging.getLogger(__name__)
    logger.debug(f"Exported constant tensor: {node.name}")


def _handle_user_input_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    ep: ExportedProgram,
) -> None:
    """Handle a user input placeholder node by exporting its tensor data.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The user input node to process
        ep: ExportedProgram containing input specs
    """
    user_inputs = [
        specs
        for specs in ep.graph_signature.input_specs
        if specs.kind == InputKind.USER_INPUT
    ]
    constant_inputs = [
        specs for specs in user_inputs if isinstance(specs.arg, ConstantArgument)
    ]
    name_to_value = {specs.arg.name: specs.arg.value for specs in constant_inputs}

    # Skip NoneType ConstantArgument
    if node.name in name_to_value and name_to_value[node.name] is None:
        return

    graph.add_tensor_from_node(node)

    logger = logging.getLogger(__name__)
    logger.debug(f"Exported user input tensor: {node.name}")


def _handle_get_attr_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
) -> None:
    """Handle a get_attr node by exporting its tensor data.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The get_attr node to process
    """
    assert isinstance(node.target, str)
    attr_tensor = getattr(node.graph.owning_module, node.target)

    if not isinstance(attr_tensor, torch.Tensor):
        raise ValueError(f"Attribute {node.target} is not a tensor")

    attr_shape, attr_shape_signature = to_circle_shape(attr_tensor.shape)

    graph.add_tensor_from_scratch(
        prefix=node.name,
        shape=attr_shape,
        shape_signature=attr_shape_signature,
        dtype=to_circle_dtype(attr_tensor.dtype),
        source_node=node,
    )

    logger = logging.getLogger(__name__)
    logger.debug(f"Exported attribute tensor: {node.name}")
