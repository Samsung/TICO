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
from torch.export.exported_program import (
    ConstantArgument,
    ExportedProgram,
    InputKind,
    TensorArgument,
)

from tico.serialize.circle_mapping import to_circle_dtype
from tico.serialize.operators import *
from tico.serialize.circle_graph import CircleModel, CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import get_node_visitors
from tico.utils import logging
from tico.utils.serialize import finalise_tensor_names


multiple_output_ops = [
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.max.dim,
]

# Build circle model from ExportedProgram
# Return raw bytes of circle model
def build_circle(edge_program: ExportedProgram) -> bytes:
    logger = logging.getLogger(__name__)

    builder = flatbuffers.Builder()

    # Init Model
    model = CircleModel()

    # Add empty buffer at the front (convention)
    model.add_buffer(circle.Buffer.BufferT())

    # Create an empty subgraph (assume a single subgraph)
    graph = CircleSubgraph(model)

    # Export tensors
    logger.debug("---------------Export tensors--------------")
    buf_name_to_data = {name: buf for name, buf in edge_program.named_buffers()}
    for node in edge_program.graph.nodes:
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

        # placeholder: function input (including parameters, buffers, constant tensors)
        elif node.op == "placeholder":
            # placeholder invariants
            assert node.args is None or len(node.args) == 0  # Not support default param

            # parameters
            if node.name in edge_program.graph_signature.inputs_to_parameters:
                param_name = edge_program.graph_signature.inputs_to_parameters[
                    node.name
                ]
                param_data = edge_program.state_dict[param_name]

                assert isinstance(
                    param_data, torch.Tensor
                ), "Expect parameters to be a tensor"
                param_value = param_data.cpu().detach().numpy()

                graph.add_tensor_from_node(node, param_value)
                logger.debug(f"placeholder(param): {node.name} tensor exported.")
            elif node.name in edge_program.graph_signature.inputs_to_buffers:
                buffer_name = edge_program.graph_signature.inputs_to_buffers[node.name]
                assert buffer_name in buf_name_to_data
                buffer_data = buf_name_to_data[buffer_name]
                assert isinstance(
                    buffer_data, torch.Tensor
                ), "Expect buffers to be a tensor"
                buffer_value = buffer_data.cpu().detach().numpy()

                graph.add_tensor_from_node(node, buffer_value)
                logger.debug(f"placeholder(buffer): {node.name} tensor exported.")
            elif (
                node.name
                in edge_program.graph_signature.inputs_to_lifted_tensor_constants
            ):
                ctensor_name = (
                    edge_program.graph_signature.inputs_to_lifted_tensor_constants[
                        node.name
                    ]
                )
                ctensor_data = edge_program.constants[ctensor_name]

                assert isinstance(
                    ctensor_data, torch.Tensor
                ), "Expect constant tensor to be a tensor"
                ctensor_value = ctensor_data.cpu().detach().numpy()

                graph.add_tensor_from_node(node, ctensor_value)
                logger.debug(
                    f"placeholder(constant tensor): {node.name} tensor exported."
                )
            else:
                user_inputs = [
                    specs
                    for specs in edge_program.graph_signature.input_specs
                    if specs.kind == InputKind.USER_INPUT
                ]
                constant_inputs = [
                    specs
                    for specs in user_inputs
                    if isinstance(specs.arg, ConstantArgument)
                ]
                name_to_value = {
                    specs.arg.name: specs.arg.value for specs in constant_inputs
                }
                # NoneType ConstantArgument is ignored.
                if node.name in name_to_value and name_to_value[node.name] == None:
                    continue
                graph.add_tensor_from_node(node)
                logger.debug(f"placeholder: {node.name} tensor exported.")

        # get_attr: retrieve parameter
        elif node.op == "get_attr":
            # node.name: Place where fetched attribute is saved
            # node.target: Attribute in the module
            attr_tensor = getattr(node.graph.owning_module, node.target)
            assert isinstance(attr_tensor, torch.Tensor)

            graph.add_tensor_from_scratch(
                prefix=node.name,
                shape=list(attr_tensor.shape),
                dtype=to_circle_dtype(attr_tensor.dtype),
                source_node=node,
            )

            logger.debug(f"get_attr: {node.name} tensor exported.")

        # output: function output
        elif node.op == "output":
            # output node itself does not need a buffer
            # argument of output node is assumed to be exported beforehand
            for output in node.args[0]:
                if isinstance(output, torch.fx.Node):
                    assert graph.has_tensor(output.name)
            continue

        # call_method: call method
        elif node.op == "call_method":
            raise AssertionError("Not yet implemented")

        # call_module: call 'forward' of module
        elif node.op == "call_module":
            raise AssertionError("Not yet implemented")

        else:
            # Add more if fx.Node is extended
            raise AssertionError(f"Unknown fx.Node op {node.op}")

    # Register inputs
    logger.debug("---------------Register inputs--------------")
    for in_spec in edge_program.graph_signature.input_specs:
        if in_spec.kind != InputKind.USER_INPUT:
            continue
        # NoneType ConstantArgument is ignored.
        if isinstance(in_spec.arg, ConstantArgument) and in_spec.arg.value == None:
            continue
        arg_name = in_spec.arg.name
        graph.add_input(arg_name)
        logger.debug(f"Registered input: {arg_name}")

    # Register outputs
    logger.debug("---------------Register outputs--------------")
    for user_output in edge_program.graph_signature.user_outputs:
        if user_output == None:
            logger.debug(f"Ignore 'None' output")
            continue

        graph.add_output(user_output)
        logger.debug(f"Registered output: {user_output}")

    # Export operators
    logger.debug("---------------Export operators--------------")
    op_codes: Dict[OpCode, int] = {}
    visitors = get_node_visitors(op_codes, graph)
    for node in edge_program.graph.nodes:
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

    # Register subgraph
    finalise_tensor_names(graph)
    model.subgraphs.append(graph)

    # Encode operator codes
    model.operatorCodes = [
        code for code, _ in sorted(op_codes.items(), key=lambda x: x[1])
    ]

    # Description
    model.description = "circle"

    # Set version
    model.version = 0

    # Finish model
    builder.Finish(model.Pack(builder), "CIR0".encode("utf8"))
    buf = builder.Output()

    return bytes(buf)
