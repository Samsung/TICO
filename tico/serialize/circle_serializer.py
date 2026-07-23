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
from typing import Dict, Optional

import flatbuffers
import torch
from torch.export.exported_program import ConstantArgument, ExportedProgram, InputKind

from tico.config import CompileConfigBase, get_default_config
from tico.serialize.operators import *
from tico.serialize.circle_graph import CircleModel, CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import get_node_visitors
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.serialize import finalise_tensor_names, validate_tensor_shapes


multiple_output_ops = [
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.max.dim,
]


QuantizationAliasKey = tuple[
    tuple[float, ...] | None,
    tuple[int, ...] | None,
    int | None,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    str,
]
TensorAliasKey = tuple[
    str,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    torch.dtype,
    torch.layout,
    QuantizationAliasKey | None,
]
SharedTensorRegistry = Dict[TensorAliasKey, str]


def _initialize_model() -> tuple[CircleModel, CircleSubgraph]:
    """Initialize a new Circle model and subgraph.

    Returns:
        Tuple containing the model and subgraph
    """
    model = CircleModel()
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


def _get_quantization_alias_key(node: torch.fx.Node) -> QuantizationAliasKey | None:
    """Return a hashable representation of a node's quantization metadata."""
    qparam = node.meta.get(QPARAM_KEY)
    if qparam is None:
        return None

    return (
        tuple(qparam.scale) if qparam.scale is not None else None,
        tuple(qparam.zero_point) if qparam.zero_point is not None else None,
        qparam.quantized_dimension,
        tuple(qparam.min) if qparam.min is not None else None,
        tuple(qparam.max) if qparam.max is not None else None,
        qparam.dtype,
    )


def _get_tensor_alias_key(
    tensor: torch.Tensor,
    node: torch.fx.Node,
) -> Optional[TensorAliasKey]:
    """
    Return a key that identifies shared logical tensor storage.

    The key is based on tensor identity, not tensor contents. This lets tied
    parameters, such as tied embeddings and LM-head weights, reuse the same
    Circle tensor while avoiding accidental deduplication of cloned tensors that
    merely contain equal values.
    """
    if tensor.layout != torch.strided:
        return None

    if tensor.numel() == 0:
        # Empty tensors often have a zero data pointer, which can cause
        # unrelated empty tensors to look identical.
        return None

    try:
        data_ptr = tensor.data_ptr()
    except RuntimeError:
        return None

    if data_ptr == 0:
        return None

    return (
        str(tensor.device),
        data_ptr,
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.layout,
        _get_quantization_alias_key(node),
    )


def _register_tensor_alias(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    canonical_tensor_name: str,
) -> None:
    """Register an FX node name as an alias of an existing Circle tensor."""
    if canonical_tensor_name not in graph.name_to_tid:
        raise KeyError(f"{canonical_tensor_name} is not registered.")

    canonical_tid = graph.name_to_tid[canonical_tensor_name]
    existing_tid = graph.name_to_tid.get(node.name)

    if existing_tid is not None:
        if existing_tid != canonical_tid:
            raise ValueError(
                f"Cannot alias {node.name} to {canonical_tensor_name}: "
                f"{node.name} is already registered to tensor id {existing_tid}."
            )
        return

    graph.name_to_tid[node.name] = canonical_tid


def _add_tensor_from_node_with_data_sharing(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    tensor_data: torch.Tensor,
    shared_tensors: SharedTensorRegistry,
) -> str | None:
    """Add a data-backed tensor, reusing a Circle tensor for shared storage.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The FX node that represents the tensor
        tensor_data: Tensor data associated with the node
        shared_tensors: Mapping from tensor identity keys to Circle tensor names

    Returns:
        The reused Circle tensor name when the node is registered as an alias.
        Returns None when a new Circle tensor is created.
    """
    alias_key = _get_tensor_alias_key(tensor_data, node)

    if alias_key is not None and alias_key in shared_tensors:
        canonical_tensor_name = shared_tensors[alias_key]
        _register_tensor_alias(graph, node, canonical_tensor_name)
        return canonical_tensor_name

    tensor_value = tensor_data.detach().cpu().numpy()
    graph.add_tensor_from_node(node, tensor_value)

    if alias_key is not None:
        shared_tensors[alias_key] = graph.tensors[-1].name

    return None


def _export_tensors(graph: CircleSubgraph, ep: ExportedProgram) -> None:
    """Export all tensors from the exported program to the circle graph.

    Args:
        graph: The CircleSubgraph to add tensors to
        ep: The exported PyTorch program
    """
    logger = logging.getLogger(__name__)
    logger.debug("---------------Export tensors--------------")
    buf_name_to_data = {name: buf for name, buf in ep.named_buffers()}
    shared_tensors: SharedTensorRegistry = {}

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
            _handle_placeholder_node(graph, node, ep, buf_name_to_data, shared_tensors)

        elif node.op == "get_attr":
            _handle_get_attr_node(node)

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
    shared_tensors: SharedTensorRegistry,
) -> None:
    """Handle a placeholder node during tensor export."""
    # placeholder invariants
    assert node.args is None or len(node.args) == 0  # Not support default param

    if node.name in ep.graph_signature.inputs_to_parameters:
        _handle_parameter_node(graph, node, ep, shared_tensors)
    elif node.name in ep.graph_signature.inputs_to_buffers:
        _handle_buffer_node(graph, node, ep, buf_name_to_data, shared_tensors)
    elif node.name in ep.graph_signature.inputs_to_lifted_tensor_constants:
        _handle_constant_tensor_node(graph, node, ep, shared_tensors)
    else:
        _handle_user_input_node(graph, node, ep)


def _handle_parameter_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    ep: ExportedProgram,
    shared_tensors: SharedTensorRegistry,
) -> None:
    """Handle a parameter placeholder node by exporting its tensor data.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The parameter node to process
        ep: ExportedProgram containing parameter data
        shared_tensors: Mapping used to reuse Circle tensors for shared data
    """
    param_name = ep.graph_signature.inputs_to_parameters[node.name]
    param_data = ep.state_dict[param_name]

    if not isinstance(param_data, torch.Tensor):
        raise ValueError(f"Parameter {param_name} is not a tensor")

    reused_tensor_name = _add_tensor_from_node_with_data_sharing(
        graph, node, param_data, shared_tensors
    )

    logger = logging.getLogger(__name__)
    if reused_tensor_name is None:
        logger.debug(f"Exported parameter tensor: {node.name}")
    else:
        logger.debug(
            f"Reused shared parameter tensor: {node.name} -> {reused_tensor_name}"
        )


def _handle_buffer_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    ep: ExportedProgram,
    buf_name_to_data: dict,
    shared_tensors: SharedTensorRegistry,
) -> None:
    """Handle a buffer placeholder node by exporting its tensor data.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The buffer node to process
        ep: ExportedProgram containing buffer info
        buf_name_to_data: Mapping of buffer names to data
        shared_tensors: Mapping used to reuse Circle tensors for shared data
    """
    buffer_name = ep.graph_signature.inputs_to_buffers[node.name]

    if buffer_name not in buf_name_to_data:
        raise ValueError(f"Buffer {buffer_name} not found in buffer data")

    buffer_data = buf_name_to_data[buffer_name]

    if not isinstance(buffer_data, torch.Tensor):
        raise ValueError(f"Buffer {buffer_name} is not a tensor")

    reused_tensor_name = _add_tensor_from_node_with_data_sharing(
        graph, node, buffer_data, shared_tensors
    )

    logger = logging.getLogger(__name__)
    if reused_tensor_name is None:
        logger.debug(f"Exported buffer tensor: {node.name}")
    else:
        logger.debug(
            f"Reused shared buffer tensor: {node.name} -> {reused_tensor_name}"
        )


def _handle_constant_tensor_node(
    graph: CircleSubgraph,
    node: torch.fx.Node,
    ep: ExportedProgram,
    shared_tensors: SharedTensorRegistry,
) -> None:
    """Handle a constant tensor placeholder node by exporting its tensor data.

    Args:
        graph: CircleSubgraph to add tensor to
        node: The constant tensor node to process
        ep: ExportedProgram containing constant data
        shared_tensors: Mapping used to reuse Circle tensors for shared data
    """
    ctensor_name = ep.graph_signature.inputs_to_lifted_tensor_constants[node.name]

    if ctensor_name not in ep.constants:
        raise ValueError(f"Constant tensor {ctensor_name} not found")

    ctensor_data = ep.constants[ctensor_name]

    if not isinstance(ctensor_data, torch.Tensor):
        raise ValueError(f"Constant tensor {ctensor_name} is not a tensor")

    reused_tensor_name = _add_tensor_from_node_with_data_sharing(
        graph, node, ctensor_data, shared_tensors
    )

    logger = logging.getLogger(__name__)
    if reused_tensor_name is None:
        logger.debug(f"Exported constant tensor: {node.name}")
    else:
        logger.debug(
            f"Reused shared constant tensor: {node.name} -> {reused_tensor_name}"
        )


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


def _handle_get_attr_node(node: torch.fx.Node) -> None:
    """
    Reject unsupported attributes in the top-level ExportedProgram graph.

    Parameters, buffers, and tensor constants in a valid top-level
    ExportedProgram graph must be lifted to placeholder nodes. A get_attr node
    therefore represents either an unsupported auxiliary object, such as a
    nested GraphModule used by a higher-order operator, or an ExportedProgram
    invariant violation.

    Args:
        node: The get_attr node to inspect.

    Raises:
        RuntimeError: If the get_attr node is malformed or references a tensor
            that should have been lifted.
        NotYetSupportedError: If the node references an unsupported non-tensor
            attribute such as a nested GraphModule.
    """
    if not isinstance(node.target, str):
        raise RuntimeError(
            "Expected get_attr target to be a string, "
            f"but got {type(node.target).__name__}."
        )

    attr = getattr(node.graph.owning_module, node.target)

    if isinstance(attr, torch.Tensor):
        raise RuntimeError(
            f"Unexpected tensor get_attr '{node.target}'. Parameters, buffers, "
            "and tensor constants must be lifted as ExportedProgram inputs."
        )

    if isinstance(attr, torch.fx.GraphModule):
        raise NotYetSupportedError(
            f"ExportedProgram contains nested GraphModule '{node.target}'. "
            "Higher-order operators and control-flow subgraphs are not supported."
        )

    raise NotYetSupportedError(
        f"Unsupported top-level get_attr '{node.target}' with type "
        f"{type(attr).__name__}."
    )
