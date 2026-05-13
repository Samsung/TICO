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
from typing import Any, List, Mapping, Optional, TYPE_CHECKING

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


_MISSING = object()

TensorStorageIdentityKey = tuple[
    str,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    torch.dtype,
    torch.layout,
]
QuantizedTensorCacheKey = tuple[Any, ...]


def _get_quantized_decomposed_default_op(op_name: str) -> Any | None:
    """Return a quantized_decomposed default op if it is registered.

    Quantized decomposed ops may not be registered when this module is imported.
    Therefore, all access to torch.ops.quantized_decomposed must be lazy.
    """
    try:
        namespace = torch.ops.quantized_decomposed
    except AttributeError:
        return None

    try:
        overload_packet = getattr(namespace, op_name)
    except AttributeError:
        return None

    try:
        return overload_packet.default
    except AttributeError:
        return None


def _get_dequantize_ops() -> tuple[Any, ...]:
    """Return registered quantized_decomposed dequantize ops."""
    return tuple(
        op
        for op in (
            _get_quantized_decomposed_default_op("dequantize_per_channel"),
            _get_quantized_decomposed_default_op("dequantize_per_tensor"),
        )
        if op is not None
    )


def _get_quantize_ops() -> tuple[Any, ...]:
    """Return registered quantized_decomposed quantize ops."""
    return tuple(
        op
        for op in (
            _get_quantized_decomposed_default_op("quantize_per_channel"),
            _get_quantized_decomposed_default_op("quantize_per_tensor"),
        )
        if op is not None
    )


def _get_tensor_storage_identity_key(
    tensor: torch.Tensor,
) -> Optional[TensorStorageIdentityKey]:
    """Return a hashable key that identifies the logical storage of a tensor.

    The key is based on storage identity, not tensor contents. This avoids
    accidentally merging cloned tensors that happen to have the same values,
    while still detecting tied weights that share the same tensor storage.
    """
    if tensor.layout != torch.strided:
        return None

    if tensor.numel() == 0:
        # Empty tensors often have a zero data pointer, so unrelated empty
        # tensors may look identical.
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
    )


def _make_hashable_constant(value: Any) -> Any:
    """Convert constant values into hashable values for cache keys.

    Quantization parameters are part of the operation semantics, so tensor
    values used as quantization parameters are compared by value. The quantized
    weight input itself is handled separately by storage identity.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.layout != torch.strided:
            return ("tensor", str(tensor.layout), repr(tensor))

        cpu_tensor = tensor.cpu().contiguous()
        return (
            "tensor",
            cpu_tensor.dtype,
            tuple(cpu_tensor.shape),
            tuple(cpu_tensor.reshape(-1).tolist()),
        )

    if isinstance(value, (tuple, list)):
        return tuple(_make_hashable_constant(v) for v in value)

    if isinstance(value, dict):
        return tuple(sorted((k, _make_hashable_constant(v)) for k, v in value.items()))

    if isinstance(value, torch.dtype):
        return ("torch.dtype", str(value))

    if isinstance(value, torch.device):
        return ("torch.device", str(value))

    if isinstance(value, torch.layout):
        return ("torch.layout", str(value))

    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _get_argument_value(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    position: int,
    names: tuple[str, ...],
) -> Any:
    """Return an argument value by position or by one of its possible names."""
    if len(args) > position:
        return args[position]

    for name in names:
        if name in kwargs:
            return kwargs[name]

    return _MISSING


def _get_quantized_tensor_cache_key(
    node: torch.fx.Node,
    args_data: tuple[Any, ...],
    kwargs_data: Mapping[str, Any],
) -> Optional[QuantizedTensorCacheKey]:
    """Return a cache key for quantizing tied constant tensors.

    The source tensor is keyed by storage identity, while quantization
    parameters are keyed by value. This lets tied weights reuse a single
    propagated quantized tensor only when the quantization operation is
    semantically identical.
    """
    quantize_per_tensor = _get_quantized_decomposed_default_op("quantize_per_tensor")
    quantize_per_channel = _get_quantized_decomposed_default_op("quantize_per_channel")

    if quantize_per_tensor is not None and node.target == quantize_per_tensor:
        input_tensor = _get_argument_value(
            args_data, kwargs_data, 0, ("input", "tensor")
        )
        scale = _get_argument_value(args_data, kwargs_data, 1, ("scale",))
        zero_point = _get_argument_value(
            args_data, kwargs_data, 2, ("zero_point", "zero_p")
        )
        quant_min = _get_argument_value(args_data, kwargs_data, 3, ("quant_min",))
        quant_max = _get_argument_value(args_data, kwargs_data, 4, ("quant_max",))
        dtype = _get_argument_value(args_data, kwargs_data, 5, ("dtype",))

        if any(
            x is _MISSING
            for x in (input_tensor, scale, zero_point, quant_min, quant_max, dtype)
        ):
            return None

        if not isinstance(input_tensor, torch.Tensor):
            return None

        input_key = _get_tensor_storage_identity_key(input_tensor)
        if input_key is None:
            return None

        return (
            str(node.target),
            input_key,
            _make_hashable_constant(scale),
            _make_hashable_constant(zero_point),
            _make_hashable_constant(quant_min),
            _make_hashable_constant(quant_max),
            _make_hashable_constant(dtype),
        )

    if quantize_per_channel is not None and node.target == quantize_per_channel:
        input_tensor = _get_argument_value(
            args_data, kwargs_data, 0, ("input", "tensor")
        )
        scales = _get_argument_value(args_data, kwargs_data, 1, ("scales", "scale"))
        zero_points = _get_argument_value(
            args_data, kwargs_data, 2, ("zero_points", "zero_point", "zero_p")
        )
        axis = _get_argument_value(args_data, kwargs_data, 3, ("axis",))
        quant_min = _get_argument_value(args_data, kwargs_data, 4, ("quant_min",))
        quant_max = _get_argument_value(args_data, kwargs_data, 5, ("quant_max",))
        dtype = _get_argument_value(args_data, kwargs_data, 6, ("dtype",))

        if any(
            x is _MISSING
            for x in (
                input_tensor,
                scales,
                zero_points,
                axis,
                quant_min,
                quant_max,
                dtype,
            )
        ):
            return None

        if not isinstance(input_tensor, torch.Tensor):
            return None

        input_key = _get_tensor_storage_identity_key(input_tensor)
        if input_key is None:
            return None

        return (
            str(node.target),
            input_key,
            _make_hashable_constant(scales),
            _make_hashable_constant(zero_points),
            _make_hashable_constant(axis),
            _make_hashable_constant(quant_min),
            _make_hashable_constant(quant_max),
            _make_hashable_constant(dtype),
        )

    return None


def get_constant_placeholder_to_tensor_dict(
    exported_program: ExportedProgram,
) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    """Return a dictionary from constant placeholder nodes to constant tensors."""
    const_node_to_tensor: OrderedDict[torch.fx.Node, torch.Tensor] = OrderedDict()
    graph_module = exported_program.graph_module
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
    """Check whether an argument has constant data.

    Placeholder nodes are checked against the exported program's constant
    placeholder mapping because placeholders do not carry enough information by
    themselves to distinguish constants from user inputs.
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
    """Return concrete constant data for a constant argument."""
    if isinstance(arg, (tuple, list)):
        return (get_data(x, exported_program, const_node_to_tensor) for x in arg)
    elif isinstance(arg, _PRIMITIVE_TYPES):
        return arg
    elif arg in const_node_to_tensor:
        return const_node_to_tensor[arg]
    return None


def propagate_constants(
    exported_program: ExportedProgram,
) -> OrderedDict[torch.fx.Node, torch.Tensor]:
    """Propagate constants and return node-to-constant tensor mappings.

    Quantize ops are cached by tied source tensor identity and quantization
    parameters. This preserves tied weight sharing through constant propagation:
    two quantize nodes that quantize the same tied weight with the same
    quantization parameters reuse the same propagated quantized tensor object.
    """
    const_node_to_tensor = get_constant_placeholder_to_tensor_dict(exported_program)
    quantized_tensor_cache: dict[QuantizedTensorCacheKey, torch.Tensor] = {}

    dequantize_ops = _get_dequantize_ops()
    quantize_ops = _get_quantize_ops()

    graph_module = exported_program.graph_module
    graph: torch.fx.Graph = graph_module.graph
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in dequantize_ops:
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

        quantized_tensor_cache_key = None
        if node.target in quantize_ops:
            quantized_tensor_cache_key = _get_quantized_tensor_cache_key(
                node, args_data, kwargs_data
            )
            if (
                quantized_tensor_cache_key is not None
                and quantized_tensor_cache_key in quantized_tensor_cache
            ):
                const_node_to_tensor[node] = quantized_tensor_cache[
                    quantized_tensor_cache_key
                ]
                continue

        # Propagate constant because all of its args are constant tensors.
        with torch.no_grad():
            prop_constant_tensor = node.target(*args_data, **kwargs_data)

        if quantized_tensor_cache_key is not None:
            quantized_tensor_cache[quantized_tensor_cache_key] = prop_constant_tensor

        const_node_to_tensor[node] = prop_constant_tensor

    return const_node_to_tensor


def erase_constant_node(
    exported_program: ExportedProgram,
    node: torch.fx.Node,
) -> None:
    """Remove the corresponding tensor from parameter or constant dictionaries.

    The input signature maps do not need to be updated here because the final
    input specs are rebuilt at the end of this pass.
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
) -> List[torch.fx.Node]:
    """Create constant placeholder nodes for propagated constant tensors.

    If multiple propagated nodes share the same tensor object, only one
    placeholder is created and the other nodes are replaced by that placeholder.
    This is used for tied weights whose quantize ops are cached by
    `propagate_constants`.
    """
    placeholders = []
    tensor_id_to_placeholder: dict[int, torch.fx.Node] = {}

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

        tensor_id = id(prop_constant_tensor)
        if tensor_id in tensor_id_to_placeholder:
            const_placeholder_node = tensor_id_to_placeholder[tensor_id]
            node.replace_all_uses_with(const_placeholder_node, propagate_meta=False)
            exported_program.graph.erase_node(node)
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

        tensor_id_to_placeholder[tensor_id] = const_placeholder_node
        placeholders.append(const_placeholder_node)

    return placeholders


def create_input_specs(
    placeholders: List[torch.fx.Node],
) -> dict[str, InputSpec]:
    """Create input specs for newly created constant placeholders."""
    name_to_spec: dict[str, InputSpec] = {}

    # https://pytorch.org/docs/stable/export.ir_spec.html#placeholder
    # %name = placeholder[target = name](args = ())
    for node in placeholders:
        name_to_spec[node.name] = create_input_spec(node, InputKind.CONSTANT_TENSOR)

    return name_to_spec


@trace_graph_diff_on_pass
@trace_const_diff_on_pass
class ConstPropPass(PassBase):
    """Perform constant folding and constant propagation.

    The exported program guarantees that parameters, buffers, and constant
    tensors are lifted out of the graph as inputs. Therefore, this pass updates
    input specs after folding constant nodes.

    [WHAT IT DOES]
    [1] Propagate the constants.
    [2] Get propagated data from constant nodes.
    [3] Create the constant placeholder nodes according to the propagated data.
    [4] Create input specs according to the created placeholders.
    [5] Update the input specs.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph

        # [1], [2]
        const_node_to_tensor: OrderedDict[
            torch.fx.Node, torch.Tensor
        ] = propagate_constants(exported_program)
        # [3]
        placeholders = create_constant_placeholder(
            const_node_to_tensor, exported_program
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
