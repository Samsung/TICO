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

import copy
from typing import Any, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx

import torch

from tico.passes.const_prop_pass import ConstPropPass
from tico.serialize.quant_param import QPARAM_KEY, QuantParam

_INVALID = object()


def _get_aten_op(op_name: str, overload_name: str) -> Any | None:
    """Return an ATen overload if it is registered."""

    try:
        overload_packet = getattr(torch.ops.aten, op_name)
    except AttributeError:
        return None

    try:
        return getattr(overload_packet, overload_name)
    except AttributeError:
        return None


_RESHAPE = _get_aten_op("reshape", "default")
_PERMUTE = _get_aten_op("permute", "default")
_TRANSPOSE = _get_aten_op("transpose", "int")
_SLICE = _get_aten_op("slice", "Tensor")
_SLICE_COPY = _get_aten_op("slice_copy", "Tensor")
_EXPAND = _get_aten_op("expand", "default")
_EXPAND_COPY = _get_aten_op("expand_copy", "default")
_SQUEEZE = _get_aten_op("squeeze", "dims")
_SQUEEZE_COPY = _get_aten_op("squeeze_copy", "dims")

_QPARAM_SAFE_TARGETS = tuple(
    op
    for op in (
        _RESHAPE,
        _PERMUTE,
        _TRANSPOSE,
        _SLICE,
        _SLICE_COPY,
        _EXPAND,
        _EXPAND_COPY,
        _SQUEEZE,
        _SQUEEZE_COPY,
    )
    if op is not None
)


def _get_argument_value(
    node: "torch.fx.Node",
    position: int,
    names: tuple[str, ...],
    default: Any = _INVALID,
) -> Any:
    """Return an FX node argument by position or keyword name."""

    if len(node.args) > position:
        return node.args[position]

    for name in names:
        if name in node.kwargs:
            return node.kwargs[name]

    return default


def _get_single_tensor_input(node: "torch.fx.Node") -> Optional["torch.fx.Node"]:
    """Return the first tensor node input of a single-input tensor transform."""

    value = _get_argument_value(node, 0, ("input", "self", "tensor"))
    if isinstance(value, torch.fx.Node):
        return value

    return None


def _get_tensor_shape(node: "torch.fx.Node") -> Optional[tuple[int, ...]]:
    """Return a static tensor shape from FX metadata if available."""

    val = node.meta.get("val", None)
    if isinstance(val, torch.Tensor):
        try:
            return tuple(int(dim) for dim in val.shape)
        except (TypeError, ValueError):
            return None

    tensor_meta = node.meta.get("tensor_meta", None)
    if tensor_meta is not None and hasattr(tensor_meta, "shape"):
        try:
            return tuple(int(dim) for dim in tensor_meta.shape)
        except (TypeError, ValueError):
            return None

    return None


def _normalize_dim(dim: int, rank: int) -> Optional[int]:
    """Normalize a possibly negative dimension for a tensor rank."""

    if dim < 0:
        dim += rank

    if dim < 0 or dim >= rank:
        return None

    return dim


def _normalize_dims(dims: Sequence[int], rank: int) -> Optional[list[int]]:
    """Normalize a dimension permutation for a tensor rank."""

    normalized = []
    for dim in dims:
        if not isinstance(dim, int):
            return None

        normalized_dim = _normalize_dim(dim, rank)
        if normalized_dim is None:
            return None

        normalized.append(normalized_dim)

    if sorted(normalized) != list(range(rank)):
        return None

    return normalized


def _copy_qparam(qparam: QuantParam) -> QuantParam:
    """Return a deep copy of a quantization parameter object."""

    return copy.deepcopy(qparam)


def _qparam_axis_values_are_valid(qparam: QuantParam, axis_len: int) -> bool:
    """Return whether per-axis qparam values match the quantized axis length."""

    for attr_name in ("scale", "zero_point", "min", "max"):
        values = getattr(qparam, attr_name)
        if values is not None and len(values) != axis_len:
            return False

    return True


def _get_source_qparam(node: "torch.fx.Node") -> QuantParam | None | object:
    """Return the qparam that should be preserved for a folded transform."""

    input_node = _get_single_tensor_input(node)
    if input_node is not None and QPARAM_KEY in input_node.meta:
        qparam = input_node.meta[QPARAM_KEY]
        if not isinstance(qparam, QuantParam):
            return _INVALID
        return qparam

    if QPARAM_KEY in node.meta:
        qparam = node.meta[QPARAM_KEY]
        if not isinstance(qparam, QuantParam):
            return _INVALID
        return qparam

    return None


def _derive_qparam_for_reshape(
    node: "torch.fx.Node",
    qparam: QuantParam,
) -> Optional[QuantParam]:
    """Derive output qparam for a folded reshape."""

    new_qparam = _copy_qparam(qparam)
    if new_qparam.quantized_dimension is None:
        return new_qparam

    input_node = _get_single_tensor_input(node)
    if input_node is None:
        return None

    input_shape = _get_tensor_shape(input_node)
    output_shape = _get_tensor_shape(node)

    # Per-channel reshape is only folded when it is shape-identical. General
    # reshape can change the semantic channel axis and needs a stronger proof.
    if input_shape is not None and input_shape == output_shape:
        qdim = _normalize_dim(new_qparam.quantized_dimension, len(input_shape))
        if qdim is None:
            return None
        if not _qparam_axis_values_are_valid(new_qparam, input_shape[qdim]):
            return None
        new_qparam.quantized_dimension = qdim
        return new_qparam

    return None


def _derive_qparam_for_permute(
    node: "torch.fx.Node",
    qparam: QuantParam,
) -> Optional[QuantParam]:
    """Derive output qparam for a folded permute."""

    input_node = _get_single_tensor_input(node)
    if input_node is None:
        return None

    input_shape = _get_tensor_shape(input_node)
    if input_shape is None:
        return None

    dims = _get_argument_value(node, 1, ("dims",))
    if not isinstance(dims, (tuple, list)):
        return None

    rank = len(input_shape)
    normalized_dims = _normalize_dims(dims, rank)
    if normalized_dims is None:
        return None

    new_qparam = _copy_qparam(qparam)
    if new_qparam.quantized_dimension is None:
        return new_qparam

    old_qdim = _normalize_dim(new_qparam.quantized_dimension, rank)
    if old_qdim is None:
        return None

    if not _qparam_axis_values_are_valid(new_qparam, input_shape[old_qdim]):
        return None

    new_qparam.quantized_dimension = normalized_dims.index(old_qdim)
    return new_qparam


def _derive_qparam_for_transpose(
    node: "torch.fx.Node",
    qparam: QuantParam,
) -> Optional[QuantParam]:
    """Derive output qparam for a folded transpose."""

    input_node = _get_single_tensor_input(node)
    if input_node is None:
        return None

    input_shape = _get_tensor_shape(input_node)
    if input_shape is None:
        return None

    dim0 = _get_argument_value(node, 1, ("dim0",))
    dim1 = _get_argument_value(node, 2, ("dim1",))
    if not isinstance(dim0, int) or not isinstance(dim1, int):
        return None

    rank = len(input_shape)
    dim0 = _normalize_dim(dim0, rank)
    dim1 = _normalize_dim(dim1, rank)
    if dim0 is None or dim1 is None:
        return None

    new_qparam = _copy_qparam(qparam)
    if new_qparam.quantized_dimension is None:
        return new_qparam

    old_qdim = _normalize_dim(new_qparam.quantized_dimension, rank)
    if old_qdim is None:
        return None

    if not _qparam_axis_values_are_valid(new_qparam, input_shape[old_qdim]):
        return None

    if old_qdim == dim0:
        new_qparam.quantized_dimension = dim1
    elif old_qdim == dim1:
        new_qparam.quantized_dimension = dim0
    else:
        new_qparam.quantized_dimension = old_qdim

    return new_qparam


def _slice_axis_values(
    values: Optional[list[Any]],
    index_slice: slice,
    axis_len: int,
) -> Optional[list[Any]] | object:
    """Slice qparam axis values or report that slicing is unsupported."""

    if values is None:
        return None

    if len(values) != axis_len:
        return _INVALID

    sliced = list(values[index_slice])
    if len(sliced) == 0:
        return _INVALID

    return sliced


def _derive_qparam_for_slice(
    node: "torch.fx.Node",
    qparam: QuantParam,
) -> Optional[QuantParam]:
    """Derive output qparam for a folded slice."""

    input_node = _get_single_tensor_input(node)
    if input_node is None:
        return None

    input_shape = _get_tensor_shape(input_node)
    if input_shape is None:
        return None

    rank = len(input_shape)
    new_qparam = _copy_qparam(qparam)

    if new_qparam.quantized_dimension is None:
        return new_qparam

    dim = _get_argument_value(node, 1, ("dim",), 0)
    start = _get_argument_value(node, 2, ("start",), None)
    end = _get_argument_value(node, 3, ("end",), None)
    step = _get_argument_value(node, 4, ("step",), 1)

    if not isinstance(dim, int):
        return None

    if start is not None and not isinstance(start, int):
        return None

    if end is not None and not isinstance(end, int):
        return None

    if step is None:
        step = 1

    if not isinstance(step, int) or step <= 0:
        return None

    dim = _normalize_dim(dim, rank)
    qdim = _normalize_dim(new_qparam.quantized_dimension, rank)
    if dim is None or qdim is None:
        return None

    if not _qparam_axis_values_are_valid(new_qparam, input_shape[qdim]):
        return None

    if dim != qdim:
        new_qparam.quantized_dimension = qdim
        return new_qparam

    axis_len = input_shape[qdim]
    normalized_start, normalized_end, normalized_step = slice(start, end, step).indices(
        axis_len
    )
    index_slice = slice(normalized_start, normalized_end, normalized_step)

    for attr_name in ("scale", "zero_point", "min", "max"):
        values = getattr(new_qparam, attr_name)
        sliced_values = _slice_axis_values(values, index_slice, axis_len)
        if sliced_values is _INVALID:
            return None
        setattr(new_qparam, attr_name, sliced_values)

    new_qparam.quantized_dimension = qdim
    return new_qparam


def _derive_qparam_for_expand(
    node: "torch.fx.Node",
    qparam: QuantParam,
) -> Optional[QuantParam]:
    """Derive output qparam for a folded expand."""

    input_node = _get_single_tensor_input(node)
    if input_node is None:
        return None

    input_shape = _get_tensor_shape(input_node)
    output_shape = _get_tensor_shape(node)
    if input_shape is None or output_shape is None:
        return None

    new_qparam = _copy_qparam(qparam)
    if new_qparam.quantized_dimension is None:
        return new_qparam

    if len(output_shape) < len(input_shape):
        return None

    old_qdim = _normalize_dim(new_qparam.quantized_dimension, len(input_shape))
    if old_qdim is None:
        return None

    if not _qparam_axis_values_are_valid(new_qparam, input_shape[old_qdim]):
        return None

    extending_rank = len(output_shape) - len(input_shape)
    new_qdim = old_qdim + extending_rank

    # Expanding the quantized axis from 1 to N would require duplicating qparams.
    if output_shape[new_qdim] != input_shape[old_qdim]:
        return None

    new_qparam.quantized_dimension = new_qdim
    return new_qparam


def _derive_qparam_for_squeeze(
    node: "torch.fx.Node",
    qparam: QuantParam,
) -> Optional[QuantParam]:
    """Derive output qparam for a folded squeeze."""

    new_qparam = _copy_qparam(qparam)
    if new_qparam.quantized_dimension is None:
        return new_qparam

    # Per-channel squeeze can remove or shift the quantized axis. Keep it
    # conservative until all squeeze axis cases are covered.
    return None


def _derive_folded_qparam(
    node: "torch.fx.Node",
    qparam: QuantParam,
) -> Optional[QuantParam]:
    """Derive the qparam metadata that a folded output placeholder should keep."""

    if _RESHAPE is not None and node.target == _RESHAPE:
        return _derive_qparam_for_reshape(node, qparam)

    if _PERMUTE is not None and node.target == _PERMUTE:
        return _derive_qparam_for_permute(node, qparam)

    if _TRANSPOSE is not None and node.target == _TRANSPOSE:
        return _derive_qparam_for_transpose(node, qparam)

    if node.target in {op for op in (_SLICE, _SLICE_COPY) if op is not None}:
        return _derive_qparam_for_slice(node, qparam)

    if node.target in {op for op in (_EXPAND, _EXPAND_COPY) if op is not None}:
        return _derive_qparam_for_expand(node, qparam)

    if node.target in {op for op in (_SQUEEZE, _SQUEEZE_COPY) if op is not None}:
        return _derive_qparam_for_squeeze(node, qparam)

    return None


def _is_qparam_safe_const_prop_node(node: "torch.fx.Node") -> bool:
    """Return whether a node may be folded by qparam-safe constant propagation."""

    if node.op != "call_function":
        return False

    if node.target not in _QPARAM_SAFE_TARGETS:
        return False

    if _get_single_tensor_input(node) is None:
        return False

    source_qparam = _get_source_qparam(node)
    if source_qparam is _INVALID:
        return False

    if source_qparam is None:
        return True

    assert isinstance(source_qparam, QuantParam)
    return _derive_folded_qparam(node, source_qparam) is not None


def _preserve_qparam_for_folded_node(
    node: "torch.fx.Node",
    prop_constant_tensor: torch.Tensor,
) -> None:
    """Attach preserved qparam metadata to a folded constant node."""

    del prop_constant_tensor

    source_qparam = _get_source_qparam(node)
    if source_qparam is None or source_qparam is _INVALID:
        return

    assert isinstance(source_qparam, QuantParam)
    folded_qparam = _derive_folded_qparam(node, source_qparam)
    if folded_qparam is None:
        return

    node.meta[QPARAM_KEY] = folded_qparam


class QParamSafeConstPropPass(ConstPropPass):
    """Fold constants only through qparam-preserving tensor transformations."""

    def __init__(self) -> None:
        super().__init__(
            fold_filter=_is_qparam_safe_const_prop_node,
            folded_node_callback=_preserve_qparam_for_folded_node,
        )
