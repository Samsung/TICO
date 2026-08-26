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

from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    is_buffer,
    is_lifted_tensor_constant,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY, QuantParam, to_qparam_dtype
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import get_quant_dtype
from tico.utils.validate_args_kwargs import (
    DequantizePerChannelArgs,
    DequantizePerTensorArgs,
)


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


def get_constant(exported_program: ExportedProgram, node: torch.fx.Node):
    """Return constant tensor data represented by a placeholder node."""
    assert isinstance(node, torch.fx.Node)
    if node.name in exported_program.constants:
        return exported_program.constants[node.name]
    elif is_buffer(exported_program, node):
        return get_buffer(exported_program, node)
    elif is_lifted_tensor_constant(exported_program, node):
        return get_lifted_tensor_constant(exported_program, node)
    else:
        raise RuntimeError("NYI constant")


def infer_dtype(quant_min: int, quant_max: int, dtype: torch.dtype) -> str:
    """Infer the logical dtype from the declared quantization range.

    PyTorch stores sub-byte affine tensors in the nearest available integer
    storage type. For example, UINT4 values are represented by ``torch.uint8``.
    The logical dtype must therefore be derived from ``quant_min`` and
    ``quant_max`` rather than from the values that happen to be present in the
    tensor. Inspecting the observed values can incorrectly classify a UINT8
    tensor as UINT4 when all of its codes happen to fall in ``[0, 15]``.
    """
    logical_dtype = get_quant_dtype(quant_min, quant_max)
    storage_dtype = to_qparam_dtype(dtype)
    expected_storage_dtype = {
        "uint4": "uint8",
        "int4": "int8",
    }.get(logical_dtype, logical_dtype)

    if storage_dtype != expected_storage_dtype:
        raise RuntimeError(
            "Quantization range and storage dtype are inconsistent: "
            f"range=({quant_min}, {quant_max}), "
            f"logical_dtype={logical_dtype}, storage_dtype={storage_dtype}."
        )

    return logical_dtype


def _same_optional_float_list(
    lhs: Optional[List[float]],
    rhs: Optional[List[float]],
) -> bool:
    """Return whether two optional float lists have identical values."""
    if lhs is None or rhs is None:
        return lhs is rhs

    if len(lhs) != len(rhs):
        return False

    return all(l == r for l, r in zip(lhs, rhs))


def _same_optional_int_list(
    lhs: Optional[List[int]],
    rhs: Optional[List[int]],
) -> bool:
    """Return whether two optional integer lists have identical values."""
    if lhs is None or rhs is None:
        return lhs is rhs

    if len(lhs) != len(rhs):
        return False

    return all(l == r for l, r in zip(lhs, rhs))


def _same_quant_param(lhs: QuantParam, rhs: QuantParam) -> bool:
    """Return whether two quantization parameter objects are equivalent."""
    return (
        _same_optional_float_list(lhs.scale, rhs.scale)
        and _same_optional_int_list(lhs.zero_point, rhs.zero_point)
        and lhs.quantized_dimension == rhs.quantized_dimension
        and _same_optional_float_list(lhs.min, rhs.min)
        and _same_optional_float_list(lhs.max, rhs.max)
        and lhs.dtype == rhs.dtype
    )


def _extract_dequantize_args(
    dq: torch.fx.Node,
) -> Optional[DequantizePerChannelArgs | DequantizePerTensorArgs]:
    """Extract typed arguments from a dequantize node."""
    dequantize_per_channel = _get_quantized_decomposed_default_op(
        "dequantize_per_channel"
    )
    dequantize_per_tensor = _get_quantized_decomposed_default_op(
        "dequantize_per_tensor"
    )

    if dequantize_per_channel is not None and dq.target == dequantize_per_channel:
        return DequantizePerChannelArgs(*dq.args, **dq.kwargs)

    if dequantize_per_tensor is not None and dq.target == dequantize_per_tensor:
        return DequantizePerTensorArgs(*dq.args, **dq.kwargs)

    return None


def _build_quant_param(
    exported_program: ExportedProgram,
    dq: torch.fx.Node,
    dq_args: DequantizePerChannelArgs | DequantizePerTensorArgs,
    q_weight: torch.fx.Node,
) -> QuantParam:
    """Build quantization parameters from a dequantize node."""
    q_weight_meta = q_weight.meta["val"]
    assert isinstance(q_weight_meta, FakeTensor)
    # Weight should have quantized values.
    assert q_weight_meta.dtype != torch.float

    quant_param = QuantParam()
    if isinstance(dq_args, DequantizePerChannelArgs):
        scales = get_constant(exported_program, dq_args.scales)
        zero_ps = get_constant(exported_program, dq_args.zero_points)

        # Sometimes users can give fp32 zero point. Let's update dtype here.
        zero_ps = zero_ps.to(torch.int64)
        quant_param.scale = scales.tolist()
        quant_param.zero_point = zero_ps.tolist()
        quant_param.quantized_dimension = dq_args.axis
        quant_param.dtype = infer_dtype(
            dq_args.quant_min,
            dq_args.quant_max,
            q_weight_meta.dtype,
        )
    elif isinstance(dq_args, DequantizePerTensorArgs):
        quant_param.scale = [dq_args.scale]
        quant_param.zero_point = [dq_args.zero_point]
        quant_param.dtype = infer_dtype(
            dq_args.quant_min,
            dq_args.quant_max,
            q_weight_meta.dtype,
        )
    else:
        raise RuntimeError(f"Invalid DQ target: {dq.target}")

    return quant_param


@trace_graph_diff_on_pass
class RemoveWeightDequantOp(PassBase):
    """Remove dequantize ops associated with quantized weights.

    Since weights are already quantized earlier, the final stage of the
    quantization pipeline typically does not require weight dequantize ops.

    If tied weights share a single quantized placeholder, multiple dequantize
    nodes can point to the same placeholder. In that case, the first dequantize
    node attaches quantization metadata to the weight, and later dequantize
    nodes are removed after verifying that their quantization parameters match.

    NOTE Removing a DQ causes a semantic change: f32 -> quantized.

    [BEFORE]
      W (quantized) - Dequantize (float)

    [AFTER]
      W (quantized)
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        dequantize_ops = _get_dequantize_ops()

        for dq in graph.nodes:
            if not dq.op == "call_function":
                continue

            if dq.target not in dequantize_ops:
                continue

            dq_args = _extract_dequantize_args(dq)
            if dq_args is None:
                raise RuntimeError(f"Invalid DQ target: {dq.target}")

            q_weight = dq_args.input
            # All weights are placeholders.
            if q_weight.op != "placeholder":
                continue

            quant_param = _build_quant_param(exported_program, dq, dq_args, q_weight)

            if QPARAM_KEY in q_weight.meta:
                existing_quant_param = q_weight.meta[QPARAM_KEY]
                if not isinstance(existing_quant_param, QuantParam):
                    raise RuntimeError(
                        f"{q_weight.name} has invalid quantization metadata."
                    )

                if not _same_quant_param(existing_quant_param, quant_param):
                    raise RuntimeError(
                        f"{dq.name} cannot be removed because {q_weight.name} "
                        "is already annotated with different quantization parameters."
                    )

                dq.replace_all_uses_with(q_weight, propagate_meta=False)
                logger.debug(f"{dq.name} is removed.")
                continue

            q_weight.meta[QPARAM_KEY] = quant_param
            dq.replace_all_uses_with(q_weight, propagate_meta=False)
            logger.debug(f"{dq.name} is removed.")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
