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

"""Legalize quantized Clamp operands into one affine quantization domain."""

from __future__ import annotations

import copy
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx

import torch
from torch._export.utils import get_buffer, get_lifted_tensor_constant, get_param
from torch.export import ExportedProgram

from tico.passes import ops
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import add_placeholder
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import (
    trace_const_diff_on_pass,
    trace_graph_diff_on_pass,
)
from tico.utils.utils import quant_min_max, set_new_meta_val
from tico.utils.validate_args_kwargs import ClampArgs


_SUPPORTED_DTYPES: dict[str, torch.dtype] = {
    "uint8": torch.uint8,
    "int16": torch.int16,
}


def _copy_qparam(qparam: QuantParam) -> QuantParam:
    """Return an independent copy of one quantization-domain descriptor."""
    return copy.deepcopy(qparam)


def _same_quantization_domain(lhs: QuantParam, rhs: QuantParam) -> bool:
    """Return whether two qparams encode values in the same integer domain."""
    return (
        lhs.dtype == rhs.dtype
        and lhs.scale == rhs.scale
        and lhs.zero_point == rhs.zero_point
        and lhs.quantized_dimension == rhs.quantized_dimension
    )


def _validate_qparam(
    qparam: QuantParam,
    *,
    context: str,
) -> tuple[float, int, int, int, torch.dtype]:
    """Validate and unpack a supported per-tensor affine qparam."""
    if qparam.dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_DTYPES))
        raise NotYetSupportedError(
            f"Quantized Clamp at {context} supports only {supported}, "
            f"got {qparam.dtype!r}."
        )
    if qparam.quantized_dimension is not None:
        raise NotYetSupportedError(
            f"Quantized Clamp at {context} requires per-tensor qparams."
        )
    if qparam.scale is None or len(qparam.scale) != 1:
        raise ValueError(f"Quantized Clamp at {context} requires exactly one scale.")
    if qparam.zero_point is None or len(qparam.zero_point) != 1:
        raise ValueError(
            f"Quantized Clamp at {context} requires exactly one zero point."
        )

    scale = float(qparam.scale[0])
    zero_point = int(qparam.zero_point[0])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Quantized Clamp at {context} has an invalid scale: {scale}.")

    qmin, qmax = quant_min_max(qparam.dtype)
    if zero_point < qmin or zero_point > qmax:
        raise ValueError(
            f"Quantized Clamp at {context} has zero_point={zero_point} outside "
            f"[{qmin}, {qmax}]."
        )
    return scale, zero_point, qmin, qmax, _SUPPORTED_DTYPES[qparam.dtype]


def _get_canonical_qparam(
    node: "torch.fx.Node",
    input_node: "torch.fx.Node",
) -> QuantParam | None:
    """Choose the Clamp output domain, falling back to its input domain."""
    qparam = node.meta.get(QPARAM_KEY)
    if qparam is not None:
        if not isinstance(qparam, QuantParam):
            raise TypeError(
                f"Clamp node {node.name} has non-QuantParam qparam metadata."
            )
        return qparam

    input_qparam = input_node.meta.get(QPARAM_KEY)
    if input_qparam is None:
        return None
    if not isinstance(input_qparam, QuantParam):
        raise TypeError(
            f"Clamp input {input_node.name} has non-QuantParam qparam metadata."
        )

    node.meta[QPARAM_KEY] = _copy_qparam(input_qparam)
    return node.meta[QPARAM_KEY]


def _is_user_input(exported_program: ExportedProgram, node: "torch.fx.Node") -> bool:
    """Return whether a placeholder is a runtime tensor input."""
    return (
        node.op == "placeholder"
        and node.name in exported_program.graph_signature.user_inputs
    )


def _insert_requantize_before(
    node: "torch.fx.Node",
    input_node: "torch.fx.Node",
    qparam: QuantParam,
) -> "torch.fx.Node":
    """Insert a per-tensor Quantize that converts Clamp input to its output domain."""
    scale, zero_point, qmin, qmax, dtype = _validate_qparam(
        qparam,
        context=node.name,
    )
    graph = node.graph
    with graph.inserting_before(node):
        quantize = graph.call_function(
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            args=(input_node, scale, zero_point, qmin, qmax, dtype),
        )
        if "nn_module_stack" in node.meta:
            quantize.meta["nn_module_stack"] = copy.deepcopy(
                node.meta["nn_module_stack"]
            )
        quantize.meta[QPARAM_KEY] = _copy_qparam(qparam)
        set_new_meta_val(quantize)

    node.replace_input_with(input_node, quantize)
    return quantize


def _ensure_input_domain(
    exported_program: ExportedProgram,
    node: "torch.fx.Node",
    input_node: "torch.fx.Node",
    qparam: QuantParam,
) -> "torch.fx.Node":
    """Return a Clamp input represented in the canonical qparam domain."""
    input_qparam = input_node.meta.get(QPARAM_KEY)
    if input_qparam is None:
        if _is_user_input(exported_program, input_node):
            input_node.meta[QPARAM_KEY] = _copy_qparam(qparam)
            return input_node
        return _insert_requantize_before(node, input_node, qparam)

    if not isinstance(input_qparam, QuantParam):
        raise TypeError(
            f"Clamp input {input_node.name} has non-QuantParam qparam metadata."
        )
    if _same_quantization_domain(input_qparam, qparam):
        return input_node
    return _insert_requantize_before(node, input_node, qparam)


def _get_constant_tensor(
    exported_program: ExportedProgram,
    value: Any,
) -> torch.Tensor | None:
    """Resolve a literal or state-backed FX value to a concrete tensor."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return torch.tensor(value)
    if isinstance(value, torch.Tensor):
        return value.detach()
    if not isinstance(value, torch.fx.Node):
        return None

    for getter in (get_buffer, get_lifted_tensor_constant, get_param):
        tensor = getter(exported_program, value)
        if tensor is not None:
            return tensor.detach()

    meta_val = value.meta.get("val")
    constant = getattr(meta_val, "constant", None)
    if isinstance(constant, torch.Tensor):
        return constant.detach()
    return None


def _dequantize_constant(
    tensor: torch.Tensor,
    qparam: QuantParam,
    *,
    context: str,
) -> torch.Tensor:
    """Convert an integer constant back to its represented real values."""
    scale, zero_point, _, _, dtype = _validate_qparam(qparam, context=context)
    if tensor.dtype != dtype:
        raise ValueError(
            f"Quantized Clamp bound {context} has qparam dtype {qparam.dtype} "
            f"but stores {tensor.dtype}."
        )
    return (tensor.to(torch.float64) - zero_point) * scale


def _quantize_real_bound(
    tensor: torch.Tensor,
    qparam: QuantParam,
    *,
    context: str,
) -> torch.Tensor:
    """Quantize one real-valued bound into a Clamp activation domain."""
    scale, zero_point, qmin, qmax, dtype = _validate_qparam(
        qparam,
        context=context,
    )
    real = tensor.detach().to(device="cpu", dtype=torch.float64)
    if torch.isnan(real).any():
        raise ValueError(f"Quantized Clamp bound {context} contains NaN.")

    quantized = torch.round(real / scale) + zero_point
    quantized = torch.clamp(quantized, min=qmin, max=qmax)
    return quantized.to(dtype=dtype)


def _is_identity_bound(
    tensor: torch.Tensor,
    *,
    is_minimum: bool,
    qparam: QuantParam,
    context: str,
) -> bool:
    """Return whether an integer bound leaves the full dtype range untouched."""
    _, _, qmin, qmax, dtype = _validate_qparam(qparam, context=context)
    if tensor.dtype != dtype:
        return False
    identity_value = qmin if is_minimum else qmax
    return bool(torch.all(tensor == identity_value).item())


def _legalize_bound(
    exported_program: ExportedProgram,
    clamp_node: "torch.fx.Node",
    value: Any,
    *,
    name: str,
    is_minimum: bool,
    qparam: QuantParam,
) -> tuple[Any, bool]:
    """Return a quantized bound node, or None when the bound is a no-op."""
    if value is None:
        return None, False

    constant = _get_constant_tensor(exported_program, value)
    value_qparam = (
        value.meta.get(QPARAM_KEY) if isinstance(value, torch.fx.Node) else None
    )

    if value_qparam is not None:
        if not isinstance(value_qparam, QuantParam):
            raise TypeError(
                "Clamp bound "
                f"{getattr(value, 'name', name)} has invalid qparam metadata."
            )
        if _same_quantization_domain(value_qparam, qparam):
            if constant is None:
                return value, False
            if _is_identity_bound(
                constant,
                is_minimum=is_minimum,
                qparam=qparam,
                context=f"{clamp_node.name}.{name}",
            ):
                return None, True
            _, _, _, _, dtype = _validate_qparam(
                qparam,
                context=f"{clamp_node.name}.{name}",
            )
            if constant.dtype == dtype:
                return value, False

    if constant is None:
        raise NotYetSupportedError(
            f"Quantized Clamp bound {clamp_node.name}.{name} must be a constant "
            "or already carry the Clamp qparam domain."
        )

    real_bound = constant
    if value_qparam is not None:
        real_bound = _dequantize_constant(
            constant,
            value_qparam,
            context=f"{clamp_node.name}.{name}",
        )

    quantized = _quantize_real_bound(
        real_bound,
        qparam,
        context=f"{clamp_node.name}.{name}",
    )
    if _is_identity_bound(
        quantized,
        is_minimum=is_minimum,
        qparam=qparam,
        context=f"{clamp_node.name}.{name}",
    ):
        return None, True

    quantized_node = add_placeholder(
        exported_program,
        quantized,
        prefix=f"{clamp_node.name}_{name}_quantized_",
    )
    quantized_node.meta[QPARAM_KEY] = _copy_qparam(qparam)
    if "nn_module_stack" in clamp_node.meta:
        quantized_node.meta["nn_module_stack"] = copy.deepcopy(
            clamp_node.meta["nn_module_stack"]
        )
    return quantized_node, True


def _set_optional_argument(
    node: "torch.fx.Node",
    *,
    position: int,
    keyword: str,
    value: Any,
) -> None:
    """Set one positional-or-keyword Clamp argument."""
    if len(node.args) > position:
        node.update_arg(position, value)
        return

    kwargs = dict(node.kwargs)
    kwargs[keyword] = value
    node.kwargs = kwargs


@trace_graph_diff_on_pass
@trace_const_diff_on_pass
class LegalizeQuantizedClamp(PassBase):
    """Quantize Clamp bounds and align every operand to one affine domain.

    Circle represents ``aten.clamp`` as ``MINIMUM`` followed by ``MAXIMUM``.
    Integer comparison is valid only when the activation and both bounds use the
    same dtype, scale, and zero point. This pass therefore:

    - selects the Clamp output qparam as the canonical domain;
    - requantizes the activation input when its domain differs;
    - replaces constant real-valued bounds with integer constants carrying the
      canonical qparam; and
    - removes bounds, or the complete Clamp, when they cover the full integer
      range.

    Backend-specific lowering and Linear/Clamp fusion intentionally remain NPU
    compiler responsibilities.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in ops.aten.clamp:
                continue

            args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            input_node = args.input
            if not isinstance(input_node, torch.fx.Node):
                raise TypeError(f"Clamp input {node.name} must be an FX tensor node.")

            qparam = _get_canonical_qparam(node, input_node)
            if qparam is None:
                # A floating-point Clamp is outside this quantization pass.
                continue
            _validate_qparam(qparam, context=node.name)
            node.meta[QPARAM_KEY] = _copy_qparam(qparam)

            canonical_input = _ensure_input_domain(
                exported_program,
                node,
                input_node,
                qparam,
            )
            modified |= canonical_input is not input_node

            min_value, min_modified = _legalize_bound(
                exported_program,
                node,
                args.min,
                name="min",
                is_minimum=True,
                qparam=qparam,
            )
            max_value, max_modified = _legalize_bound(
                exported_program,
                node,
                args.max,
                name="max",
                is_minimum=False,
                qparam=qparam,
            )
            modified |= min_modified or max_modified

            if min_value is None and max_value is None:
                node.replace_all_uses_with(canonical_input, propagate_meta=False)
                modified = True
                logger.debug(
                    f"Removed identity quantized Clamp {node.name} in "
                    f"{qparam.dtype} domain."
                )
                continue

            _set_optional_argument(node, position=1, keyword="min", value=min_value)
            _set_optional_argument(node, position=2, keyword="max", value=max_value)

            # Quantized bounds are tensor constants. Normalize the scalar overload
            # to the tensor overload so the rewritten graph remains schema-correct.
            if node.target == torch.ops.aten.clamp.default and (
                isinstance(min_value, torch.fx.Node)
                or isinstance(max_value, torch.fx.Node)
            ):
                node.target = torch.ops.aten.clamp.Tensor
                modified = True

            logger.debug(
                f"Legalized quantized Clamp {node.name} to {qparam.dtype} with "
                "shared scale/zero-point operands."
            )

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        # This pass is idempotent. Reporting the modification lets the default
        # restart strategy re-run earlier canonicalization passes when needed.
        return PassResult(modified)
