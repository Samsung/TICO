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

import copy
import operator

import torch
from torch.export import ExportedProgram

from tico.quantization.passes.mx_qparam import remap_mx_qparam
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.dtypes import (
    assert_supported_mx_export_options,
    is_mx_dtype,
    mx_dtype_from_elem_format,
    normalize_mx_axis,
    normalize_mx_elem_format,
)
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import get_quant_dtype, quant_min_max
from tico.utils.validate_args_kwargs import (
    DequantizePerTensorArgs,
    QuantizePerTensorArgs,
)

_QPARAM_PRESERVING_VIEW_TARGETS = frozenset(
    {
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.expand.default,
    }
)

# Weight-DQ removal must also preserve the inputs of a producer whose original
# floating-point result is still observed outside its explicit Quantize branch.
_PRESERVE_FLOAT_INPUTS_KEY = "_tico_preserve_float_quantization_inputs"


def _restore_float_inputs(graph: torch.fx.Graph) -> None:
    """Bridge folded affine inputs back to protected producers' float domains."""
    for node in list(graph.nodes):
        if not node.meta.get(_PRESERVE_FLOAT_INPUTS_KEY):
            continue
        for source in list(node.all_input_nodes):
            qparam = source.meta.get(QPARAM_KEY)
            if qparam is None:
                continue
            if (
                qparam.quantized_dimension is not None
                or qparam.scale is None
                or len(qparam.scale) != 1
                or qparam.zero_point is None
                or len(qparam.zero_point) != 1
                or is_mx_dtype(qparam.dtype)
            ):
                raise NotYetSupportedError(
                    "A shared floating-point producer requires per-tensor affine "
                    f"Dequantize inputs; {source.name} has unsupported qparams."
                )
            quant_min, quant_max = quant_min_max(qparam.dtype)
            storage_dtype = (
                torch.uint8
                if qparam.dtype == "uint4"
                else (
                    torch.int8
                    if qparam.dtype == "int4"
                    else getattr(torch, qparam.dtype)
                )
            )
            with graph.inserting_before(node):
                dequantize = graph.call_function(
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    args=(
                        source,
                        qparam.scale[0],
                        qparam.zero_point[0],
                        quant_min,
                        quant_max,
                        storage_dtype,
                    ),
                )
            dequantize.meta = copy.copy(source.meta)
            dequantize.meta.pop(QPARAM_KEY)
            dequantize.meta["val"] = source.meta["val"].to(dtype=torch.float32)
            node.replace_input_with(source, dequantize)


def _inherit_qparam_from_view_chain(node: torch.fx.Node) -> bool:
    """Seed a view chain with its source's affine or remapped MX domain.

    ``FoldQuantOps`` runs before ``PropagateQParamForward``.  A
    requantization after a slice/view chain must therefore discover
    the already-folded producer qparams by walking through the view
    chain, including concatenations whose inputs share one domain. Otherwise
    the target Q-DQ pair can be folded into RESHAPE and the required Quantize
    operation disappears.

    MX views must preserve the shared-exponent blocks. Affine per-channel
    propagation is not handled here.
    """
    if QPARAM_KEY in node.meta:
        return False

    chain: list[torch.fx.Node] = []
    current = node
    while QPARAM_KEY not in current.meta:
        if current.op != "call_function":
            return False
        if current.target == torch.ops.aten.cat.default:
            inputs = current.args[0]
            if not isinstance(inputs, (list, tuple)) or not inputs:
                return False
            sources: list[torch.fx.Node] = []
            for source in inputs:
                if not isinstance(source, torch.fx.Node):
                    return False
                sources.append(source)
            for source in sources:
                _inherit_qparam_from_view_chain(source)
            qparams = [source.meta.get(QPARAM_KEY) for source in sources]
            first_qparam = qparams[0]
            if first_qparam is None or any(qparam is None for qparam in qparams):
                return False
            if is_mx_dtype(first_qparam.dtype):
                current.meta[QPARAM_KEY] = remap_mx_qparam(first_qparam, current)
            else:
                if first_qparam.quantized_dimension is not None or any(
                    qparam != first_qparam for qparam in qparams[1:]
                ):
                    return False
                current.meta[QPARAM_KEY] = copy.deepcopy(first_qparam)
            break
        if current.target == operator.getitem:
            split = current.args[0]
            if (
                not isinstance(split, torch.fx.Node)
                or split.target != torch.ops.aten.split_with_sizes.default
            ):
                return False
            source = split.args[0]
        elif current.target in _QPARAM_PRESERVING_VIEW_TARGETS or current.target == (
            torch.ops.aten.constant_pad_nd.default
        ):
            source = current.args[0]
        else:
            return False
        if not isinstance(source, torch.fx.Node):
            return False
        chain.append(current)
        current = source

    source_qparam: QuantParam = current.meta[QPARAM_KEY]
    if source_qparam.quantized_dimension is not None and not is_mx_dtype(
        source_qparam.dtype
    ):
        return False
    if not is_mx_dtype(source_qparam.dtype) and any(
        view.target in {operator.getitem, torch.ops.aten.constant_pad_nd.default}
        for view in chain
    ):
        return False

    for view in reversed(chain):
        if is_mx_dtype(source_qparam.dtype):
            mapped = remap_mx_qparam(source_qparam, view)
            assert mapped is not None
            source_qparam = mapped
        view.meta[QPARAM_KEY] = copy.deepcopy(source_qparam)
    return bool(chain)


def _mx_op_params(node) -> tuple[str, int, str, str]:
    """Return normalized MX quantization parameters from an FX custom op node."""
    assert len(node.args) >= 3
    elem_format = normalize_mx_elem_format(node.args[1])
    axis = node.args[2]
    source = node.args[0]
    assert isinstance(source, torch.fx.Node)
    # DecomposeFakeQuantize may not have refreshed the new Q node's value
    # metadata yet. Q/DQ preserve shape, so use the quantizer's real input.
    if node.target == torch.ops.circle_custom.dequantize_mx.default:
        source = source.args[0]
        assert isinstance(source, torch.fx.Node)
    axis = normalize_mx_axis(axis, source.meta["val"].ndim)
    shared_exp_method = node.kwargs.get(
        "shared_exp_method", node.args[3] if len(node.args) > 3 else "max"
    )
    round_mode = node.kwargs.get(
        "round", node.args[4] if len(node.args) > 4 else "nearest"
    )
    assert isinstance(shared_exp_method, str)
    assert isinstance(round_mode, str)
    return elem_format, axis, shared_exp_method, round_mode


def _mx_qparam_from_quant_node(q) -> QuantParam:
    """Build a QuantParam from a logical MX quantize node."""
    elem_format, axis, shared_exp_method, round_mode = _mx_op_params(q)
    assert_supported_mx_export_options(
        elem_format=elem_format,
        shared_exp_method=shared_exp_method,
        round=round_mode,
    )
    qparam = QuantParam()
    qparam.dtype = mx_dtype_from_elem_format(elem_format)
    qparam.quantized_dimension = axis
    return qparam


def _same_mx_qparam(lhs: QuantParam, rhs: QuantParam) -> bool:
    """Return True when two QuantParams describe the same MX quantization."""
    return (
        lhs.dtype == rhs.dtype
        and is_mx_dtype(lhs.dtype)
        and lhs.quantized_dimension == rhs.quantized_dimension
    )


def _fold_mx_qdq(dq: torch.fx.Node) -> None:
    """Fold an MX pair only after resolving its producer's existing domain."""
    q = dq.args[0]
    if not isinstance(q, torch.fx.Node):
        return
    if q.target != torch.ops.circle_custom.quantize_mx.default:
        return
    op = q.args[0]
    if not isinstance(op, torch.fx.Node):
        return
    if _mx_op_params(q) != _mx_op_params(dq):
        return

    _inherit_qparam_from_view_chain(op)
    qparam = _mx_qparam_from_quant_node(q)
    existing = op.meta.get(QPARAM_KEY)
    if existing is None:
        # An explicit quantizer on a shared float value must not change its
        # other users. Layout ops also cannot quantize their own inputs.
        layout_output = op.target in _QPARAM_PRESERVING_VIEW_TARGETS or op.target in {
            torch.ops.aten.cat.default,
            torch.ops.aten.constant_pad_nd.default,
            operator.getitem,
        }
        if (
            layout_output
            or op.target == torch.ops.aten.embedding.default
            or any(user is not q for user in op.users)
        ):
            if not layout_output:
                op.meta[_PRESERVE_FLOAT_INPUTS_KEY] = True
            q.meta[QPARAM_KEY] = qparam
            dq.replace_all_uses_with(q, propagate_meta=False)
        else:
            op.meta[QPARAM_KEY] = qparam
            dq.replace_all_uses_with(op, propagate_meta=False)
    elif _same_mx_qparam(existing, qparam):
        dq.replace_all_uses_with(op, propagate_meta=False)
    else:
        # A different axis or element format requires an explicit QUANTIZE,
        # even when both sides use the same MX dtype.
        q.meta[QPARAM_KEY] = qparam
        dq.replace_all_uses_with(q, propagate_meta=False)


@trace_graph_diff_on_pass
class FoldQuantOps(PassBase):
    """
    This pass folds (Q - DQ) pattern to previous op. After quantization from torch, activation ops
     have (op - Q - DQ) pattern.

    To export quantized circle, this pass removes (Q - DQ) nodes and saves those quantization info
     to previous op's metadata.

    ────────────────────────────────────────────────────────────────
    BEFORE                             AFTER
    ────────────────────────────────────────────────────────────────
      op(float) ─ Q ─ DQ ─ …            op(float, meta[QPARAM])

      op ─ Q1 ─ DQ1 ─ Q2 ─ DQ2          op(meta[QPARAM]) ─ Q2
                 ▲                                          ▲
                 │ (Q1, DQ1 folded)                         │ (re-quantization kept)

      op ─ Q ─┬─ DQ0                    op(meta[QPARAM])
              ├─ DQ1                    (each DQ* folded, Q dropped when orphaned)
              └─ DQ2
    ────────────────────────────────────────────────────────────────

    Algorithm
    ---------
    1. Iterate over affine and MX Dequantize nodes in graph order, so mixed
       quantization boundaries see the preceding domain.
    2. For each DQ, verify it is driven by a Quantize node `q` and that
       `q` and `dq` share identical (scale, zero-point, dtype).
    3. a) If the producer op has **no** QPARAM, attach one, then replace
          *this* DQ's usages with the producer op.
       b) If the producer is already quantized with different affine qparams,
          including a different scale or zero point with the same dtype, this
          is a *re-quantization*: attach QPARAM to `q` and keep it, but still
          remove the DQ.
    4. After all replacements, run `graph.eliminate_dead_code()`.
       Any Quantize that became orphaned because *all* its DQs were folded
       is deleted automatically.

    Embedding lowers to value-preserving Circle GATHER, which cannot apply an
    independently calibrated output quantization grid. For an unquantized
    embedding output, keep Q explicitly and remove only DQ. Its table remains
    dequantized so the lookup preserves FLOAT32 values before that Quantize.

    Also keep Q when the unquantized producer has other users: folding it into
    that producer would round those users' values too. Preserve that producer's
    float inputs with explicit Dequantize bridges after the remaining folds.

    Discover an existing source domain through slices, views, and same-domain
    concatenations before deciding whether a trailing Q is a requantization.
    A float producer behind a layout view can still adopt its first affine Q
    domain. MX keeps an explicit Q after an unquantized layout operation.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        for dq in graph.nodes:
            if dq.op != "call_function":
                continue
            if dq.target == torch.ops.circle_custom.dequantize_mx.default:
                _fold_mx_qdq(dq)
                continue
            if (
                dq.target
                != torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                continue
            dq_args = DequantizePerTensorArgs(*dq.args, **dq.kwargs)

            q = dq_args.input
            if q.target != torch.ops.quantized_decomposed.quantize_per_tensor.default:
                continue
            q_args = QuantizePerTensorArgs(*q.args, **q.kwargs)  # type: ignore[arg-type]
            op = q_args.tensor

            # Check if Q and DQ have same quant param
            if q_args.scale != dq_args.scale:
                continue
            if q_args.zero_p != dq_args.zero_point:
                continue
            if q_args.dtype != dq_args.dtype:
                continue

            _inherit_qparam_from_view_chain(op)

            # ───────────────────────────────────────────
            # Case 1: op not yet quantized
            # ───────────────────────────────────────────
            if QPARAM_KEY not in op.meta:
                qparam = QuantParam()
                qparam.scale = [q_args.scale]
                qparam.zero_point = [q_args.zero_p]
                qparam.dtype = get_quant_dtype(q_args.quant_min, q_args.quant_max)
                if op.target == torch.ops.aten.embedding.default or any(
                    user is not q for user in op.users
                ):
                    if op.target not in _QPARAM_PRESERVING_VIEW_TARGETS:
                        op.meta[_PRESERVE_FLOAT_INPUTS_KEY] = True
                    q.meta[QPARAM_KEY] = qparam
                    dq.replace_all_uses_with(q, propagate_meta=False)
                    logger.debug(
                        f"{dq.name} is folded; {q.name} preserves its branch output grid."
                    )
                    continue
                op.meta[QPARAM_KEY] = qparam

                dq.replace_all_uses_with(op, propagate_meta=False)

                logger.debug(f"{q.name} and {dq.name} are folded to {op.name}.")
            # ───────────────────────────────────────────
            # Case 2: op already quantized
            #        2.1 same dtype  → nothing to do
            #        2.2 diff dtype  → leave Q in place
            # ───────────────────────────────────────────
            else:
                op_qparam: QuantParam = op.meta[QPARAM_KEY]
                qdq_dtype = get_quant_dtype(q_args.quant_min, q_args.quant_max)
                same_qparams = (
                    op_qparam.dtype == qdq_dtype
                    and op_qparam.scale == [q_args.scale]
                    and op_qparam.zero_point == [q_args.zero_p]
                    and op_qparam.quantized_dimension is None
                )

                if not same_qparams:
                    if QPARAM_KEY not in q.meta:
                        qparam = QuantParam()
                        qparam.scale = [q_args.scale]
                        qparam.zero_point = [q_args.zero_p]
                        qparam.dtype = qdq_dtype
                        q.meta[QPARAM_KEY] = qparam
                        assert len(q.users) == 1, "Fix me unless"

                    dq.replace_all_uses_with(q, propagate_meta=False)
                    logger.debug(
                        f"{dq.name} is folded as requantization " f"({q.name} is left)."
                    )
                else:
                    dq.replace_all_uses_with(op, propagate_meta=False)
                    logger.debug(f"Removed redundant {dq.name}")

        _restore_float_inputs(graph)
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
