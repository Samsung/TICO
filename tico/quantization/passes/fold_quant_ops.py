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

import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils import logging
from tico.utils.mx.dtypes import (
    assert_supported_mx_export_options,
    is_mx_dtype,
    mx_dtype_from_elem_format,
    normalize_mx_elem_format,
)
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import get_quant_dtype
from tico.utils.validate_args_kwargs import (
    DequantizePerTensorArgs,
    QuantizePerTensorArgs,
)


def _mx_op_params(node) -> tuple[str, int, str, str]:
    """Return normalized MX quantization parameters from an FX custom op node."""
    assert len(node.args) >= 3
    elem_format = normalize_mx_elem_format(node.args[1])
    axis = node.args[2]
    assert isinstance(axis, int)
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
    1. Iterate over *all* Dequantize nodes.
    2. For each DQ, verify it is driven by a Quantize node `q` and that
       `q` and `dq` share identical (scale, zero-point, dtype).
    3. a) If the producer op has **no** QPARAM, attach one, then replace
          *this* DQ's usages with the producer op.
       b) If the producer is already quantized with a different dtype,
          this is a *re-quantization*: attach QPARAM to `q` and keep it,
          but still remove the DQ.
    4. After all replacements, run `graph.eliminate_dead_code()`.
       Any Quantize that became orphaned because *all* its DQs were folded
       is deleted automatically.
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

            # ───────────────────────────────────────────
            # Case 1: op not yet quantized
            # ───────────────────────────────────────────
            if QPARAM_KEY not in op.meta:
                qparam = QuantParam()
                qparam.scale = [q_args.scale]
                qparam.zero_point = [q_args.zero_p]
                qparam.dtype = get_quant_dtype(q_args.quant_min, q_args.quant_max)
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

                if op_qparam.dtype != qdq_dtype:
                    # Attach QPARAM to Q once
                    if QPARAM_KEY not in q.meta:
                        qparam = QuantParam()
                        qparam.scale = [q_args.scale]
                        qparam.zero_point = [q_args.zero_p]
                        qparam.dtype = qdq_dtype
                        q.meta[QPARAM_KEY] = qparam
                        assert len(q.users) == 1, "Fix me unless"

                    dq.replace_all_uses_with(q, propagate_meta=False)
                    logger.debug(f"{dq.name} is folded ({q.name} is left).")
                else:
                    # Same dtype → the Quantize–Dequantize pair is redundant.
                    assert op_qparam.scale and op_qparam.scale[0] == q_args.scale
                    assert (
                        op_qparam.zero_point
                        and op_qparam.zero_point[0] == q_args.zero_p
                    )
                    dq.replace_all_uses_with(op, propagate_meta=False)
                    logger.debug(f"Removed redundant {dq.name}")

        for dq in graph.nodes:
            if dq.op != "call_function":
                continue
            if dq.target != torch.ops.circle_custom.dequantize_mx.default:
                continue

            q = dq.args[0]
            if not isinstance(q, torch.fx.Node):
                continue
            if q.target != torch.ops.circle_custom.quantize_mx.default:
                continue

            op = q.args[0]
            if not isinstance(op, torch.fx.Node):
                continue

            if _mx_op_params(q) != _mx_op_params(dq):
                continue

            qparam = _mx_qparam_from_quant_node(q)

            if QPARAM_KEY not in op.meta:
                op.meta[QPARAM_KEY] = qparam
                dq.replace_all_uses_with(op, propagate_meta=False)
                logger.debug(f"{q.name} and {dq.name} are folded to {op.name}.")
            else:
                op_qparam = op.meta[QPARAM_KEY]
                if not _same_mx_qparam(op_qparam, qparam):
                    if QPARAM_KEY not in q.meta:
                        q.meta[QPARAM_KEY] = qparam
                        assert len(q.users) == 1, "Fix me unless"
                    dq.replace_all_uses_with(q, propagate_meta=False)
                    logger.debug(f"{dq.name} is folded ({q.name} is left).")
                else:
                    dq.replace_all_uses_with(op, propagate_meta=False)
                    logger.debug(f"Removed redundant {dq.name}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
