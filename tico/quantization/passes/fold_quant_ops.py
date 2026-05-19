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

import torch
from torch.export import ExportedProgram

from tico.quantization.passes.insert_quantize_on_dtype_mismatch import qparam_dtype

from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import get_quant_dtype, quant_min_max, set_new_meta_val
from tico.utils.validate_args_kwargs import (
    DequantizePerTensorArgs,
    QuantizePerTensorArgs,
)


def _insert_mx_quantize_op(node, qparam):
    graph = node.graph
    assert qparam.quantized_dimension is not None
    assert qparam.dtype is not None

    with graph.inserting_after(node):
        q_args = (node, qparam.dtype, qparam.quantized_dimension)
        quantize = create_node(
            graph,
            torch.ops.circle_custom.quantize_mx_decomposed.default,
            args=q_args,
        )

    node.replace_all_uses_with(quantize, propagate_meta=True)
    quantize.replace_input_with(quantize, node)

    quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)

    return quantize


def _insert_quantize_op(node, qparam):
    graph = node.graph
    min_, max_ = quant_min_max(qparam.dtype)
    dtype = getattr(torch, qparam.dtype)

    with graph.inserting_after(node):
        q_args = (node, qparam.scale[0], qparam.zero_point[0], min_, max_, dtype)
        quantize = create_node(
            graph,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            args=q_args,
        )

    node.replace_all_uses_with(quantize, propagate_meta=True)
    quantize.replace_input_with(quantize, node)

    quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)

    return quantize


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
                assert (
                    QPARAM_KEY not in dq.meta
                )  # we should not abandon quantization calibrated parameters
            # if QPARAM_KEY in dq.meta: #right now it's not needed
            #    if (qparam_dtype(op) == "int16" or qparam_dtype(op) == "uint8") and qparam_dtype(dq) == "mxint8":
            #        #need to insert requantization
            #        assert(False)
            #        _insert_mx_quantize_op(op, dq.meta[QPARAM_KEY])

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
            if dq.target != torch.ops.circle_custom.dequantize_mx_decomposed.default:
                continue

            dq_args = dq.args

            q = dq_args[0]  # type: ignore[index]
            if q.target != torch.ops.circle_custom.quantize_mx_decomposed.default:
                continue
            q_args = q.args
            op = q_args[0]  # type: ignore[index]

            # Check if Q and DQ have same parameters
            if q_args[1] != dq_args[1]:  # type: ignore[index]
                continue
            if q_args[2] != dq_args[2]:  # type: ignore[index]
                continue

            # ───────────────────────────────────────────
            # Case 1: op not yet quantized
            # ───────────────────────────────────────────
            if QPARAM_KEY not in op.meta:
                # TODO
                qparam = QuantParam()
                qparam.dtype = "mxint8"  # q_args[1] #TODO
                qparam.quantized_dimension = q_args[2]  # type: ignore[index]
                op.meta[QPARAM_KEY] = qparam

                dq.replace_all_uses_with(op, propagate_meta=False)

                logger.debug(f"{q.name} and {dq.name} are folded to {op.name}.")
                if QPARAM_KEY in dq.meta:
                    if qparam_dtype(op) == "mxint8" and (
                        qparam_dtype(dq) == "int16" or qparam_dtype(dq) == "uint8"
                    ):
                        # need to insert requantization
                        _insert_quantize_op(op, dq.meta[QPARAM_KEY])

            # ───────────────────────────────────────────
            # Case 2: op already quantized
            #        2.1 same dtype  → nothing to do
            #        2.2 diff dtype  → leave Q in place
            # ───────────────────────────────────────────
            else:
                op_qparam: QuantParam = op.meta[QPARAM_KEY]  # type: ignore[no-redef]
                qdq_dtype = "mxint8"  # q_args[1] #TODO

                if op_qparam.dtype != qdq_dtype:
                    # Attach QPARAM to Q once
                    if QPARAM_KEY not in q.meta:
                        qparam = QuantParam()
                        qparam.dtype = qdq_dtype
                        qparam.quantized_dimension = q_args[2]  # type: ignore[index]
                        q.meta[QPARAM_KEY] = qparam
                        assert len(q.users) == 1, "Fix me unless"

                    dq.replace_all_uses_with(q, propagate_meta=False)
                    logger.debug(f"{dq.name} is folded ({q.name} is left).")
                else:
                    # Same dtype → the Quantize–Dequantize pair is redundant.
                    assert not op_qparam.scale
                    assert not op_qparam.zero_point
                    assert op_qparam.dtype and op_qparam.dtype == "mxint8"  # TODO
                    assert (
                        op_qparam.quantized_dimension is not None
                        and op_qparam.quantized_dimension == q_args[2]  # type: ignore[index]
                    )
                    dq.replace_all_uses_with(op, propagate_meta=False)
                    logger.debug(f"Removed redundant {dq.name}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
