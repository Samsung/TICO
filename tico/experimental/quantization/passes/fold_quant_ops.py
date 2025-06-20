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

from tico.serialize.quant_param import QPARAM_KEY, QuantParam, to_qparam_dtype
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import get_quant_dtype
from tico.utils.validate_args_kwargs import (
    DequantizePerTensorArgs,
    QuantizePerTensorArgs,
)


@trace_graph_diff_on_pass
class FoldQuantOps(PassBase):
    """
    This pass folds (Q - DQ) pattern to previous op. After quantization from torch, activation ops
     have (op - Q - DQ) pattern.

    To export quantized circle, this pass removes (Q - DQ) nodes and saves those quantization info
     to previous op's metadata.

    [BEFORE]
      op (float) - Quantize - Dequantize - (float)

    [AFTER]
      op (float with meta[QPARAM_KEY])
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

            # Case 1. op is not quantized
            # - Quantize op
            # Case 2. op is quantized
            # 2.1. op_dtype == qdq_dtype
            # - Just skip (NOTE Need requantization?)
            # 2.2. op_dtype != qdq_dtype
            # - Insert Quantize operator

            # Case 1
            if QPARAM_KEY not in op.meta:
                qparam = QuantParam()
                qparam.scale = [q_args.scale]
                qparam.zero_point = [q_args.zero_p]
                qparam.dtype = get_quant_dtype(q_args.quant_min, q_args.quant_max)
                op.meta[QPARAM_KEY] = qparam

                dq.replace_all_uses_with(op, propagate_meta=False)

                logger.debug(f"{q.name} and {dq.name} are folded to {op.name}.")
            else:
                op_qparam: QuantParam = op.meta[QPARAM_KEY]
                qdq_dtype = get_quant_dtype(q_args.quant_min, q_args.quant_max)
                # Case 2.2
                if op_qparam.dtype != qdq_dtype:
                    # If op is already quantized with a dtype different from qdq, leave quantize
                    qparam = QuantParam()
                    qparam.scale = [q_args.scale]
                    qparam.zero_point = [q_args.zero_p]
                    qparam.dtype = qdq_dtype
                    q.meta[QPARAM_KEY] = qparam
                    assert len(q.users) == 1, "Fix me unless"

                    dq.replace_all_uses_with(q, propagate_meta=False)
                    logger.debug(f"{dq.name} is folded ({q.name} is left).")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)
