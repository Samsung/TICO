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

import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass


def _qparam_dtype(node: torch.fx.Node) -> str:
    """Return the quantization dtype of a node."""
    assert QPARAM_KEY in node.meta
    return node.meta[QPARAM_KEY].dtype


@trace_graph_diff_on_pass
class RemoveRedundantQuantisers(PassBase):
    """Remove redundant pairs of consecutive quantizers that form a round-trip.

    After ``InsertQuantizeOnDtypeMismatch`` runs, the graph may contain
    consecutive quantize ops that convert to an intermediate dtype and
    immediately back, e.g.:

    * **Pattern 1 – int16 → mxint8 → int16**

      ``node(int16) → quantize_mx(mxint8) → quantize_per_tensor(int16)``

    * **Pattern 2 – mxint8 → int16 → mxint8**

      ``node(mxint8) → quantize_per_tensor(int16) → quantize_mx(mxint8)``

    In both cases the output dtype equals the input dtype, so the second
    quantiser (and the first, when it has no other users) is redundant.

    ────────────────────────────────────────────────────────────────
    BEFORE                                    AFTER
    ────────────────────────────────────────────────────────────────
    A(int16) ─ Q_mx(mxint8) ─ Q_pt(int16)   A(int16)
    A(mxint8) ─ Q_pt(int16) ─ Q_mx(mxint8)  A(mxint8)
    ────────────────────────────────────────────────────────────────
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        modified = False

        # ── Pattern 1: int16 → quantize_mx(mxint8) → quantize_per_tensor(int16) ──
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != torch.ops.quantized_decomposed.quantize_per_tensor.default:
                continue
            if QPARAM_KEY not in node.meta:
                continue
            if _qparam_dtype(node) != "int16":
                continue

            q_pt_input = node.args[0]  # type: ignore[index]
            if not isinstance(q_pt_input, torch.fx.Node):
                continue
            if q_pt_input.target != torch.ops.circle_custom.quantize_mx_decomposed.default:
                continue
            if QPARAM_KEY not in q_pt_input.meta:
                continue
            if _qparam_dtype(q_pt_input) != "mxint8":
                continue

            q_mx_input = q_pt_input.args[0]  # type: ignore[index]
            if not isinstance(q_mx_input, torch.fx.Node):
                continue
            if QPARAM_KEY not in q_mx_input.meta:
                continue
            if _qparam_dtype(q_mx_input) != "int16":
                continue

            # Redundant round-trip: int16 → mxint8 → int16
            node.replace_all_uses_with(q_mx_input, propagate_meta=False)
            modified = True
            logger.debug(
                f"Removed redundant quantisers: {q_mx_input.name}(int16) → "
                f"{q_pt_input.name}(mxint8) → {node.name}(int16)"
            )

        # ── Pattern 2: mxint8 → quantize_per_tensor(int16) → quantize_mx(mxint8) ──
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != torch.ops.circle_custom.quantize_mx_decomposed.default:
                continue
            if QPARAM_KEY not in node.meta:
                continue
            if _qparam_dtype(node) != "mxint8":
                continue

            q_mx_input = node.args[0]  # type: ignore[index]
            if not isinstance(q_mx_input, torch.fx.Node):
                continue
            if q_mx_input.target != torch.ops.quantized_decomposed.quantize_per_tensor.default:
                continue
            if QPARAM_KEY not in q_mx_input.meta:
                continue
            if _qparam_dtype(q_mx_input) != "int16":
                continue

            q_pt_input = q_mx_input.args[0]  # type: ignore[index]
            if not isinstance(q_pt_input, torch.fx.Node):
                continue
            if QPARAM_KEY not in q_pt_input.meta:
                continue
            if _qparam_dtype(q_pt_input) != "mxint8":
                continue

            # Redundant round-trip: mxint8 → int16 → mxint8
            node.replace_all_uses_with(q_pt_input, propagate_meta=False)
            modified = True
            logger.debug(
                f"Removed redundant quantisers: {q_pt_input.name}(mxint8) → "
                f"{q_mx_input.name}(int16) → {node.name}(mxint8)"
            )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
