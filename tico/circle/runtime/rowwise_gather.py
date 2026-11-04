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

"""Prepare selected-row execution of explicit constant DEQUANTIZE -> GATHER.

The serialized graph keeps ordinary float GATHER semantics. Only its execution
plan changes: the full float DEQUANTIZE result is virtual, and each GATHER
materializes its selected rows. Q/DQ and MUL after GATHER remain separate kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tico.circle.runtime.errors import UnsupportedCircleOperatorError
from tico.circle.runtime.kernels.base import KernelContext
from tico.circle.runtime.kernels.shape_ops import check_indices
from tico.circle.runtime.program import CircleProgram, RuntimeOperator


@dataclass(frozen=True)
class RowwiseGatherPlan:
    """One GATHER's immutable execution plan, containing O(V) metadata only."""

    weight_index: int
    dequantize_index: int
    scale: np.ndarray
    zero_point: np.ndarray

    def execute(self, ctx: KernelContext) -> tuple[np.ndarray, ...]:
        """Gather integer rows and their qparams, then restore only those rows."""

        weight = ctx.input(0)
        indices = check_indices(ctx, ctx.input(1), weight.shape[0], what="indices")
        selected = np.take(weight, indices, axis=0).astype(np.float32)
        # In-place arithmetic bounds FP32 working storage to selected.shape.
        selected -= np.take(self.zero_point, indices)[..., None]
        selected *= np.take(self.scale, indices)[..., None]
        return (selected,)


def plan_rowwise_gathers(
    program: CircleProgram,
) -> tuple[dict[int, RowwiseGatherPlan], set[int]]:
    """Plan row-wise constant UINT8 lookups once, rejecting unsafe target cases.

    Other DQ graphs retain ordinary execution. A constant rank-2 quantized table
    feeding axis-0 GATHER is a target: an unsupported target receives a diagnostic
    rather than silently allocating the full FP32 table. Multiple GATHER consumers
    of one DQ are supported; other DQ consumers and a visible full-table output
    cannot be served by selected rows and are rejected.
    """

    consumers: dict[int, list[RuntimeOperator]] = {}
    for operator in program.operators:
        for index in operator.inputs:
            consumers.setdefault(index, []).append(operator)
    plans: dict[int, RowwiseGatherPlan] = {}
    virtual: set[int] = set()
    for dq in program.operators:
        if dq.name != "DEQUANTIZE" or len(dq.inputs) != 1 or len(dq.outputs) != 1:
            continue
        weight = program.tensor(dq.inputs[0])
        uses = consumers.get(dq.outputs[0], [])
        qparams = weight.quantization
        if (
            not weight.is_constant
            or len(weight.shape) != 2
            or not weight.is_quantized_integer
            or qparams is None
            or weight.dtype != np.dtype(np.uint8)
            or (len(qparams.scale) == 1 and weight.shape[0] != 1)
            or not any(
                use.name == "GATHER"
                and len(use.inputs) == 2
                and use.inputs[0] == dq.outputs[0]
                and int(getattr(use.options, "axis", 0)) in (0, -2)
                for use in uses
            )
        ):
            continue

        def unsupported(reason: str) -> None:
            raise UnsupportedCircleOperatorError(
                "Selected-row DEQUANTIZE -> GATHER cannot execute this table: "
                f"{reason}. No full-table fallback is used.\n"
                f"  at {program.describe_operator(dq)}"
            )

        assert qparams is not None
        if weight.dtype != np.dtype(np.uint8) or weight.is_variable:
            unsupported("requires constant UINT8 weight")
        if (
            qparams.quantized_dimension != 0
            or len(qparams.scale) != weight.shape[0]
            or len(qparams.zero_point) != weight.shape[0]
        ):
            unsupported("requires one scale/zero_point per vocabulary row on axis 0")
        scales = np.asarray(qparams.scale, dtype=np.float32)
        zero_points = np.asarray(qparams.zero_point, dtype=np.float32)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            unsupported("row scales must be finite and positive")
        if np.any(zero_points < 0) or np.any(zero_points > 255):
            unsupported("UINT8 row zero_points must lie in [0, 255]")
        float_table = program.tensor(dq.outputs[0])
        if (
            float_table.dtype != np.dtype(np.float32)
            or float_table.is_quantized
            or float_table.shape != weight.shape
            or float_table.is_dynamic
        ):
            unsupported("requires an unquantized FLOAT32 DQ result matching the table")
        if dq.outputs[0] in program.outputs:
            unsupported("full DQ result is a graph output")
        for gather in uses:
            if (
                gather.name != "GATHER"
                or len(gather.inputs) != 2
                or len(gather.outputs) != 1
                or gather.inputs[0] != dq.outputs[0]
                or gather.index <= dq.index
                or int(getattr(gather.options, "axis", 0)) not in (0, -2)
                or int(getattr(gather.options, "batchDims", 0)) != 0
            ):
                unsupported(
                    "DQ result has a consumer other than axis-0 batchDims=0 GATHER"
                )
            indices = program.tensor(gather.inputs[1])
            output = program.tensor(gather.outputs[0])
            if (
                indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
                or indices.is_quantized
            ):
                unsupported("indices must be unquantized INT32 or INT64 (any rank)")
            if output.dtype != np.dtype(np.float32) or output.is_quantized:
                unsupported("GATHER must select FLOAT32 values without a domain change")
            plans[gather.index] = RowwiseGatherPlan(
                weight.index, dq.index, scales, zero_points
            )
        scales.setflags(write=False)
        zero_points.setflags(write=False)
        virtual.add(dq.index)
    return plans, virtual
