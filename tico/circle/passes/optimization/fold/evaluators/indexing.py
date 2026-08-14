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

from __future__ import annotations

import numpy as np

from tico.circle.analysis import TensorContract
from tico.circle.passes.optimization.fold.evaluators.base import (
    ConstantEvaluation,
    ConstantEvaluationContext,
    ConstantEvaluator,
    contract_is_dense_value,
    contract_is_fully_static,
)
from tico.circle.value import TensorValue


class GatherEvaluator(ConstantEvaluator):
    """Fold a constant GATHER with batch_dims zero and in-range integer indices."""

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require both params and indices tensors to be constants."""

        return (0, 1)

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Evaluate one axis gather while preserving safe per-tensor qparams."""

        if len(context.input_indices) != 2 or len(context.output_indices) != 1:
            return None
        params_contract = context.input_contract(0)
        indices_contract = context.input_contract(1)
        output_contract = context.output_contract()
        if not all(
            contract_is_fully_static(contract) and contract_is_dense_value(contract)
            for contract in (
                params_contract,
                indices_contract,
                output_contract,
            )
        ):
            return None
        if params_contract.tensor_type != output_contract.tensor_type:
            return None
        if indices_contract.quantization is not None:
            return None
        if not _gather_quantization_is_safe(params_contract, output_contract):
            return None

        options = context.options
        batch_dims = int(getattr(options, "batchDims", 0) or 0)
        if batch_dims != 0:
            return None
        axis = int(getattr(options, "axis", 0) or 0)
        if axis < 0:
            axis += params_contract.rank
        if axis < 0 or axis >= params_contract.rank:
            return None

        indices_spec = context.codec.registry.get(indices_contract.tensor_type)
        if indices_spec is None or indices_spec.packed:
            return None
        if indices_spec.logical_dtype.kind not in {"i", "u"}:
            return None
        indices = context.input_value(1).data
        axis_size = params_contract.shape[axis]
        if indices.size:
            if np.any(indices < 0) or np.any(indices >= axis_size):
                return None

        expected_shape = (
            params_contract.shape[:axis]
            + indices_contract.shape
            + params_contract.shape[axis + 1 :]
        )
        if expected_shape != output_contract.shape:
            return None

        normalized_indices = indices.astype(np.intp, copy=False)
        result = np.take(
            context.input_value(0).data,
            normalized_indices,
            axis=axis,
        )
        if tuple(result.shape) != output_contract.shape:
            return None
        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=result,
                    quantization=output_contract.quantization,
                ),
            )
        )


def _gather_quantization_is_safe(
    params_contract: TensorContract,
    output_contract: TensorContract,
) -> bool:
    """Allow absent or exactly equal per-tensor affine quantization only."""

    params_quantization = params_contract.quantization
    output_quantization = output_contract.quantization
    if params_quantization != output_quantization:
        return False
    if params_quantization is None:
        return True
    return len(params_quantization.scale) <= 1
