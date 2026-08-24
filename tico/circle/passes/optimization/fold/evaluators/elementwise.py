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

from collections.abc import Callable
from typing import Any

import numpy as np

from tico.circle.passes.optimization.fold.evaluators.base import (
    ConstantEvaluation,
    ConstantEvaluationContext,
    ConstantEvaluator,
    contract_is_dense_value,
    contract_is_fully_static,
)
from tico.circle.value import TensorValue


class BinaryElementwiseEvaluator(ConstantEvaluator):
    """Fold an unfused binary ADD or MUL with exact dense constant inputs."""

    def __init__(
        self,
        operation_name: str,
        operation: Callable[[object, object], object],
    ) -> None:
        """Bind one NumPy-compatible binary operation to the evaluator."""

        if not operation_name:
            raise ValueError("operation_name must not be empty.")
        if not callable(operation):
            raise TypeError("operation must be callable.")
        self.operation_name = operation_name
        self.operation = operation

    @property
    def name(self) -> str:
        """Return a diagnostic name that identifies the bound binary operation."""

        return f"{self.__class__.__name__}({self.operation_name})"

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require both binary operands to be constants."""

        return (0, 1)

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Evaluate a safe non-quantized binary operation with broadcasting."""

        if len(context.input_indices) != 2 or len(context.output_indices) != 1:
            return None
        lhs_contract = context.input_contract(0)
        rhs_contract = context.input_contract(1)
        output_contract = context.output_contract()
        contracts = (lhs_contract, rhs_contract, output_contract)
        if not all(
            contract_is_fully_static(contract) and contract_is_dense_value(contract)
            for contract in contracts
        ):
            return None
        if any(contract.quantization is not None for contract in contracts):
            return None
        if not (
            lhs_contract.tensor_type
            == rhs_contract.tensor_type
            == output_contract.tensor_type
        ):
            return None
        if not _has_no_fused_activation(context.options):
            return None
        if bool(getattr(context.options, "potScaleInt16", False)):
            return None

        try:
            broadcast_shape = tuple(
                int(dimension)
                for dimension in np.broadcast_shapes(
                    lhs_contract.shape,
                    rhs_contract.shape,
                )
            )
        except ValueError:
            return None
        if broadcast_shape != output_contract.shape:
            return None

        spec = context.codec.registry.get(output_contract.tensor_type)
        if spec is None or spec.packed or spec.name == "BFLOAT16":
            return None
        if spec.logical_dtype.kind not in {"f", "i", "u"}:
            return None

        lhs = context.input_value(0).data
        rhs = context.input_value(1).data
        result = _safe_binary_result(
            lhs,
            rhs,
            dtype=spec.logical_dtype,
            operation=self.operation,
        )
        if result is None or tuple(result.shape) != output_contract.shape:
            return None
        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=result,
                    quantization=None,
                ),
            )
        )


class CastEvaluator(ConstantEvaluator):
    """Fold a representable dense CAST without relying on overflow wrapping."""

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require the CAST source tensor to be constant."""

        return (0,)

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Cast one constant when all values have defined target representations."""

        if len(context.input_indices) != 1 or len(context.output_indices) != 1:
            return None
        input_contract = context.input_contract(0)
        output_contract = context.output_contract()
        if not all(
            contract_is_fully_static(contract) and contract_is_dense_value(contract)
            for contract in (input_contract, output_contract)
        ):
            return None
        if input_contract.quantization is not None:
            return None
        if output_contract.quantization is not None:
            return None
        if input_contract.shape != output_contract.shape:
            return None
        if not _cast_options_match(
            context,
            input_contract.tensor_type,
            output_contract.tensor_type,
        ):
            return None

        source_spec = context.codec.registry.get(input_contract.tensor_type)
        target_spec = context.codec.registry.get(output_contract.tensor_type)
        if source_spec is None or target_spec is None:
            return None
        if source_spec.packed or target_spec.packed:
            return None
        if source_spec.name == "BFLOAT16" or target_spec.name == "BFLOAT16":
            return None
        if source_spec.logical_dtype.kind not in {"b", "f", "i", "u"}:
            return None
        if target_spec.logical_dtype.kind not in {"b", "f", "i", "u"}:
            return None

        result = _safe_cast(
            context.input_value(0).data,
            target_spec.logical_dtype,
        )
        if result is None:
            return None
        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=result,
                    quantization=None,
                ),
            )
        )


def _has_no_fused_activation(options: object | None) -> bool:
    """Return whether an optional fused activation field encodes NONE."""

    return (
        options is None or int(getattr(options, "fusedActivationFunction", 0) or 0) == 0
    )


def _cast_options_match(
    context: ConstantEvaluationContext,
    input_tensor_type: int,
    output_tensor_type: int,
) -> bool:
    """Check optional CAST type fields against serialized tensor contracts."""

    options = context.options
    if options is None:
        return True
    if hasattr(options, "inDataType"):
        raw_input_type = getattr(options, "inDataType")
        if raw_input_type is None or int(raw_input_type) != input_tensor_type:
            return False
    if hasattr(options, "outDataType"):
        raw_output_type = getattr(options, "outDataType")
        if raw_output_type is None or int(raw_output_type) != output_tensor_type:
            return False
    return True


def _safe_binary_result(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    dtype: np.dtype[Any],
    operation: Callable[[object, object], object],
) -> np.ndarray | None:
    """Evaluate one binary operation and reject fixed-width integer overflow."""

    normalized_dtype = np.dtype(dtype)
    if normalized_dtype.kind == "f":
        with np.errstate(all="ignore"):
            result = operation(lhs, rhs)
        return np.asarray(result, dtype=normalized_dtype)

    if normalized_dtype.kind not in {"i", "u"}:
        return None
    with np.errstate(all="ignore"):
        result_object = operation(
            np.asarray(lhs, dtype=object),
            np.asarray(rhs, dtype=object),
        )
    result_array = np.asarray(result_object, dtype=object)
    limits = np.iinfo(normalized_dtype)
    if result_array.size and (
        np.any(result_array < limits.min) or np.any(result_array > limits.max)
    ):
        return None
    return result_array.astype(normalized_dtype, copy=False)


def _safe_cast(
    value: np.ndarray,
    target_dtype: np.dtype[Any],
) -> np.ndarray | None:
    """Cast values only when conversion avoids undefined or wrapping behavior."""

    source = np.asarray(value)
    target = np.dtype(target_dtype)
    source_kind = source.dtype.kind
    target_kind = target.kind

    if target_kind == "b":
        return np.asarray(source != 0, dtype=target)
    if source_kind == "b" and target_kind in {"f", "i", "u"}:
        return source.astype(target, copy=False)
    if target_kind in {"i", "u"}:
        limits = np.iinfo(target)
        if source_kind == "f":
            if source.size and not np.all(np.isfinite(source)):
                return None
            candidate = np.trunc(source)
            if not _float_values_fit_integer_dtype(candidate, limits):
                return None
        elif source_kind in {"i", "u"}:
            candidate = source
            if not _integer_values_fit_dtype(candidate, limits):
                return None
        else:
            return None
        return candidate.astype(target, copy=False)
    if target_kind == "f" and source_kind in {"b", "f", "i", "u"}:
        with np.errstate(all="ignore"):
            return source.astype(target, copy=False)
    return None


def _float_values_fit_integer_dtype(
    values: np.ndarray,
    limits: Any,
) -> bool:
    """Check truncated floats against integer limits without float-bound rounding."""

    array = np.asarray(values)
    if not array.size:
        return True
    minimum = int(array.min())
    maximum = int(array.max())
    return limits.min <= minimum and maximum <= limits.max


def _integer_values_fit_dtype(
    values: np.ndarray,
    limits: Any,
) -> bool:
    """Check integer extrema using Python integers across signedness boundaries."""

    array = np.asarray(values)
    if not array.size:
        return True
    minimum = int(array.min())
    maximum = int(array.max())
    return limits.min <= minimum and maximum <= limits.max
