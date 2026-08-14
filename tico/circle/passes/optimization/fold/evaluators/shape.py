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

from math import prod
from typing import Any

import numpy as np

from tico.circle.analysis import TensorContract
from tico.circle.graph import OPTIONAL_TENSOR_INDEX
from tico.circle.passes.optimization.fold.evaluators.base import (
    ConstantEvaluation,
    ConstantEvaluationContext,
    ConstantEvaluator,
    contract_is_dense_value,
    contract_is_fully_static,
)
from tico.circle.value import TensorValue


class ReshapeEvaluator(ConstantEvaluator):
    """Fold a constant RESHAPE whose requested and serialized shapes agree."""

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require data and any present shape tensor to be constants."""

        positions = [0]
        if len(context.input_indices) > 1:
            if context.input_indices[1] != OPTIONAL_TENSOR_INDEX:
                positions.append(1)
        return tuple(positions)

    def estimate_compute_cost(self, context: ConstantEvaluationContext) -> int:
        """Treat reshape as metadata-only computation for budget accounting."""

        return 0

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Reshape the logical payload after validating element count and target."""

        if len(context.input_indices) not in {1, 2}:
            return None
        if len(context.output_indices) != 1:
            return None
        input_contract = context.input_contract(0)
        output_contract = context.output_contract()
        if not _view_contracts_match(input_contract, output_contract):
            return None
        if input_contract.element_count != output_contract.element_count:
            return None

        requested_shape = _reshape_target(context)
        if requested_shape is None:
            return None
        if len(context.input_indices) > 1:
            if context.input_indices[1] != OPTIONAL_TENSOR_INDEX:
                shape_contract = context.input_contract(1)
                shape_spec = context.codec.registry.get(shape_contract.tensor_type)
                if (
                    not contract_is_fully_static(shape_contract)
                    or not contract_is_dense_value(shape_contract)
                    or shape_contract.quantization is not None
                    or shape_spec is None
                    or shape_spec.name not in {"INT32", "INT64"}
                ):
                    return None
        resolved_shape = _resolve_reshape_shape(
            input_contract.element_count,
            requested_shape,
        )
        if resolved_shape != output_contract.shape:
            return None

        value = context.input_value(0)
        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=value.data.reshape(output_contract.shape),
                    quantization=output_contract.quantization,
                ),
            )
        )


class SqueezeEvaluator(ConstantEvaluator):
    """Fold a constant SQUEEZE with static axes and exact output metadata."""

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require the squeezed data tensor to be constant."""

        return (0,)

    def estimate_compute_cost(self, context: ConstantEvaluationContext) -> int:
        """Treat squeeze as metadata-only computation for budget accounting."""

        return 0

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Remove size-one dimensions selected by SqueezeOptions."""

        if len(context.input_indices) != 1 or len(context.output_indices) != 1:
            return None
        input_contract = context.input_contract(0)
        output_contract = context.output_contract()
        if not _view_contracts_match(input_contract, output_contract):
            return None

        raw_axes = tuple(
            int(axis) for axis in (_option_vector(context.options, "squeezeDims") or ())
        )
        if raw_axes:
            normalized_axes: list[int] = []
            for axis in raw_axes:
                normalized = axis + input_contract.rank if axis < 0 else axis
                if normalized < 0 or normalized >= input_contract.rank:
                    return None
                if normalized in normalized_axes:
                    return None
                if input_contract.shape[normalized] != 1:
                    return None
                normalized_axes.append(normalized)
            removed = set(normalized_axes)
        else:
            removed = {
                axis
                for axis, dimension in enumerate(input_contract.shape)
                if dimension == 1
            }
        expected_shape = tuple(
            dimension
            for axis, dimension in enumerate(input_contract.shape)
            if axis not in removed
        )
        if expected_shape != output_contract.shape:
            return None

        value = context.input_value(0)
        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=value.data.reshape(output_contract.shape),
                    quantization=output_contract.quantization,
                ),
            )
        )


class ShapeEvaluator(ConstantEvaluator):
    """Fold SHAPE from a fully static input contract without reading its value."""

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require no constant input because only serialized shape metadata is read."""

        return ()

    def estimate_compute_cost(self, context: ConstantEvaluationContext) -> int:
        """Treat shape materialization as zero-cost metadata evaluation."""

        return 0

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Materialize one static tensor shape as INT32 or INT64 values."""

        if len(context.input_indices) != 1 or len(context.output_indices) != 1:
            return None
        input_contract = context.input_contract(0)
        output_contract = context.output_contract()
        if not contract_is_fully_static(input_contract):
            return None
        if not contract_is_fully_static(output_contract):
            return None
        if not contract_is_dense_value(output_contract):
            return None
        if output_contract.quantization is not None:
            return None
        if output_contract.shape != (input_contract.rank,):
            return None

        output_spec = context.codec.registry.get(output_contract.tensor_type)
        if output_spec is None or output_spec.name not in {"INT32", "INT64"}:
            return None
        if context.options is not None and hasattr(context.options, "outType"):
            raw_output_type = getattr(context.options, "outType")
            if (
                raw_output_type is None
                or int(raw_output_type) != output_contract.tensor_type
            ):
                return None

        limits = np.iinfo(output_spec.logical_dtype)
        dimensions = input_contract.shape
        if any(
            dimension < limits.min or dimension > limits.max for dimension in dimensions
        ):
            return None
        data = np.asarray(dimensions, dtype=output_spec.logical_dtype)
        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=data,
                    quantization=None,
                ),
            )
        )


def _view_contracts_match(
    input_contract: TensorContract,
    output_contract: TensorContract,
) -> bool:
    """Return whether a dense view rewrite preserves type and quantization semantics."""

    return (
        contract_is_fully_static(input_contract)
        and contract_is_fully_static(output_contract)
        and contract_is_dense_value(input_contract)
        and contract_is_dense_value(output_contract)
        and input_contract.tensor_type == output_contract.tensor_type
        and input_contract.quantization == output_contract.quantization
    )


def _reshape_target(
    context: ConstantEvaluationContext,
) -> tuple[int, ...] | None:
    """Read the requested RESHAPE target from an input tensor or options."""

    if len(context.input_indices) > 1:
        if context.input_indices[1] != OPTIONAL_TENSOR_INDEX:
            value = context.input_value(1)
            if value.data.dtype.kind not in {"i", "u"} or value.data.ndim != 1:
                return None
            return tuple(int(dimension) for dimension in value.data.tolist())
    values = _option_vector(context.options, "newShape")
    if values is None:
        return None
    return tuple(int(dimension) for dimension in values)


def _resolve_reshape_shape(
    input_element_count: int,
    requested_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Resolve one inferred dimension and validate the element count."""

    if any(dimension < -1 for dimension in requested_shape):
        return None
    inferred = [
        position
        for position, dimension in enumerate(requested_shape)
        if dimension == -1
    ]
    if len(inferred) > 1:
        return None
    resolved = list(requested_shape)
    known_dimensions = [dimension for dimension in resolved if dimension != -1]
    known_count = prod(known_dimensions, start=1)
    if inferred:
        if known_count == 0 or input_element_count % known_count != 0:
            return None
        resolved[inferred[0]] = input_element_count // known_count
    elif known_count != input_element_count:
        return None
    return tuple(resolved)


def _option_vector(
    options: object | None,
    field_name: str,
) -> tuple[Any, ...] | None:
    """Return an optional generated vector field as a plain tuple."""

    if options is None or not hasattr(options, field_name):
        return None
    value = getattr(options, field_name)
    if value is None:
        return ()
    return tuple(value)
