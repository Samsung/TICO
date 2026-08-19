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

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np

from tico.circle._schema import circle_schema
from tico.circle.graph import as_list, OPTIONAL_TENSOR_INDEX
from tico.circle.passes.optimization.fold.evaluators.base import (
    ConstantEvaluation,
    ConstantEvaluationContext,
    ConstantEvaluator,
    ConstantEvaluatorRegistry,
    contract_is_dense_value,
    contract_is_fully_static,
)
from tico.circle.value import TensorValue


@dataclass(frozen=True)
class HeavyConstantEvaluatorPolicy:
    """Control conservative semantic limits for expensive constant evaluators."""

    allow_dequantize: bool = True
    allow_fully_connected: bool = True
    allow_depthwise_conv2d: bool = True
    allow_densify: bool = True
    allow_sparse_to_dense: bool = True
    maximum_sparse_rank: int = 8

    def __post_init__(self) -> None:
        """Normalize flags and reject a non-positive sparse-rank limit."""

        for field_name in (
            "allow_dequantize",
            "allow_fully_connected",
            "allow_depthwise_conv2d",
            "allow_densify",
            "allow_sparse_to_dense",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        object.__setattr__(self, "maximum_sparse_rank", int(self.maximum_sparse_rank))
        if self.maximum_sparse_rank <= 0:
            raise ValueError("maximum_sparse_rank must be positive.")


class DequantizeEvaluator(ConstantEvaluator):
    """Fold FLOAT16 conversion or affine integer dequantization to FLOAT32."""

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require the dequantized input tensor to be an inline constant."""

        return (0,)

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Apply exact serialized affine parameters to one constant tensor."""

        if len(context.input_indices) != 1 or len(context.output_indices) != 1:
            return None
        input_contract = context.input_contract(0)
        output_contract = context.output_contract()
        if not all(
            contract_is_fully_static(contract)
            for contract in (input_contract, output_contract)
        ):
            return None
        if input_contract.sparsity is not None:
            return None
        if not contract_is_dense_value(output_contract):
            return None
        if input_contract.shape != output_contract.shape:
            return None
        if output_contract.quantization is not None:
            return None

        input_spec = context.codec.registry.get(input_contract.tensor_type)
        output_spec = context.codec.registry.get(output_contract.tensor_type)
        if input_spec is None or output_spec is None:
            return None
        if output_spec.name != "FLOAT32":
            return None

        source = context.input_value(0).data
        if input_spec.name == "FLOAT16":
            if input_contract.quantization is not None:
                return None
            result = np.asarray(source, dtype=np.float32)
        else:
            if input_spec.logical_dtype.kind not in {"i", "u"}:
                return None
            quantization = input_contract.quantization
            if quantization is None or not quantization.scale:
                return None
            result = _affine_dequantize(
                source,
                input_contract.shape,
                quantization.scale,
                quantization.zero_point,
                quantization.quantized_dimension,
            )
            if result is None:
                return None

        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=np.asarray(result, dtype=np.float32),
                    quantization=None,
                ),
            )
        )


class FullyConnectedEvaluator(ConstantEvaluator):
    """Fold a static non-quantized FLOAT32 FULLY_CONNECTED operation."""

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require data, weights, and any present bias to be constants."""

        positions = [0, 1]
        if len(context.input_indices) > 2:
            if context.input_indices[2] != OPTIONAL_TENSOR_INDEX:
                positions.append(2)
        return tuple(positions)

    def estimate_compute_cost(self, context: ConstantEvaluationContext) -> int:
        """Estimate multiply-add work from static input and weight shapes."""

        if len(context.input_contracts) < 2:
            return super().estimate_compute_cost(context)
        input_contract = context.input_contracts[0]
        weight_contract = context.input_contracts[1]
        if input_contract is None or weight_contract is None:
            return super().estimate_compute_cost(context)
        if weight_contract.rank != 2 or weight_contract.shape[1] <= 0:
            return super().estimate_compute_cost(context)
        batches = input_contract.element_count // weight_contract.shape[1]
        return max(0, batches * weight_contract.shape[0] * weight_contract.shape[1])

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Evaluate a default-format dense matrix multiplication and optional bias."""

        if len(context.input_indices) not in {2, 3}:
            return None
        if len(context.output_indices) != 1:
            return None
        input_contract = context.input_contract(0)
        weight_contract = context.input_contract(1)
        output_contract = context.output_contract()
        bias_contract = None
        if len(context.input_indices) == 3:
            if context.input_indices[2] != OPTIONAL_TENSOR_INDEX:
                bias_contract = context.input_contract(2)
        contracts = [input_contract, weight_contract, output_contract]
        if bias_contract is not None:
            contracts.append(bias_contract)
        if not all(
            contract_is_fully_static(contract) and contract_is_dense_value(contract)
            for contract in contracts
        ):
            return None
        if any(contract.quantization is not None for contract in contracts):
            return None
        if weight_contract.rank != 2 or input_contract.rank == 0:
            return None

        specs = [
            context.codec.registry.get(contract.tensor_type) for contract in contracts
        ]
        if any(spec is None or spec.name != "FLOAT32" for spec in specs):
            return None
        options = context.options
        if int(getattr(options, "weightsFormat", 0) or 0) != 0:
            return None

        units, input_size = weight_contract.shape
        if units <= 0 or input_size <= 0:
            return None
        if input_contract.element_count % input_size != 0:
            return None
        keep_num_dims = bool(getattr(options, "keepNumDims", False))
        if keep_num_dims:
            if input_contract.shape[-1] != input_size:
                return None
            expected_output_shape = input_contract.shape[:-1] + (units,)
        else:
            expected_output_shape = (
                input_contract.element_count // input_size,
                units,
            )
        if output_contract.shape != expected_output_shape:
            return None
        if bias_contract is not None and bias_contract.shape != (units,):
            return None

        data = np.asarray(context.input_value(0).data, dtype=np.float32)
        weights = np.asarray(context.input_value(1).data, dtype=np.float32)
        flat = data.reshape(-1, input_size)
        with np.errstate(all="ignore"):
            result = np.matmul(flat, weights.T).astype(np.float32, copy=False)
        if bias_contract is not None:
            bias = np.asarray(context.input_value(2).data, dtype=np.float32)
            result = np.add(result, bias.reshape(1, units), dtype=np.float32)
        result = _apply_fused_activation(result, context.options)
        if result is None:
            return None
        result = np.asarray(result, dtype=np.float32).reshape(output_contract.shape)
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


class DepthwiseConv2DEvaluator(ConstantEvaluator):
    """Fold a static non-quantized FLOAT32 NHWC DEPTHWISE_CONV_2D operation."""

    def __init__(self, *, padding_same: int = 0, padding_valid: int = 1) -> None:
        """Bind Circle padding enum values used by the generated schema."""

        self.padding_same = int(padding_same)
        self.padding_valid = int(padding_valid)

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require input, filter, and bias tensors to be constants."""

        if len(context.input_indices) != 3:
            return ()
        return (0, 1, 2)

    def estimate_compute_cost(self, context: ConstantEvaluationContext) -> int:
        """Estimate scalar multiply-add work from output and filter shapes."""

        if len(context.input_contracts) < 2 or not context.output_contracts:
            return super().estimate_compute_cost(context)
        filter_contract = context.input_contracts[1]
        output_contract = context.output_contracts[0]
        if filter_contract is None or filter_contract.rank != 4:
            return super().estimate_compute_cost(context)
        return max(
            0,
            output_contract.element_count
            * filter_contract.shape[1]
            * filter_contract.shape[2],
        )

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Evaluate depthwise convolution with stride, dilation, and padding."""

        if len(context.input_indices) != 3 or len(context.output_indices) != 1:
            return None
        input_contract = context.input_contract(0)
        filter_contract = context.input_contract(1)
        bias_contract = context.input_contract(2)
        output_contract = context.output_contract()
        contracts = (
            input_contract,
            filter_contract,
            bias_contract,
            output_contract,
        )
        if not all(
            contract_is_fully_static(contract) and contract_is_dense_value(contract)
            for contract in contracts
        ):
            return None
        if any(contract.quantization is not None for contract in contracts):
            return None
        if (
            input_contract.rank != 4
            or filter_contract.rank != 4
            or bias_contract.rank != 1
            or output_contract.rank != 4
        ):
            return None
        specs = [
            context.codec.registry.get(contract.tensor_type) for contract in contracts
        ]
        if any(spec is None or spec.name != "FLOAT32" for spec in specs):
            return None

        batch, input_height, input_width, input_channels = input_contract.shape
        one, filter_height, filter_width, output_channels = filter_contract.shape
        if one != 1:
            return None
        options = context.options
        depth_multiplier = int(getattr(options, "depthMultiplier", 0) or 0)
        stride_h = int(getattr(options, "strideH", 0) or 0)
        stride_w = int(getattr(options, "strideW", 0) or 0)
        dilation_h = int(getattr(options, "dilationHFactor", 1) or 1)
        dilation_w = int(getattr(options, "dilationWFactor", 1) or 1)
        if min(depth_multiplier, stride_h, stride_w, dilation_h, dilation_w) <= 0:
            return None
        if output_channels != input_channels * depth_multiplier:
            return None
        if bias_contract.shape != (output_channels,):
            return None

        padding = int(getattr(options, "padding", -1))
        geometry = _conv_geometry(
            input_height,
            input_width,
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            padding,
            padding_same=self.padding_same,
            padding_valid=self.padding_valid,
        )
        if geometry is None:
            return None
        output_height, output_width, pad_top, pad_left = geometry
        expected_shape = (batch, output_height, output_width, output_channels)
        if output_contract.shape != expected_shape:
            return None

        input_value = np.asarray(context.input_value(0).data, dtype=np.float32)
        filter_value = np.asarray(context.input_value(1).data, dtype=np.float32)
        bias_value = np.asarray(context.input_value(2).data, dtype=np.float32)
        result = np.zeros(expected_shape, dtype=np.float32)
        reshaped_filter = filter_value.reshape(
            filter_height,
            filter_width,
            input_channels,
            depth_multiplier,
        )
        output_view = result.reshape(
            batch,
            output_height,
            output_width,
            input_channels,
            depth_multiplier,
        )
        for output_y in range(output_height):
            input_y_origin = output_y * stride_h - pad_top
            for output_x in range(output_width):
                input_x_origin = output_x * stride_w - pad_left
                accumulator = output_view[:, output_y, output_x, :, :]
                for filter_y in range(filter_height):
                    input_y = input_y_origin + filter_y * dilation_h
                    if input_y < 0 or input_y >= input_height:
                        continue
                    for filter_x in range(filter_width):
                        input_x = input_x_origin + filter_x * dilation_w
                        if input_x < 0 or input_x >= input_width:
                            continue
                        source = input_value[:, input_y, input_x, :, None]
                        weight = reshaped_filter[filter_y, filter_x, :, :][None, :, :]
                        accumulator += source * weight
        result += bias_value.reshape(1, 1, 1, output_channels)
        activated = _apply_fused_activation(result, options)
        if activated is None:
            return None
        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=np.asarray(activated, dtype=np.float32),
                    quantization=None,
                ),
            )
        )


class DensifyEvaluator(ConstantEvaluator):
    """Fold an unblocked DENSE/SPARSE_CSR constant into dense storage."""

    def __init__(self, *, maximum_rank: int = 8) -> None:
        """Set the largest sparse tensor rank accepted by this evaluator."""

        self.maximum_rank = int(maximum_rank)
        if self.maximum_rank <= 0:
            raise ValueError("maximum_rank must be positive.")

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Decode sparse storage directly because the dense codec cannot read it."""

        return ()

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Expand one sparse constant with no block mapping into its dense shape."""

        if len(context.input_indices) != 1 or len(context.output_indices) != 1:
            return None
        input_contract = context.input_contract(0)
        output_contract = context.output_contract()
        if not all(
            contract_is_fully_static(contract)
            for contract in (input_contract, output_contract)
        ):
            return None
        if input_contract.sparsity is None:
            return None
        if context.input_indices[0] in context.graph.inputs:
            return None
        if not contract_is_dense_value(output_contract):
            return None
        if input_contract.rank == 0 or input_contract.rank > self.maximum_rank:
            return None
        if input_contract.shape != output_contract.shape:
            return None
        if input_contract.tensor_type != output_contract.tensor_type:
            return None
        if (
            input_contract.quantization is not None
            or output_contract.quantization is not None
        ):
            return None

        spec = context.codec.registry.get(input_contract.tensor_type)
        if spec is None or spec.packed or spec.name not in {"FLOAT16", "FLOAT32"}:
            return None
        metadata = _parse_sparsity(input_contract.sparsity, input_contract.shape)
        if metadata is None:
            return None
        traversal_order, levels = metadata
        coordinates = _enumerate_sparse_coordinates(levels)
        if coordinates is None:
            return None
        payload = _input_buffer_payload(context, 0)
        if payload is None:
            return None
        value_count = len(coordinates)
        storage_dtype = spec.storage_dtype.newbyteorder("<")
        if len(payload) != value_count * storage_dtype.itemsize:
            return None
        sparse_values = np.frombuffer(payload, dtype=storage_dtype, count=value_count)
        sparse_values = sparse_values.astype(spec.logical_dtype, copy=True)
        dense = np.zeros(output_contract.shape, dtype=spec.logical_dtype)
        for coordinate, value in zip(coordinates, sparse_values):
            original_coordinate = [0] * input_contract.rank
            for traversal_axis, coordinate_value in enumerate(coordinate):
                original_axis = traversal_order[traversal_axis]
                original_coordinate[original_axis] = coordinate_value
            dense[tuple(original_coordinate)] = value
        return ConstantEvaluation(
            outputs=(
                TensorValue(
                    tensor_type=output_contract.tensor_type,
                    shape=output_contract.shape,
                    data=dense,
                    quantization=output_contract.quantization,
                ),
            )
        )


class SparseToDenseEvaluator(ConstantEvaluator):
    """Fold the empty-indices SPARSE_TO_DENSE default-fill pattern."""

    def constant_input_positions(
        self,
        context: ConstantEvaluationContext,
    ) -> tuple[int, ...]:
        """Require only output shape and default value for the empty-indices case."""

        return (1, 3)

    def evaluate(
        self,
        context: ConstantEvaluationContext,
    ) -> ConstantEvaluation | None:
        """Materialize an output filled with the scalar default value."""

        if len(context.input_indices) != 4 or len(context.output_indices) != 1:
            return None
        indices_contract = context.input_contract(0)
        shape_contract = context.input_contract(1)
        values_contract = context.input_contract(2)
        default_contract = context.input_contract(3)
        output_contract = context.output_contract()
        contracts = (
            indices_contract,
            shape_contract,
            values_contract,
            default_contract,
            output_contract,
        )
        if not all(contract_is_fully_static(contract) for contract in contracts):
            return None
        if not any(dimension == 0 for dimension in indices_contract.shape):
            return None
        if not all(
            contract_is_dense_value(contract)
            for contract in (shape_contract, default_contract, output_contract)
        ):
            return None
        if default_contract.element_count != 1:
            return None
        if indices_contract.tensor_type != shape_contract.tensor_type:
            return None
        if values_contract.tensor_type != output_contract.tensor_type:
            return None
        if default_contract.tensor_type != output_contract.tensor_type:
            return None
        if default_contract.quantization != output_contract.quantization:
            return None

        shape_value = context.input_value(1)
        if shape_value.data.dtype.kind not in {"i", "u"}:
            return None
        requested_shape = tuple(int(value) for value in shape_value.data.reshape(-1))
        if any(dimension < 0 for dimension in requested_shape):
            return None
        if requested_shape != output_contract.shape:
            return None
        spec = context.codec.registry.get(output_contract.tensor_type)
        if spec is None or spec.packed:
            return None
        default_value = context.input_value(3)
        scalar = default_value.data.reshape(-1)[0]
        result = np.full(output_contract.shape, scalar, dtype=spec.logical_dtype)
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


def register_heavy_constant_evaluators(
    registry: ConstantEvaluatorRegistry,
    *,
    policy: HeavyConstantEvaluatorPolicy | None = None,
    builtin_codes: Mapping[str, int] | None = None,
    padding_values: Mapping[str, int] | None = None,
) -> ConstantEvaluatorRegistry:
    """Register enabled heavy evaluators and return the supplied registry."""

    selected = policy or HeavyConstantEvaluatorPolicy()
    codes = {
        str(name).upper(): int(value) for name, value in (builtin_codes or {}).items()
    }
    paddings = {
        str(name).upper(): int(value) for name, value in (padding_values or {}).items()
    }

    def code(name: str) -> int:
        if name in codes:
            return codes[name]
        return _schema_enum_value("BuiltinOperator", name)

    def padding(name: str, fallback: int) -> int:
        if name in paddings:
            return paddings[name]
        return _maybe_schema_enum_value("Padding", name, fallback)

    entries: list[tuple[str, ConstantEvaluator]] = []
    if selected.allow_dequantize:
        entries.append(("DEQUANTIZE", DequantizeEvaluator()))
    if selected.allow_fully_connected:
        entries.append(("FULLY_CONNECTED", FullyConnectedEvaluator()))
    if selected.allow_depthwise_conv2d:
        entries.append(
            (
                "DEPTHWISE_CONV_2D",
                DepthwiseConv2DEvaluator(
                    padding_same=padding("SAME", 0),
                    padding_valid=padding("VALID", 1),
                ),
            )
        )
    if selected.allow_densify:
        entries.append(
            ("DENSIFY", DensifyEvaluator(maximum_rank=selected.maximum_sparse_rank))
        )
    if selected.allow_sparse_to_dense:
        entries.append(("SPARSE_TO_DENSE", SparseToDenseEvaluator()))
    for name, evaluator in entries:
        registry.register(code(name), evaluator)
    return registry


def _affine_dequantize(
    source: np.ndarray[Any, Any],
    shape: Sequence[int],
    scales: Sequence[float],
    zero_points: Sequence[int],
    quantized_dimension: int,
) -> np.ndarray[Any, Any] | None:
    """Apply per-tensor or per-axis affine dequantization with broadcast parameters."""

    if not scales or len(scales) != len(zero_points):
        return None
    rank = len(shape)
    if len(scales) == 1:
        scale = np.float32(scales[0])
        zero_point = np.float32(zero_points[0])
        return (np.asarray(source, dtype=np.float32) - zero_point) * scale
    if rank == 0:
        return None
    axis = int(quantized_dimension)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank or shape[axis] != len(scales):
        return None
    broadcast_shape = [1] * rank
    broadcast_shape[axis] = len(scales)
    scale = np.asarray(scales, dtype=np.float32).reshape(broadcast_shape)
    zero_point = np.asarray(zero_points, dtype=np.float32).reshape(broadcast_shape)
    return (np.asarray(source, dtype=np.float32) - zero_point) * scale


def _apply_fused_activation(
    value: np.ndarray[Any, Any],
    options: Any,
) -> np.ndarray[Any, Any] | None:
    """Apply standard Circle fused activation enum values to FLOAT32 data."""

    activation = int(getattr(options, "fusedActivationFunction", 0) or 0)
    if activation == 0:
        return np.asarray(value, dtype=np.float32)
    if activation == 1:
        return np.maximum(value, np.float32(0.0)).astype(np.float32, copy=False)
    if activation == 2:
        return np.clip(value, np.float32(-1.0), np.float32(1.0)).astype(
            np.float32,
            copy=False,
        )
    if activation == 3:
        return np.clip(value, np.float32(0.0), np.float32(6.0)).astype(
            np.float32,
            copy=False,
        )
    if activation == 4:
        return np.tanh(value).astype(np.float32, copy=False)
    return None


def _conv_geometry(
    input_height: int,
    input_width: int,
    filter_height: int,
    filter_width: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    padding: int,
    *,
    padding_same: int,
    padding_valid: int,
) -> tuple[int, int, int, int] | None:
    """Return output geometry and top-left padding for one 2-D convolution."""

    effective_height = (filter_height - 1) * dilation_h + 1
    effective_width = (filter_width - 1) * dilation_w + 1
    if padding == padding_same:
        output_height = int(ceil(input_height / stride_h))
        output_width = int(ceil(input_width / stride_w))
        total_height = max(
            0,
            (output_height - 1) * stride_h + effective_height - input_height,
        )
        total_width = max(
            0,
            (output_width - 1) * stride_w + effective_width - input_width,
        )
        return output_height, output_width, total_height // 2, total_width // 2
    if padding == padding_valid:
        output_height = max(0, (input_height - effective_height + stride_h) // stride_h)
        output_width = max(0, (input_width - effective_width + stride_w) // stride_w)
        return output_height, output_width, 0, 0
    return None


def _input_buffer_payload(
    context: ConstantEvaluationContext,
    position: int,
) -> bytes | None:
    """Return raw inline bytes for one input tensor without dense-size validation."""

    tensor_index = context.input_indices[position]
    tensors = as_list(context.graph.subgraph.tensors)
    if tensor_index < 0 or tensor_index >= len(tensors):
        return None
    tensor = tensors[tensor_index]
    buffer_index = int(getattr(tensor, "buffer", 0) or 0)
    buffers = as_list(context.document.model.buffers)
    if buffer_index <= 0 or buffer_index >= len(buffers):
        return None
    buffer = buffers[buffer_index]
    if int(getattr(buffer, "offset", 0) or 0) or int(getattr(buffer, "size", 0) or 0):
        return None
    data = getattr(buffer, "data", None)
    if data is None:
        return None
    return bytes(np.ascontiguousarray(np.asarray(data, dtype=np.uint8)).reshape(-1))


def _parse_sparsity(
    sparsity: Any,
    dense_shape: Sequence[int],
) -> tuple[
    tuple[int, ...],
    tuple[tuple[str, int, tuple[int, ...], tuple[int, ...]], ...],
] | None:
    """Parse an unblocked Circle sparsity record into traversal-level metadata."""

    rank = len(dense_shape)
    traversal = tuple(
        int(value) for value in _vector(getattr(sparsity, "traversalOrder", None))
    )
    block_map = tuple(
        int(value) for value in _vector(getattr(sparsity, "blockMap", None))
    )
    dimensions = tuple(_vector(getattr(sparsity, "dimMetadata", None)))
    if block_map:
        return None
    if len(traversal) != rank or sorted(traversal) != list(range(rank)):
        return None
    if len(dimensions) != rank:
        return None

    parsed: list[tuple[str, int, tuple[int, ...], tuple[int, ...]]] = []
    for traversal_axis, metadata in enumerate(dimensions):
        format_value = int(getattr(metadata, "format", -1))
        dense_size = int(getattr(metadata, "denseSize", 0) or 0)
        original_axis = traversal[traversal_axis]
        if format_value == 0:
            if dense_size != int(dense_shape[original_axis]):
                return None
            parsed.append(("dense", dense_size, (), ()))
            continue
        if format_value != 1:
            return None
        segments = tuple(
            int(value)
            for value in _union_vector(getattr(metadata, "arraySegments", None))
        )
        indices = tuple(
            int(value)
            for value in _union_vector(getattr(metadata, "arrayIndices", None))
        )
        if not segments or segments[0] != 0 or any(value < 0 for value in indices):
            return None
        if any(left > right for left, right in zip(segments, segments[1:])):
            return None
        if segments[-1] != len(indices):
            return None
        if any(value >= int(dense_shape[original_axis]) for value in indices):
            return None
        parsed.append(("sparse", int(dense_shape[original_axis]), segments, indices))
    return traversal, tuple(parsed)


def _enumerate_sparse_coordinates(
    levels: Sequence[tuple[str, int, tuple[int, ...], tuple[int, ...]]],
) -> tuple[tuple[int, ...], ...] | None:
    """Enumerate leaf coordinates in the storage order encoded by sparse metadata."""

    prefixes: list[tuple[int, ...]] = [()]
    for kind, dense_size, segments, indices in levels:
        next_prefixes: list[tuple[int, ...]] = []
        if kind == "dense":
            for prefix in prefixes:
                next_prefixes.extend(prefix + (index,) for index in range(dense_size))
        else:
            if len(segments) != len(prefixes) + 1:
                return None
            for parent, prefix in enumerate(prefixes):
                start = segments[parent]
                end = segments[parent + 1]
                next_prefixes.extend(
                    prefix + (indices[index],) for index in range(start, end)
                )
        prefixes = next_prefixes
    return tuple(prefixes)


def _vector(value: Any) -> tuple[Any, ...]:
    """Normalize generated vectors and NumPy arrays to tuples."""

    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return tuple(value.reshape(-1).tolist())
    try:
        return tuple(value)
    except TypeError:
        return ()


def _union_vector(value: Any) -> tuple[Any, ...]:
    """Read the values field of a generated sparse-index union object."""

    if value is None:
        return ()
    for attribute in ("values", "Values"):
        if hasattr(value, attribute):
            return _vector(getattr(value, attribute))
    for method_name in ("ValuesAsNumpy", "valuesAsNumpy"):
        method = getattr(value, method_name, None)
        if callable(method):
            return _vector(method())
    return _vector(value)


def _schema_enum_value(enum_name: str, member_name: str) -> int:
    """Return one generated Circle enum member by symbolic name."""

    schema = circle_schema()
    module = getattr(schema, enum_name, None)
    enum_type = getattr(module, enum_name, None) if module is not None else None
    if enum_type is None:
        enum_type = module
    if enum_type is None or not hasattr(enum_type, member_name):
        raise RuntimeError(f"Circle schema does not provide {enum_name}.{member_name}.")
    return int(getattr(enum_type, member_name))


def _maybe_schema_enum_value(
    enum_name: str,
    member_name: str,
    fallback: int,
) -> int:
    """Return a generated enum value or a stable legacy fallback."""

    try:
        return _schema_enum_value(enum_name, member_name)
    except (AttributeError, ImportError, RuntimeError):
        return int(fallback)


__all__ = [
    "DensifyEvaluator",
    "DepthwiseConv2DEvaluator",
    "DequantizeEvaluator",
    "FullyConnectedEvaluator",
    "HeavyConstantEvaluatorPolicy",
    "SparseToDenseEvaluator",
    "register_heavy_constant_evaluators",
]
