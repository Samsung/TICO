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

import copy
import struct
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization.remove.layout_ops import (
    _check_perm,
    _get_const_data,
    _is_transpose_op,
)
from tico.circle.rewrite import replace_tensor_uses
from .transpose_bounded_layout_region_rules import (
    _ADD_BUILTIN_CODE,
    _OperatorRewritePlan,
    _PAD_BUILTIN_CODE,
    _RegionOpContext,
    _rule_for_operator,
)


@dataclass(frozen=True)
class _PerTensorEncoding:
    """Describe a tensor type and its optional per-tensor affine qparam."""

    tensor_type: int
    scale: float | None
    zero_point: int | None


@dataclass(frozen=True)
class _InputBoundary:
    """Describe one source-layout tensor entering a region through Transpose."""

    region_operator_index: int
    region_input_position: int
    transpose_index: int
    transpose_output_index: int
    source_tensor_index: int
    permutation: tuple[int, ...]


@dataclass(frozen=True)
class _OutputBoundary:
    """Describe one region-layout tensor leaving through inverse Transpose."""

    region_tensor_index: int
    transpose_index: int
    transpose_output_index: int
    permutation: tuple[int, ...]


@dataclass(frozen=True)
class _RegionPlan:
    """Describe one validated Transpose-bounded layout region rewrite."""

    operator_indices: tuple[int, ...]
    source_to_region_permutation: tuple[int, ...]
    region_to_source_permutation: tuple[int, ...]
    input_boundaries: tuple[_InputBoundary, ...]
    output_boundaries: tuple[_OutputBoundary, ...]
    operator_rewrites: tuple[_OperatorRewritePlan, ...]

    @property
    def bypassed_transpose_indices(self) -> tuple[int, ...]:
        """Return unique boundary Transpose indices in graph order."""

        indices = {boundary.transpose_index for boundary in self.input_boundaries}
        indices.update(boundary.transpose_index for boundary in self.output_boundaries)
        return tuple(sorted(indices))


@dataclass(frozen=True)
class _TransposeInfo:
    """Describe one well-formed single-output Transpose operator."""

    operator_index: int
    source_tensor_index: int
    permutation_tensor_index: int
    output_tensor_index: int
    permutation: tuple[int, ...]


def _vector(value: Any) -> tuple[Any, ...]:
    """Convert one generated vector field to an immutable Python tuple."""

    if value is None:
        return ()
    return tuple(value)


def _operator_consumes_data_tensor(
    operator: Any,
    operator_codes: list[Any],
    tensor_index: int,
) -> bool:
    """Return whether an operator consumes a tensor through a data input."""

    rule = _rule_for_operator(operator, operator_codes)
    if rule is None:
        return False
    inputs = as_indices(getattr(operator, "inputs", None))
    return any(
        position < len(inputs) and inputs[position] == tensor_index
        for position in rule.data_input_positions(operator)
    )


def _shape(tensor: Any) -> tuple[int, ...]:
    """Return one tensor shape as a tuple of integers."""

    return tuple(int(value) for value in _vector(getattr(tensor, "shape", None)))


def _shape_signature(tensor: Any) -> tuple[int, ...]:
    """Return one optional tensor shape signature as an integer tuple."""

    return tuple(
        int(value) for value in _vector(getattr(tensor, "shapeSignature", None))
    )


def _is_valid_permutation(permutation: Iterable[int]) -> bool:
    """Return whether values form a complete rank-preserving permutation."""

    values = tuple(int(value) for value in permutation)
    return bool(values) and sorted(values) == list(range(len(values)))


def _inverse_permutation(permutation: Iterable[int]) -> tuple[int, ...] | None:
    """Return the inverse permutation, or None when the input is invalid."""

    values = tuple(int(value) for value in permutation)
    if not _is_valid_permutation(values):
        return None
    inverse = [0] * len(values)
    for output_axis, input_axis in enumerate(values):
        inverse[input_axis] = output_axis
    return tuple(inverse)


def _permuted_shape(
    shape: tuple[int, ...],
    permutation: Iterable[int],
) -> tuple[int, ...] | None:
    """Return a permuted shape, or None for an invalid rank or permutation."""

    values = tuple(int(value) for value in permutation)
    if len(shape) != len(values) or not _is_valid_permutation(values):
        return None
    return tuple(shape[index] for index in values)


def _per_tensor_encoding(tensor: Any) -> _PerTensorEncoding | None:
    """Return one tensor encoding, or None for per-channel activation qparams."""

    tensor_type = int(getattr(tensor, "type", 0))
    quantization = getattr(tensor, "quantization", None)
    if quantization is None:
        return _PerTensorEncoding(tensor_type, None, None)

    scales = tuple(
        float(value) for value in _vector(getattr(quantization, "scale", None))
    )
    zero_points = tuple(
        int(value) for value in _vector(getattr(quantization, "zeroPoint", None))
    )
    if not scales and not zero_points:
        return _PerTensorEncoding(tensor_type, None, None)
    if len(scales) != 1 or len(zero_points) != 1:
        return None
    quantized_dimension = int(getattr(quantization, "quantizedDimension", 0) or 0)
    if quantized_dimension != 0:
        return None
    return _PerTensorEncoding(tensor_type, scales[0], zero_points[0])


def _same_tensor_encoding(first: Any, second: Any) -> bool:
    """Return whether two tensors use the same type and per-tensor qparam."""

    first_encoding = _per_tensor_encoding(first)
    second_encoding = _per_tensor_encoding(second)
    return (
        first_encoding is not None
        and second_encoding is not None
        and first_encoding == second_encoding
    )


def _same_runtime_buffer(first: Any, second: Any) -> bool:
    """Return whether two runtime tensors reference the same Circle buffer."""

    return int(getattr(first, "buffer", 0) or 0) == int(
        getattr(second, "buffer", 0) or 0
    )


def _copy_tensor_contract(target: Any, source: Any) -> None:
    """Copy externally visible tensor metadata without changing its buffer."""

    fields = (
        "name",
        "shape",
        "shapeSignature",
        "type",
        "quantization",
        "isVariable",
        "sparsity",
        "hasRank",
        "variantTensors",
    )
    for field_name in fields:
        if hasattr(source, field_name):
            setattr(target, field_name, copy.deepcopy(getattr(source, field_name)))


def _append_name_suffix(value: Any, suffix: str) -> Any:
    """Append a stable suffix while preserving bytes or string storage."""

    if isinstance(value, bytes):
        return value + suffix.encode("utf-8")
    return f"{value or ''}{suffix}"


def _rename_dead_tensor(tensor: Any, suffix: str) -> None:
    """Rename one bypassed tensor before dead-code compaction."""

    if hasattr(tensor, "name"):
        tensor.name = _append_name_suffix(getattr(tensor, "name", ""), suffix)


def _shape_signature_matches_permutation(
    source: Any,
    transposed: Any,
    permutation: tuple[int, ...],
) -> bool:
    """Return whether optional shape signatures follow one permutation."""

    source_signature = _shape_signature(source)
    transposed_signature = _shape_signature(transposed)
    if not source_signature and not transposed_signature:
        return True
    if not source_signature or not transposed_signature:
        return False
    return _permuted_shape(source_signature, permutation) == transposed_signature


def _tensor_has_visible_use(
    document: CircleDocument,
    graph: Any,
    tensor_index: int,
) -> bool:
    """Return whether a tensor has a consumer or an exported graph use."""

    return bool(
        graph.consumers(tensor_index)
        or tensor_index in graph.outputs
        or _is_signature_output(
            document,
            graph.subgraph_index,
            tensor_index,
        )
    )


def _set_tensor_layout(
    tensor: Any,
    permutation: tuple[int, ...],
) -> bool:
    """Permute one tensor's shape and optional shape signature in place."""

    new_shape = _permuted_shape(_shape(tensor), permutation)
    if new_shape is None:
        return False
    signature = _shape_signature(tensor)
    new_signature = _permuted_shape(signature, permutation) if signature else None
    if signature and new_signature is None:
        return False

    tensor.shape = list(new_shape)
    if new_signature is not None:
        tensor.shapeSignature = list(new_signature)
    return True


def _is_signature_output(
    document: CircleDocument,
    subgraph_index: int,
    tensor_index: int,
) -> bool:
    """Return whether a tensor is exposed by a signature output mapping."""

    return any(
        int(getattr(tensor_map, "tensorIndex", -1)) == tensor_index
        for signature in as_list(getattr(document.model, "signatureDefs", None))
        if int(getattr(signature, "subgraphIndex", -1)) == subgraph_index
        for tensor_map in as_list(getattr(signature, "outputs", None))
    )


def _transpose_info(
    graph: Any,
    operator_index: int,
    operator_codes: list[Any],
) -> _TransposeInfo | None:
    """Return decoded information for one well-formed Transpose operator."""

    operators = as_list(graph.subgraph.operators)
    if operator_index < 0 or operator_index >= len(operators):
        return None
    operator = operators[operator_index]
    if not _is_transpose_op(operator, operator_codes):
        return None

    inputs = as_indices(getattr(operator, "inputs", None))
    outputs = as_indices(getattr(operator, "outputs", None))
    if len(inputs) < 2 or len(outputs) != 1:
        return None
    permutation = _get_const_data(graph, inputs[1])
    if permutation is None or not _is_valid_permutation(permutation):
        return None
    return _TransposeInfo(
        operator_index=operator_index,
        source_tensor_index=inputs[0],
        permutation_tensor_index=inputs[1],
        output_tensor_index=outputs[0],
        permutation=tuple(int(value) for value in permutation),
    )


def _supported_components(
    graph: Any,
    operator_codes: list[Any],
) -> tuple[tuple[int, ...], ...]:
    """Return dataflow-connected components of registered region operators."""

    operators = as_list(graph.subgraph.operators)
    supported = {
        index
        for index, operator in enumerate(operators)
        if _rule_for_operator(operator, operator_codes) is not None
    }
    pending = set(supported)
    components: list[tuple[int, ...]] = []

    while pending:
        seed = min(pending)
        queue: deque[int] = deque([seed])
        component: set[int] = set()
        while queue:
            operator_index = queue.popleft()
            if operator_index in component:
                continue
            component.add(operator_index)
            operator = operators[operator_index]
            rule = _rule_for_operator(operator, operator_codes)
            if rule is None:
                continue

            inputs = as_indices(getattr(operator, "inputs", None))
            for position in rule.data_input_positions(operator):
                if position >= len(inputs):
                    continue
                producer = graph.producer(inputs[position])
                if producer in supported and producer not in component:
                    queue.append(producer)

            outputs = as_indices(getattr(operator, "outputs", None))
            for position in rule.data_output_positions(operator):
                if position >= len(outputs):
                    continue
                tensor_index = outputs[position]
                for consumer in graph.consumers(tensor_index):
                    if consumer not in supported or consumer in component:
                        continue
                    if _operator_consumes_data_tensor(
                        operators[consumer],
                        operator_codes,
                        tensor_index,
                    ):
                        queue.append(consumer)

        pending.difference_update(component)
        components.append(tuple(sorted(component)))

    return tuple(sorted(components, key=lambda values: values[0]))


def _encode_i32_payload_like(original: Any, values: tuple[int, ...]) -> Any:
    """Encode INT32 values while preserving a buffer's common container type."""

    payload = struct.pack(f"<{len(values)}i", *values)
    if isinstance(original, bytes) or original is None:
        return payload
    if isinstance(original, bytearray):
        return bytearray(payload)
    if isinstance(original, list):
        return list(payload)
    if isinstance(original, tuple):
        return tuple(payload)

    try:
        import numpy as np

        if isinstance(original, np.ndarray):
            return np.frombuffer(payload, dtype=np.uint8).copy()
    except ImportError:
        pass
    return payload


def _clone_i32_constant(
    graph: Any,
    tensor_index: int,
    values: tuple[int, ...],
) -> int:
    """Clone one inline INT32 constant with replacement values."""

    tensors = as_list(graph.subgraph.tensors)
    if tensor_index < 0 or tensor_index >= len(tensors):
        raise IndexError(f"Constant tensor index {tensor_index} is out of range.")
    source_tensor = tensors[tensor_index]
    buffer_index = int(getattr(source_tensor, "buffer", 0) or 0)
    buffers = as_list(graph.model.buffers)
    if buffer_index <= 0 or buffer_index >= len(buffers):
        raise ValueError(f"Constant tensor {tensor_index} has no inline buffer.")

    source_buffer = buffers[buffer_index]
    source_data = getattr(source_buffer, "data", None)
    if source_data is None or len(source_data) == 0:
        raise ValueError(f"Constant tensor {tensor_index} has no inline data.")

    new_buffer = copy.deepcopy(source_buffer)
    new_buffer.data = _encode_i32_payload_like(source_data, values)
    if hasattr(new_buffer, "offset"):
        new_buffer.offset = 0
    if hasattr(new_buffer, "size"):
        new_buffer.size = 0
    new_buffer_index = len(buffers)
    graph.model.buffers = [*buffers, new_buffer]

    new_tensor = copy.deepcopy(source_tensor)
    new_tensor.buffer = new_buffer_index
    new_tensor_index = len(tensors)
    suffix = f"::layout_remapped_{new_tensor_index}"
    name = getattr(source_tensor, "name", "")
    if isinstance(name, bytes):
        new_tensor.name = name + suffix.encode("utf-8")
    else:
        new_tensor.name = f"{name}{suffix}"
    graph.subgraph.tensors = [*tensors, new_tensor]
    return new_tensor_index


def _region_data_tensor_indices(
    graph: Any,
    component: tuple[int, ...],
    operator_codes: list[Any],
) -> tuple[int, ...]:
    """Return unique data tensor indices referenced by one region component."""

    operators = as_list(graph.subgraph.operators)
    result: list[int] = []
    seen: set[int] = set()
    for operator_index in component:
        operator = operators[operator_index]
        rule = _rule_for_operator(operator, operator_codes)
        if rule is None:
            continue
        inputs = as_indices(getattr(operator, "inputs", None))
        for position in rule.data_input_positions(operator):
            if position < len(inputs) and inputs[position] not in seen:
                seen.add(inputs[position])
                result.append(inputs[position])
        outputs = as_indices(getattr(operator, "outputs", None))
        for position in rule.data_output_positions(operator):
            if position < len(outputs) and outputs[position] not in seen:
                seen.add(outputs[position])
                result.append(outputs[position])
    return tuple(result)


def _build_region_plan(
    document: CircleDocument,
    graph: Any,
    component: tuple[int, ...],
    operator_codes: list[Any],
) -> _RegionPlan | None:
    """Validate one supported component and return its complete rewrite plan."""

    operators = as_list(graph.subgraph.operators)
    tensors = as_list(graph.subgraph.tensors)
    component_set = set(component)
    input_boundaries: list[_InputBoundary] = []
    output_boundaries: list[_OutputBoundary] = []

    for operator_index in component:
        operator = operators[operator_index]
        rule = _rule_for_operator(operator, operator_codes)
        if rule is None:
            return None
        inputs = as_indices(getattr(operator, "inputs", None))
        outputs = as_indices(getattr(operator, "outputs", None))
        data_inputs = rule.data_input_positions(operator)
        data_outputs = rule.data_output_positions(operator)
        if not data_inputs or not data_outputs:
            return None
        if any(position >= len(inputs) for position in data_inputs):
            return None
        if any(position >= len(outputs) for position in data_outputs):
            return None

        for position in data_inputs:
            tensor_index = inputs[position]
            producer = graph.producer(tensor_index)
            if producer in component_set:
                continue
            if producer is None:
                return None
            transpose = _transpose_info(graph, producer, operator_codes)
            if transpose is None or transpose.output_tensor_index != tensor_index:
                return None
            input_boundaries.append(
                _InputBoundary(
                    region_operator_index=operator_index,
                    region_input_position=position,
                    transpose_index=producer,
                    transpose_output_index=tensor_index,
                    source_tensor_index=transpose.source_tensor_index,
                    permutation=transpose.permutation,
                )
            )

        for position in data_outputs:
            tensor_index = outputs[position]
            if tensor_index in graph.outputs or _is_signature_output(
                document,
                graph.subgraph_index,
                tensor_index,
            ):
                return None
            for consumer in graph.consumers(tensor_index):
                if consumer in component_set:
                    if not _operator_consumes_data_tensor(
                        operators[consumer],
                        operator_codes,
                        tensor_index,
                    ):
                        return None
                    continue
                transpose = _transpose_info(graph, consumer, operator_codes)
                if transpose is None or transpose.source_tensor_index != tensor_index:
                    return None
                output_boundaries.append(
                    _OutputBoundary(
                        region_tensor_index=tensor_index,
                        transpose_index=consumer,
                        transpose_output_index=transpose.output_tensor_index,
                        permutation=transpose.permutation,
                    )
                )

    if not input_boundaries or not output_boundaries:
        return None

    source_to_region = input_boundaries[0].permutation
    if any(boundary.permutation != source_to_region for boundary in input_boundaries):
        return None
    region_to_source = output_boundaries[0].permutation
    if any(boundary.permutation != region_to_source for boundary in output_boundaries):
        return None
    if not _check_perm(list(source_to_region), list(region_to_source)):
        return None
    if _inverse_permutation(source_to_region) != region_to_source:
        return None

    rank = len(source_to_region)
    boundary_tensor_indices = {
        boundary.source_tensor_index for boundary in input_boundaries
    }
    boundary_tensor_indices.update(
        boundary.transpose_output_index for boundary in input_boundaries
    )
    boundary_tensor_indices.update(
        boundary.region_tensor_index for boundary in output_boundaries
    )
    boundary_tensor_indices.update(
        boundary.transpose_output_index for boundary in output_boundaries
    )
    if any(index < 0 or index >= len(tensors) for index in boundary_tensor_indices):
        return None

    for boundary in input_boundaries:
        source_tensor = tensors[boundary.source_tensor_index]
        transposed = tensors[boundary.transpose_output_index]
        if _permuted_shape(_shape(source_tensor), source_to_region) != _shape(
            transposed
        ):
            return None
        if not _shape_signature_matches_permutation(
            source_tensor,
            transposed,
            source_to_region,
        ):
            return None
        if not _same_tensor_encoding(source_tensor, transposed):
            return None
        if not _same_runtime_buffer(source_tensor, transposed):
            return None

    output_contracts: dict[int, tuple[Any, ...]] = {}
    for output_boundary in output_boundaries:
        region_tensor = tensors[output_boundary.region_tensor_index]
        final_tensor = tensors[output_boundary.transpose_output_index]
        if _permuted_shape(_shape(region_tensor), region_to_source) != _shape(
            final_tensor
        ):
            return None
        if not _shape_signature_matches_permutation(
            region_tensor,
            final_tensor,
            region_to_source,
        ):
            return None
        if not _same_tensor_encoding(region_tensor, final_tensor):
            return None
        if not _same_runtime_buffer(region_tensor, final_tensor):
            return None
        if not _tensor_has_visible_use(
            document,
            graph,
            output_boundary.transpose_output_index,
        ):
            return None
        contract = (
            getattr(final_tensor, "name", ""),
            _shape(final_tensor),
            _shape_signature(final_tensor),
            _per_tensor_encoding(final_tensor),
            int(getattr(final_tensor, "buffer", 0) or 0),
        )
        previous = output_contracts.setdefault(
            output_boundary.region_tensor_index,
            contract,
        )
        if previous != contract:
            return None

    data_tensor_indices = _region_data_tensor_indices(
        graph,
        component,
        operator_codes,
    )
    for tensor_index in data_tensor_indices:
        if tensor_index < 0 or tensor_index >= len(tensors):
            return None
        tensor = tensors[tensor_index]
        if len(_shape(tensor)) != rank:
            return None
        signature = _shape_signature(tensor)
        if signature and len(signature) != rank:
            return None
        if _per_tensor_encoding(tensor) is None:
            return None

    operator_rewrites: list[_OperatorRewritePlan] = []
    for operator_index in component:
        operator = operators[operator_index]
        rule = _rule_for_operator(operator, operator_codes)
        if rule is None:
            return None
        rewrite = rule.plan_rewrite(
            _RegionOpContext(
                graph=graph,
                operator_index=operator_index,
                operator=operator,
                source_to_region_permutation=source_to_region,
                region_to_source_permutation=region_to_source,
            )
        )
        if rewrite is None:
            return None
        operator_rewrites.append(rewrite)

    return _RegionPlan(
        operator_indices=component,
        source_to_region_permutation=source_to_region,
        region_to_source_permutation=region_to_source,
        input_boundaries=tuple(input_boundaries),
        output_boundaries=tuple(output_boundaries),
        operator_rewrites=tuple(operator_rewrites),
    )


def _apply_region_plan(
    document: CircleDocument,
    graph: Any,
    plan: _RegionPlan,
) -> None:
    """Apply one validated region rewrite without deleting dead operators."""

    operators = as_list(graph.subgraph.operators)

    for operator_rewrite in plan.operator_rewrites:
        operator = operators[operator_rewrite.operator_index]
        if operator_rewrite.constant_input_rewrites:
            inputs = as_indices(getattr(operator, "inputs", None))
            for rewrite in operator_rewrite.constant_input_rewrites:
                new_tensor_index = _clone_i32_constant(
                    graph,
                    rewrite.source_tensor_index,
                    rewrite.values,
                )
                inputs[rewrite.input_position] = new_tensor_index
            operator.inputs = inputs

        if operator_rewrite.builtin_option_rewrites:
            options = copy.deepcopy(getattr(operator, "builtinOptions", None))
            if options is None:
                raise RuntimeError("Axis-remap rule expected Circle builtin options.")
            for rewrite in operator_rewrite.builtin_option_rewrites:
                if not hasattr(options, rewrite.field_name):
                    raise RuntimeError(
                        "Axis-remap rule references missing builtin option "
                        f"{rewrite.field_name}."
                    )
                setattr(options, rewrite.field_name, rewrite.value)
            operator.builtinOptions = options

    for boundary in plan.input_boundaries:
        operator = operators[boundary.region_operator_index]
        inputs = as_indices(getattr(operator, "inputs", None))
        inputs[boundary.region_input_position] = boundary.source_tensor_index
        operator.inputs = inputs

    tensors = as_list(graph.subgraph.tensors)
    output_boundaries: dict[int, list[_OutputBoundary]] = {}
    for output_boundary in plan.output_boundaries:
        output_boundaries.setdefault(output_boundary.region_tensor_index, []).append(
            output_boundary
        )

    operator_codes = as_list(document.model.operatorCodes)
    for operator_index in plan.operator_indices:
        operator = operators[operator_index]
        rule = _rule_for_operator(operator, operator_codes)
        if rule is None:
            raise RuntimeError(f"Missing region rule for operator {operator_index}.")
        outputs = as_indices(getattr(operator, "outputs", None))
        for output_position in rule.data_output_positions(operator):
            tensor_index = outputs[output_position]
            boundaries = output_boundaries.get(tensor_index, [])
            if boundaries:
                final_tensor = tensors[boundaries[0].transpose_output_index]
                _copy_tensor_contract(tensors[tensor_index], final_tensor)
            elif not _set_tensor_layout(
                tensors[tensor_index],
                plan.region_to_source_permutation,
            ):
                raise RuntimeError(
                    f"Failed to update tensor {tensor_index} to source layout."
                )

    for output_boundary in plan.output_boundaries:
        final_tensor = tensors[output_boundary.transpose_output_index]
        _rename_dead_tensor(
            final_tensor,
            f"::dead_layout_{output_boundary.transpose_index}",
        )
        replacement = replace_tensor_uses(
            document.model,
            subgraph_index=graph.subgraph_index,
            old_tensor_index=output_boundary.transpose_output_index,
            new_tensor_index=output_boundary.region_tensor_index,
        )
        if not replacement.modified:
            raise RuntimeError(
                "Layout-region elimination expected a visible Transpose output use."
            )


class EliminateTransposeBoundedLayoutRegionPass(CirclePass):
    """Rewrite registered Transpose-bounded regions into their source layout.

    The pass finds dataflow-connected components composed only of operators with
    registered region rules. The registry includes rank-preserving unary,
    same-shape binary, and same-shape variadic elementwise operators plus
    axis-remapped CONCATENATION, constant-padding operators, TILE, SLICE,
    SPLIT, and SPLIT_V. Every external data input must enter through the same
    Transpose
    permutation, and every external data output must leave through its inverse.
    The complete component is then executed directly in the source layout,
    operator-local constants and builtin options are rewritten through their rules,
    and the boundary Transpose outputs are bypassed. Dead-code elimination removes
    Transpose nodes.

    Float tensors and per-tensor quantized activation tensors are supported.
    Per-channel activation qparams are rejected because the target backend only
    supports per-tensor activation quantization and this pass does not remap a
    quantized axis.
    """

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Rewrite every supported bounded layout region in every subgraph."""

        changes = 0
        diagnostics: list[str] = []
        for subgraph_index in range(document.subgraph_count):
            while True:
                graph = document.graph(subgraph_index)
                operator_codes = as_list(document.model.operatorCodes)
                rewritten = False
                for component in _supported_components(graph, operator_codes):
                    plan = _build_region_plan(
                        document,
                        graph,
                        component,
                        operator_codes,
                    )
                    if plan is None:
                        continue
                    _apply_region_plan(document, graph, plan)
                    bypassed = plan.bypassed_transpose_indices
                    changes += len(bypassed)
                    diagnostics.append(
                        f"Subgraph {subgraph_index}: rewrote layout region "
                        f"{list(plan.operator_indices)} and bypassed Transpose "
                        f"operators {list(bypassed)}."
                    )
                    rewritten = True
                    break
                if not rewritten:
                    break

        context.logger.debug(
            "EliminateTransposeBoundedLayoutRegionPass bypassed %d Transpose "
            "operators.",
            changes,
        )
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
            diagnostics=tuple(diagnostics),
        )
