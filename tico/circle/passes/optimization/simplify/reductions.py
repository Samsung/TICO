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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from tico.circle._object import clone_object, ObjectFactory
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list, CircleGraph
from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.optimization._pattern_utils import (
    capture_supporting_operators,
    decode_integer_constant,
    normalize_axes,
    operator_is_live,
    producer_matching,
    supported_float_contract,
    SupportingOperatorsPlan,
    tensor_name,
)
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    operator_builtin_code,
    operator_is_plain,
    operator_version,
    OptimizationSchemaResolver,
    output_shape_matches_transpose,
    tensor_contract,
)
from tico.circle.passes.optimization.policy import FloatingPointRewritePolicy
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
)
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True)
class ReductionSimplificationPolicy:
    """Control numerical and shape limits for reduction simplification."""

    floating_point_policy: FloatingPointRewritePolicy = (
        FloatingPointRewritePolicy.ALLOW_REASSOCIATION
    )
    maximum_merged_axes: int = 64

    def __post_init__(self) -> None:
        """Reject an axis limit that cannot represent one reduction."""

        if self.maximum_merged_axes <= 0:
            raise ValueError("maximum_merged_axes must be positive.")


@dataclass(frozen=True, kw_only=True)
class ReductionRewritePlan(SupportingOperatorsPlan):
    """Describe one MEAN replacement with a newly interned axis constant."""

    replacement_input: int
    axes: tuple[int, ...]
    axes_tensor_type: int
    axes_dtype: str
    replacement_version: int
    replacement_options_type: int
    replacement_options: Any


class _ReductionRuleBase(CircleRewriteRule[ReductionRewritePlan]):
    """Provide common MEAN matching and transactional replacement behavior."""

    def __init__(
        self,
        *,
        codes: Mapping[str, int],
        options_types: Mapping[str, int],
        float32_type: int,
        integer_axis_types: Sequence[int],
        codec: TensorValueCodec,
        object_factory: ObjectFactory | None,
        policy: ReductionSimplificationPolicy,
    ) -> None:
        """Store immutable services used by reduction rules."""

        self.codes = dict(codes)
        self.options_types = dict(options_types)
        self.float32_type = int(float32_type)
        self.integer_axis_types = frozenset(
            int(tensor_type) for tensor_type in integer_axis_types
        )
        self.codec = codec
        self.object_factory = object_factory
        self.policy = policy

    def apply(
        self,
        document: CircleDocument,
        plan: ReductionRewritePlan,
        context: CirclePassContext,
    ) -> RewriteApplication:
        """Replace the anchor MEAN while retaining old producers for external DCE."""

        del context
        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=plan.subgraph_index,
        )
        builder = CircleBuilder(
            document,
            subgraph_index=plan.subgraph_index,
            codec=self.codec,
            object_factory=self.object_factory,
        )
        try:
            dtype = np.dtype(plan.axes_dtype)
            axes_value = TensorValue.from_values(
                plan.axes_tensor_type,
                np.asarray(plan.axes, dtype=dtype),
                dtype=dtype,
            )
            axes_tensor = builder.add_constant(
                tensor_name(
                    document.graph(plan.subgraph_index),
                    plan.anchor.inputs[1],
                    "reduction_axes",
                ),
                axes_value,
            )
            replacement = builder.make_operator(
                self.codes["MEAN"],
                inputs=(plan.replacement_input, axes_tensor),
                outputs=plan.anchor.outputs,
                version=plan.replacement_version,
                builtin_options_type=plan.replacement_options_type,
                builtin_options=clone_object(plan.replacement_options),
            )
            builder.replace_operator(plan.anchor_operator_index, replacement)
        except Exception:
            checkpoint.rollback(document)
            raise
        return RewriteApplication(changes=1)

    def _mean(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
    ) -> tuple[Any, tuple[int, int], bool] | None:
        """Return one plain static FLOAT32 MEAN and its options."""

        operators = as_list(graph.subgraph.operators)
        if operator_index < 0 or operator_index >= len(operators):
            return None
        operator = operators[operator_index]
        if operator_builtin_code(document.model, operator) != self.codes["MEAN"]:
            return None
        if not operator_is_plain(operator):
            return None
        if int(getattr(operator, "builtinOptionsType", 0) or 0) != (
            self.options_types["ReducerOptions"]
        ):
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None or not hasattr(options, "keepDims"):
            return None
        inputs = tuple(as_indices(getattr(operator, "inputs", None)))
        outputs = tuple(as_indices(getattr(operator, "outputs", None)))
        if len(inputs) != 2 or len(outputs) != 1:
            return None
        if not supported_float_contract(
            tensor_contract(graph, inputs[0]),
            float32_type=self.float32_type,
        ):
            return None
        if not supported_float_contract(
            tensor_contract(graph, outputs[0]),
            float32_type=self.float32_type,
        ):
            return None
        return operator, (inputs[0], inputs[1]), bool(options.keepDims)

    def _axes(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        tensor_index: int,
        rank: int,
    ) -> tuple[tuple[int, ...], int, str] | None:
        """Decode and normalize one integer reduction-axis constant."""

        pair = decode_integer_constant(
            self.codec,
            document,
            graph,
            tensor_index,
        )
        if pair is None:
            return None
        raw_axes, contract = pair
        if contract.tensor_type not in self.integer_axis_types:
            return None
        normalized = normalize_axes(raw_axes, rank)
        if normalized is None or not normalized:
            return None
        value = self.codec.decode_tensor(
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=tensor_index,
        )
        return normalized, contract.tensor_type, value.data.dtype.str

    def _plan(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        *,
        anchor_operator_index: int,
        replacement_input: int,
        axes: Sequence[int],
        axes_tensor_type: int,
        axes_dtype: str,
        supporting_operator_indices: Sequence[int],
        tensor_indices: Sequence[int],
        anchor_operator: Any,
    ) -> ReductionRewritePlan:
        """Capture a reduction rewrite after all shape checks succeeded."""

        plan = ReductionRewritePlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=anchor_operator_index,
            tensor_indices=tensor_indices,
            supporting_operators=capture_supporting_operators(
                document,
                subgraph_index=graph.subgraph_index,
                operator_indices=supporting_operator_indices,
            ),
            replacement_input=int(replacement_input),
            axes=tuple(int(axis) for axis in axes),
            axes_tensor_type=int(axes_tensor_type),
            axes_dtype=str(axes_dtype),
            replacement_version=operator_version(
                document.model,
                anchor_operator,
            ),
            replacement_options_type=int(
                getattr(anchor_operator, "builtinOptionsType", 0) or 0
            ),
            replacement_options=clone_object(
                getattr(anchor_operator, "builtinOptions", None)
            ),
        )
        return cast(ReductionRewritePlan, plan)


class _MergeConsecutiveMeanRule(_ReductionRuleBase):
    """Merge two consecutive MEAN operators into one reduction."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> ReductionRewritePlan | None:
        """Map both reduction axis sets back to the first input rank."""

        del context
        if not self.policy.floating_point_policy.allows_reassociation:
            return None
        if not operator_is_live(graph, operator_index):
            return None
        current = self._mean(document, graph, operator_index)
        if current is None:
            return None
        anchor, (intermediate, current_axes_tensor), keep_dims = current
        previous_index = graph.producer(intermediate)
        if previous_index is None:
            return None
        previous = self._mean(document, graph, previous_index)
        if previous is None:
            return None
        (
            previous_operator,
            (
                source_tensor,
                previous_axes_tensor,
            ),
            previous_keep_dims,
        ) = previous
        if previous_keep_dims != keep_dims:
            return None
        source_contract = tensor_contract(graph, source_tensor)
        intermediate_contract = tensor_contract(graph, intermediate)
        output_tensor = tuple(as_indices(getattr(anchor, "outputs", None)))[0]
        output_contract = tensor_contract(graph, output_tensor)
        first = self._axes(
            document,
            graph,
            previous_axes_tensor,
            source_contract.rank,
        )
        if first is None:
            return None
        first_axes, _first_type, _first_dtype = first
        expected_intermediate = _reduced_shape(
            source_contract.shape,
            first_axes,
            keep_dims=keep_dims,
        )
        if expected_intermediate != intermediate_contract.shape:
            return None
        second = self._axes(
            document,
            graph,
            current_axes_tensor,
            intermediate_contract.rank,
        )
        if second is None:
            return None
        second_axes, axes_type, axes_dtype = second
        if keep_dims:
            mapped_second = second_axes
        else:
            remaining = tuple(
                axis
                for axis in range(source_contract.rank)
                if axis not in set(first_axes)
            )
            try:
                mapped_second = tuple(remaining[axis] for axis in second_axes)
            except IndexError:
                return None
        merged = tuple(sorted(set(first_axes) | set(mapped_second)))
        if not merged or len(merged) > self.policy.maximum_merged_axes:
            return None
        expected_output = _reduced_shape(
            source_contract.shape,
            merged,
            keep_dims=keep_dims,
        )
        if expected_output != output_contract.shape:
            return None
        return self._plan(
            document,
            graph,
            anchor_operator_index=operator_index,
            replacement_input=source_tensor,
            axes=merged,
            axes_tensor_type=axes_type,
            axes_dtype=axes_dtype,
            supporting_operator_indices=(previous_index,),
            tensor_indices=(
                source_tensor,
                previous_axes_tensor,
                intermediate,
                current_axes_tensor,
                output_tensor,
            ),
            anchor_operator=anchor,
        )


class _SinkTransposeThroughMeanRule(_ReductionRuleBase):
    """Remove a transpose whose surviving axes are already in source order."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> ReductionRewritePlan | None:
        """Remap MEAN axes through a static permutation when output order agrees."""

        del context
        if not self.policy.floating_point_policy.allows_reassociation:
            return None
        if not operator_is_live(graph, operator_index):
            return None
        current = self._mean(document, graph, operator_index)
        if current is None:
            return None
        anchor, (transposed_tensor, axes_tensor), keep_dims = current
        if keep_dims:
            return None
        transpose_match = producer_matching(
            document,
            graph,
            transposed_tensor,
            builtin_code=self.codes["TRANSPOSE"],
            input_count=2,
        )
        if transpose_match is None:
            return None
        transpose_index, transpose = transpose_match
        transpose_inputs = tuple(as_indices(getattr(transpose, "inputs", None)))
        source_tensor, permutation_tensor = transpose_inputs
        source_contract = tensor_contract(graph, source_tensor)
        transposed_contract = tensor_contract(graph, transposed_tensor)
        output_tensor = tuple(as_indices(getattr(anchor, "outputs", None)))[0]
        output_contract = tensor_contract(graph, output_tensor)
        permutation_pair = decode_integer_constant(
            self.codec,
            document,
            graph,
            permutation_tensor,
        )
        if permutation_pair is None:
            return None
        permutation, _permutation_contract = permutation_pair
        if not output_shape_matches_transpose(
            source_contract.shape,
            transposed_contract.shape,
            permutation,
        ):
            return None
        axes_pair = self._axes(
            document,
            graph,
            axes_tensor,
            transposed_contract.rank,
        )
        if axes_pair is None:
            return None
        reduced_axes, axes_type, axes_dtype = axes_pair
        reduced_set = set(reduced_axes)
        surviving_source_axes = tuple(
            permutation[axis]
            for axis in range(len(permutation))
            if axis not in reduced_set
        )
        if surviving_source_axes != tuple(sorted(surviving_source_axes)):
            return None
        source_axes = tuple(sorted(permutation[axis] for axis in reduced_axes))
        if len(source_axes) > self.policy.maximum_merged_axes:
            return None
        if (
            _reduced_shape(
                source_contract.shape,
                source_axes,
                keep_dims=False,
            )
            != output_contract.shape
        ):
            return None
        return self._plan(
            document,
            graph,
            anchor_operator_index=operator_index,
            replacement_input=source_tensor,
            axes=source_axes,
            axes_tensor_type=axes_type,
            axes_dtype=axes_dtype,
            supporting_operator_indices=(transpose_index,),
            tensor_indices=(
                source_tensor,
                permutation_tensor,
                transposed_tensor,
                axes_tensor,
                output_tensor,
            ),
            anchor_operator=anchor,
        )


class SimplifyReductionOpsPass(CircleRulePass):
    """Simplify consecutive and transpose-prefixed MEAN reductions."""

    def __init__(
        self,
        *,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
        policy: ReductionSimplificationPolicy | None = None,
        maximum_rewrites: int = 10_000,
    ) -> None:
        """Create reduction rules with schema-independent enum overrides."""

        resolver = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            object_factory=object_factory,
        )
        shared: dict[str, Any] = {
            "codes": {
                "MEAN": resolver.builtin_code("MEAN"),
                "TRANSPOSE": resolver.builtin_code("TRANSPOSE"),
            },
            "options_types": {
                "ReducerOptions": resolver.builtin_options_type("ReducerOptions")
            },
            "float32_type": resolver.tensor_type("FLOAT32"),
            "integer_axis_types": (
                resolver.tensor_type("INT32"),
                resolver.tensor_type("INT64"),
            ),
            "codec": codec or TensorValueCodec(),
            "object_factory": object_factory,
            "policy": policy or ReductionSimplificationPolicy(),
        }
        super().__init__(
            [
                _MergeConsecutiveMeanRule(**shared),
                _SinkTransposeThroughMeanRule(**shared),
            ],
            maximum_rewrites=maximum_rewrites,
        )


def _reduced_shape(
    shape: Sequence[int],
    axes: Sequence[int],
    *,
    keep_dims: bool,
) -> tuple[int, ...]:
    """Return the static shape produced by one reduction."""

    reduced = set(int(axis) for axis in axes)
    if keep_dims:
        return tuple(
            1 if axis in reduced else int(dimension)
            for axis, dimension in enumerate(shape)
        )
    return tuple(
        int(dimension) for axis, dimension in enumerate(shape) if axis not in reduced
    )
