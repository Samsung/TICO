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
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from tico.circle._object import ObjectFactory
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list, CircleGraph
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    constant_represents_real_zero,
    decode_constant_value,
    decode_integer_vector,
    normalize_axis,
    operator_builtin_code,
    operator_is_plain,
    OptimizationSchemaResolver,
    output_shape_matches_transpose,
    strided_slice_view_shape,
    tensor_contract,
    transpose_is_view_only,
    view_contracts_compatible,
)
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
)
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True, kw_only=True)
class _ReshapeCanonicalizationPlan(RewritePlan):
    """Carry a data input and static target shape into a RESHAPE replacement."""

    data_input: int
    output_shape: tuple[int, ...]
    output_name: str
    source_kind: str


@dataclass(frozen=True, kw_only=True)
class _PadCanonicalizationPlan(RewritePlan):
    """Carry PAD inputs into a PADV2-to-PAD replacement."""

    data_input: int
    paddings_input: int


@dataclass(frozen=True, kw_only=True)
class _SplitCanonicalizationPlan(RewritePlan):
    """Carry equal-size SPLIT inputs into a SPLIT_V-to-SPLIT replacement."""

    data_input: int
    axis_input: int
    num_splits: int


class _EquivalentRule(CircleRewriteRule[RewritePlan]):
    """Share schema, value, and operator-replacement services across rules."""

    def __init__(
        self,
        schema: OptimizationSchemaResolver,
        codec: TensorValueCodec,
    ) -> None:
        """Bind schema and constant decoding services."""

        self.schema = schema
        self.codec = codec
        self.reshape_code = schema.builtin_code("RESHAPE")
        self.reshape_options_type = schema.builtin_options_type("ReshapeOptions")
        self.int32_type = schema.tensor_type("INT32")

    def _replace_operator_atomically(
        self,
        document: CircleDocument,
        plan: RewritePlan,
        build_replacement: Callable[[CircleBuilder], Any],
    ) -> None:
        """Build and install one replacement with allocation rollback on failure."""

        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=plan.subgraph_index,
        )
        subgraph = document.subgraph(plan.subgraph_index)
        operators = as_list(getattr(subgraph, "operators", None))
        original_operator = operators[plan.anchor_operator_index]
        try:
            builder = CircleBuilder(
                document,
                subgraph_index=plan.subgraph_index,
                codec=self.codec,
                object_factory=self.schema.object_factory,
            )
            replacement = build_replacement(builder)
            builder.replace_operator(plan.anchor_operator_index, replacement)
        except Exception:
            checkpoint.rollback(document)
            restored_operators = as_list(
                getattr(document.subgraph(plan.subgraph_index), "operators", None)
            )
            restored_operators[plan.anchor_operator_index] = original_operator
            document.subgraph(plan.subgraph_index).operators = restored_operators
            raise

    def _replace_with_reshape(
        self,
        document: CircleDocument,
        plan: _ReshapeCanonicalizationPlan,
    ) -> RewriteApplication:
        """Replace the anchor with a static two-input RESHAPE operator."""

        def build_replacement(builder: CircleBuilder) -> Any:
            """Create the canonical RESHAPE and its static shape constant."""

            shape_value = TensorValue.from_values(
                self.int32_type,
                np.asarray(plan.output_shape, dtype=np.int32),
                dtype=np.int32,
            )
            shape_input = builder.add_constant(
                f"{plan.output_name}_shape",
                shape_value,
            )
            options = self.schema.create("ReshapeOptions")
            options.newShape = list(plan.output_shape)
            return builder.make_operator(
                self.reshape_code,
                inputs=(plan.data_input, shape_input),
                outputs=plan.anchor.outputs,
                version=1,
                builtin_options_type=self.reshape_options_type,
                builtin_options=options,
            )

        self._replace_operator_atomically(document, plan, build_replacement)
        return RewriteApplication(
            changes=1,
            diagnostics=(
                RewriteDiagnostic(
                    code="CANONICALIZED_TO_RESHAPE",
                    message=(
                        f"Replaced {plan.source_kind} with a "
                        "storage-preserving RESHAPE."
                    ),
                    object_path=(
                        f"subgraphs[{plan.subgraph_index}].operators"
                        f"[{plan.anchor_operator_index}]"
                    ),
                ),
            ),
        )


class _ExpandDimsToReshapeRule(_EquivalentRule):
    """Canonicalize static EXPAND_DIMS to RESHAPE."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind the EXPAND_DIMS source opcode."""

        super().__init__(schema, codec)
        self.source_code = schema.builtin_code("EXPAND_DIMS")

    def match(self, document, graph, operator_index, context):
        """Match static EXPAND_DIMS whose storage order and qparams are unchanged."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 2 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        source = tensor_contract(graph, inputs[0])
        output = tensor_contract(graph, outputs[0])
        if not view_contracts_compatible(source, output):
            return None
        axis_values = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
            expected_count=1,
        )
        if axis_values is None or not output.shape:
            return None
        axis = normalize_axis(axis_values[0], source.rank, allow_end=True)
        if axis is None:
            return None
        if output.shape != source.shape[:axis] + (1,) + source.shape[axis:]:
            return None
        return _ReshapeCanonicalizationPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            data_input=inputs[0],
            output_shape=output.shape,
            output_name=_tensor_name(graph, outputs[0]),
            source_kind="EXPAND_DIMS",
        )

    def apply(self, document, plan, context):
        """Replace one matched EXPAND_DIMS with RESHAPE."""

        del context
        return self._replace_with_reshape(document, plan)


class _SingleInputPackToReshapeRule(_EquivalentRule):
    """Canonicalize one-input PACK to RESHAPE."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind the PACK source opcode."""

        super().__init__(schema, codec)
        self.source_code = schema.builtin_code("PACK")

    def match(self, document, graph, operator_index, context):
        """Match a one-input PACK that only inserts a unit dimension."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 1 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        source = tensor_contract(graph, inputs[0])
        output = tensor_contract(graph, outputs[0])
        if not view_contracts_compatible(source, output) or not output.shape:
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        values_count = int(getattr(options, "valuesCount", 0) or 0)
        axis = normalize_axis(
            int(getattr(options, "axis", 0) or 0),
            source.rank,
            allow_end=True,
        )
        if values_count != 1 or axis is None:
            return None
        if output.shape != source.shape[:axis] + (1,) + source.shape[axis:]:
            return None
        return _ReshapeCanonicalizationPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            data_input=inputs[0],
            output_shape=output.shape,
            output_name=_tensor_name(graph, outputs[0]),
            source_kind="PACK",
        )

    def apply(self, document, plan, context):
        """Replace one matched PACK with RESHAPE."""

        del context
        return self._replace_with_reshape(document, plan)


class _SqueezeToReshapeRule(_EquivalentRule):
    """Canonicalize static SQUEEZE to RESHAPE."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind the SQUEEZE source opcode."""

        super().__init__(schema, codec)
        self.source_code = schema.builtin_code("SQUEEZE")

    def match(self, document, graph, operator_index, context):
        """Match SQUEEZE when static dimensions determine the exact output shape."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 1 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        source = tensor_contract(graph, inputs[0])
        output = tensor_contract(graph, outputs[0])
        if not view_contracts_compatible(source, output) or not output.shape:
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        raw_axes = tuple(
            int(axis) for axis in as_list(getattr(options, "squeezeDims", None))
        )
        if raw_axes:
            axes: set[int] = set()
            for raw_axis in raw_axes:
                axis = normalize_axis(raw_axis, source.rank)
                if axis is None or source.shape[axis] != 1:
                    return None
                axes.add(axis)
        else:
            axes = {
                axis for axis, dimension in enumerate(source.shape) if dimension == 1
            }
        expected = tuple(
            dimension for axis, dimension in enumerate(source.shape) if axis not in axes
        )
        if output.shape != expected:
            return None
        return _ReshapeCanonicalizationPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            data_input=inputs[0],
            output_shape=output.shape,
            output_name=_tensor_name(graph, outputs[0]),
            source_kind="SQUEEZE",
        )

    def apply(self, document, plan, context):
        """Replace one matched SQUEEZE with RESHAPE."""

        del context
        return self._replace_with_reshape(document, plan)


class _ViewOnlyStridedSliceToReshapeRule(_EquivalentRule):
    """Canonicalize rank-only STRIDED_SLICE patterns to RESHAPE."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind the STRIDED_SLICE source opcode."""

        super().__init__(schema, codec)
        self.source_code = schema.builtin_code("STRIDED_SLICE")

    def match(self, document, graph, operator_index, context):
        """Match slices that retain every element in original storage order."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 4 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        source = tensor_contract(graph, inputs[0])
        output = tensor_contract(graph, outputs[0])
        if not view_contracts_compatible(source, output) or not output.shape:
            return None
        begin = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
        )
        end = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[2],
        )
        strides = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[3],
        )
        if begin is None or end is None or strides is None:
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        expected = strided_slice_view_shape(
            source.shape,
            begin,
            end,
            strides,
            begin_mask=int(getattr(options, "beginMask", 0) or 0),
            end_mask=int(getattr(options, "endMask", 0) or 0),
            ellipsis_mask=int(getattr(options, "ellipsisMask", 0) or 0),
            new_axis_mask=int(getattr(options, "newAxisMask", 0) or 0),
            shrink_axis_mask=int(getattr(options, "shrinkAxisMask", 0) or 0),
        )
        if expected is None or output.shape != expected:
            return None
        # Identity STRIDED_SLICE belongs to EliminateIdentityOpsPass. Keeping
        # identity elimination out of canonicalization makes rule ownership
        # independent of O1 pass ordering.
        if output.shape == source.shape:
            return None
        return _ReshapeCanonicalizationPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            data_input=inputs[0],
            output_shape=output.shape,
            output_name=_tensor_name(graph, outputs[0]),
            source_kind="STRIDED_SLICE",
        )

    def apply(self, document, plan, context):
        """Replace one matched STRIDED_SLICE with RESHAPE."""

        del context
        return self._replace_with_reshape(document, plan)


class _ViewOnlyTransposeToReshapeRule(_EquivalentRule):
    """Canonicalize transposes that only relocate unit dimensions."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind the TRANSPOSE source opcode."""

        super().__init__(schema, codec)
        self.source_code = schema.builtin_code("TRANSPOSE")

    def match(self, document, graph, operator_index, context):
        """Match a transpose whose non-unit axes retain their original order."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 2 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        source = tensor_contract(graph, inputs[0])
        output = tensor_contract(graph, outputs[0])
        if not view_contracts_compatible(source, output) or not output.shape:
            return None
        permutation = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
            expected_count=source.rank,
        )
        if permutation is None:
            return None
        # Identity TRANSPOSE belongs to SimplifyViewOpsPass.
        if tuple(permutation) == tuple(range(source.rank)):
            return None
        if not output_shape_matches_transpose(
            source.shape,
            output.shape,
            permutation,
        ):
            return None
        if not transpose_is_view_only(source.shape, permutation):
            return None
        return _ReshapeCanonicalizationPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            data_input=inputs[0],
            output_shape=output.shape,
            output_name=_tensor_name(graph, outputs[0]),
            source_kind="TRANSPOSE",
        )

    def apply(self, document, plan, context):
        """Replace one matched TRANSPOSE with RESHAPE."""

        del context
        return self._replace_with_reshape(document, plan)


class _ZeroPadV2ToPadRule(_EquivalentRule):
    """Canonicalize zero-valued PADV2 to PAD."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind PADV2 and PAD schema identities."""

        super().__init__(schema, codec)
        self.source_code = schema.builtin_code("PADV2")
        self.target_code = schema.builtin_code("PAD")
        self.target_options_type = schema.builtin_options_type("PadOptions")

    def match(self, document, graph, operator_index, context):
        """Match PADV2 when its explicit value is exact real zero."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 3 or len(outputs) != 1 or not operator_is_plain(operator):
            return None
        source = tensor_contract(graph, inputs[0])
        output = tensor_contract(graph, outputs[0])
        if not view_contracts_compatible(source, output):
            return None
        paddings = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
            expected_count=source.rank * 2,
        )
        value = decode_constant_value(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[2],
        )
        if (
            paddings is None
            or value is None
            or value.element_count != 1
            or value.tensor_type != source.tensor_type
        ):
            return None
        if any(padding < 0 for padding in paddings):
            return None
        expected_shape = tuple(
            dimension + paddings[2 * axis] + paddings[2 * axis + 1]
            for axis, dimension in enumerate(source.shape)
        )
        if output.shape != expected_shape:
            return None
        if not constant_represents_real_zero(value, reference_contract=source):
            return None
        return _PadCanonicalizationPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            data_input=inputs[0],
            paddings_input=inputs[1],
        )

    def apply(self, document, plan, context):
        """Replace one matched PADV2 with PAD."""

        del context

        def build_replacement(builder: CircleBuilder) -> Any:
            """Create the canonical two-input PAD operator."""

            options = self.schema.create("PadOptions")
            return builder.make_operator(
                self.target_code,
                inputs=(plan.data_input, plan.paddings_input),
                outputs=plan.anchor.outputs,
                version=1,
                builtin_options_type=self.target_options_type,
                builtin_options=options,
            )

        self._replace_operator_atomically(document, plan, build_replacement)
        return RewriteApplication(
            changes=1,
            diagnostics=(
                RewriteDiagnostic(
                    code="PADV2_TO_PAD",
                    message="Replaced zero-valued PADV2 with PAD.",
                ),
            ),
        )


class _UniformSplitVToSplitRule(_EquivalentRule):
    """Canonicalize equal-size SPLIT_V to SPLIT."""

    def __init__(self, schema: OptimizationSchemaResolver, codec: TensorValueCodec):
        """Bind SPLIT_V and SPLIT schema identities."""

        super().__init__(schema, codec)
        self.source_code = schema.builtin_code("SPLIT_V")
        self.target_code = schema.builtin_code("SPLIT")
        self.target_options_type = schema.builtin_options_type("SplitOptions")

    def match(self, document, graph, operator_index, context):
        """Match static SPLIT_V with equal resolved split sizes."""

        del context
        operator = as_list(graph.subgraph.operators)[operator_index]
        if operator_builtin_code(document.model, operator) != self.source_code:
            return None
        inputs = as_indices(operator.inputs)
        outputs = as_indices(operator.outputs)
        if len(inputs) != 3 or not outputs or not operator_is_plain(operator):
            return None
        # A one-output SPLIT_V is an identity and is removed directly by
        # EliminateIdentityOpsPass instead of being canonicalized to SPLIT.
        if len(outputs) == 1:
            return None
        source = tensor_contract(graph, inputs[0])
        output_contracts = tuple(tensor_contract(graph, index) for index in outputs)
        if not all(
            view_contracts_compatible(source, output) for output in output_contracts
        ):
            return None
        axis_values = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[2],
            expected_count=1,
        )
        split_sizes = decode_integer_vector(
            self.codec,
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
            expected_count=len(outputs),
        )
        if axis_values is None or split_sizes is None:
            return None
        axis = normalize_axis(axis_values[0], source.rank)
        if axis is None:
            return None
        resolved = _resolve_split_sizes(split_sizes, source.shape[axis])
        if resolved is None or len(set(resolved)) != 1:
            return None
        expected_output_shape = list(source.shape)
        expected_output_shape[axis] = resolved[0]
        if any(
            output.shape != tuple(expected_output_shape) for output in output_contracts
        ):
            return None
        options = getattr(operator, "builtinOptions", None)
        if options is None:
            return None
        num_splits = int(getattr(options, "numSplits", 0) or 0)
        if num_splits != len(outputs):
            return None
        return _SplitCanonicalizationPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=(*inputs, *outputs),
            data_input=inputs[0],
            axis_input=inputs[2],
            num_splits=len(outputs),
        )

    def apply(self, document, plan, context):
        """Replace one matched SPLIT_V with SPLIT."""

        del context

        def build_replacement(builder: CircleBuilder) -> Any:
            """Create the canonical equal-size SPLIT operator."""

            options = self.schema.create("SplitOptions")
            options.numSplits = plan.num_splits
            return builder.make_operator(
                self.target_code,
                inputs=(plan.axis_input, plan.data_input),
                outputs=plan.anchor.outputs,
                version=1,
                builtin_options_type=self.target_options_type,
                builtin_options=options,
            )

        self._replace_operator_atomically(document, plan, build_replacement)
        return RewriteApplication(
            changes=1,
            diagnostics=(
                RewriteDiagnostic(
                    code="SPLIT_V_TO_SPLIT",
                    message="Replaced equal-size SPLIT_V with SPLIT.",
                ),
            ),
        )


class CanonicalizeEquivalentOpsPass(CirclePass):
    """Replace equivalent Circle operators with a smaller canonical vocabulary."""

    def __init__(
        self,
        *,
        maximum_rewrites: int = 10_000,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create equivalent-op rules with schema or test enum mappings."""

        self.maximum_rewrites = int(maximum_rewrites)
        if self.maximum_rewrites <= 0:
            raise ValueError("maximum_rewrites must be positive.")
        self.codec = codec or TensorValueCodec()
        self.schema = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            object_factory=object_factory,
        )
        self.rules = (
            _ExpandDimsToReshapeRule(self.schema, self.codec),
            _SingleInputPackToReshapeRule(self.schema, self.codec),
            _ZeroPadV2ToPadRule(self.schema, self.codec),
            _UniformSplitVToSplitRule(self.schema, self.codec),
            _SqueezeToReshapeRule(self.schema, self.codec),
            _ViewOnlyStridedSliceToReshapeRule(self.schema, self.codec),
            _ViewOnlyTransposeToReshapeRule(self.schema, self.codec),
        )

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Canonicalize all supported equivalent operators to a fixed point."""

        return CircleRulePass(
            self.rules,
            maximum_rewrites=self.maximum_rewrites,
        ).run(document, context)


def _resolve_split_sizes(
    split_sizes: tuple[int, ...],
    input_size: int,
) -> tuple[int, ...] | None:
    """Resolve one optional inferred split size and validate the total."""

    if input_size < 0 or any(size < -1 for size in split_sizes):
        return None
    inferred = [index for index, size in enumerate(split_sizes) if size == -1]
    if len(inferred) > 1:
        return None
    resolved = list(split_sizes)
    known = sum(size for size in resolved if size >= 0)
    if inferred:
        inferred_size = input_size - known
        if inferred_size < 0:
            return None
        resolved[inferred[0]] = inferred_size
    if sum(resolved) != input_size:
        return None
    return tuple(resolved)


def _tensor_name(graph: CircleGraph, tensor_index: int) -> str:
    """Return a stable output-derived name for generated shape constants."""

    tensor = as_list(graph.subgraph.tensors)[tensor_index]
    name = getattr(tensor, "name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return str(name) or f"tensor_{tensor_index}"
