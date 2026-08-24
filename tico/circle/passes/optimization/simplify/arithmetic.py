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
    decode_float32_constant,
    has_no_fused_activation,
    operator_is_live,
    producer_matching,
    supported_float_contract,
    SupportingOperatorsPlan,
    tensor_has_single_consumer,
    tensor_name,
)
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    decode_constant_value,
    operator_builtin_code,
    operator_is_plain,
    operator_version,
    OptimizationSchemaResolver,
    tensor_contract,
    tensor_is_signature_bound,
)
from tico.circle.passes.optimization.policy import FloatingPointRewritePolicy
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
)
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True)
class ArithmeticCanonicalizationPolicy:
    """Control numerical assumptions used by arithmetic canonicalization."""

    floating_point_policy: FloatingPointRewritePolicy = (
        FloatingPointRewritePolicy.ALLOW_REASSOCIATION
    )
    require_finite_constants: bool = True
    maximum_constant_bytes: int = 1024

    def __post_init__(self) -> None:
        """Reject a byte limit that cannot hold one FLOAT32 scalar."""

        if self.maximum_constant_bytes < np.dtype(np.float32).itemsize:
            raise ValueError(
                "maximum_constant_bytes must hold at least one FLOAT32 value."
            )


@dataclass(frozen=True, kw_only=True)
class ArithmeticRewritePlan(SupportingOperatorsPlan):
    """Describe one anchor replacement and an optional producer replacement."""

    replacement_builtin_code: int
    replacement_inputs: tuple[int, ...]
    replacement_version: int
    replacement_options_type: int
    replacement_options: Any
    constant_value: float | None = None
    constant_shape: tuple[int, ...] = ()
    constant_tensor_type: int | None = None
    constant_dtype: str | None = None
    constant_input_position: int | None = None
    producer_operator_index: int | None = None
    producer_builtin_code: int | None = None
    producer_inputs: tuple[int, ...] = ()
    producer_version: int = 1


class _ArithmeticRuleBase(CircleRewriteRule[ArithmeticRewritePlan]):
    """Provide common matching and transactional arithmetic replacement."""

    def __init__(
        self,
        *,
        codes: Mapping[str, int],
        options_types: Mapping[str, int],
        float32_type: int,
        activation_none: int,
        codec: TensorValueCodec,
        object_factory: ObjectFactory | None,
        policy: ArithmeticCanonicalizationPolicy,
    ) -> None:
        """Store immutable services used by arithmetic rules."""

        self.codes = dict(codes)
        self.options_types = dict(options_types)
        self.float32_type = int(float32_type)
        self.activation_none = int(activation_none)
        self.codec = codec
        self.object_factory = object_factory
        self.policy = policy

    def apply(
        self,
        document: CircleDocument,
        plan: ArithmeticRewritePlan,
        context: CirclePassContext,
    ) -> RewriteApplication:
        """Apply an arithmetic replacement without deleting dead support nodes."""

        del context
        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=plan.subgraph_index,
        )
        subgraph = document.subgraph(plan.subgraph_index)
        operators = as_list(getattr(subgraph, "operators", None))
        original_anchor = operators[plan.anchor_operator_index]
        original_producer = (
            None
            if plan.producer_operator_index is None
            else operators[plan.producer_operator_index]
        )
        builder = CircleBuilder(
            document,
            subgraph_index=plan.subgraph_index,
            codec=self.codec,
            object_factory=self.object_factory,
        )
        try:
            replacement_inputs = list(plan.replacement_inputs)
            if plan.constant_value is not None:
                if (
                    plan.constant_tensor_type is None
                    or plan.constant_dtype is None
                    or plan.constant_input_position is None
                ):
                    raise ValueError(
                        "Scalar rewrites require a complete constant description."
                    )
                dtype = np.dtype(plan.constant_dtype)
                data = np.full(
                    plan.constant_shape,
                    plan.constant_value,
                    dtype=dtype,
                )
                value = TensorValue.from_values(
                    plan.constant_tensor_type,
                    data,
                    dtype=dtype,
                )
                constant_tensor = builder.add_constant(
                    tensor_name(
                        document.graph(plan.subgraph_index),
                        plan.anchor.inputs[
                            min(
                                plan.constant_input_position,
                                len(plan.anchor.inputs) - 1,
                            )
                        ],
                        "canonical_scalar",
                    ),
                    value,
                )
                replacement_inputs[plan.constant_input_position] = constant_tensor

            if plan.producer_operator_index is not None:
                if plan.producer_builtin_code is None:
                    raise ValueError("Producer replacement requires a builtin code.")
                assert original_producer is not None
                producer_replacement = builder.make_operator(
                    plan.producer_builtin_code,
                    inputs=plan.producer_inputs,
                    outputs=original_producer.outputs,
                    version=plan.producer_version,
                )
                builder.replace_operator(
                    plan.producer_operator_index,
                    producer_replacement,
                )

            replacement = builder.make_operator(
                plan.replacement_builtin_code,
                inputs=replacement_inputs,
                outputs=plan.anchor.outputs,
                version=plan.replacement_version,
                builtin_options_type=plan.replacement_options_type,
                builtin_options=clone_object(plan.replacement_options),
            )
            builder.replace_operator(plan.anchor_operator_index, replacement)
        except Exception:
            mutable_operators = as_list(
                getattr(document.subgraph(plan.subgraph_index), "operators", None)
            )
            if plan.anchor_operator_index < len(mutable_operators):
                mutable_operators[plan.anchor_operator_index] = original_anchor
            if (
                plan.producer_operator_index is not None
                and plan.producer_operator_index < len(mutable_operators)
            ):
                mutable_operators[plan.producer_operator_index] = original_producer
            document.subgraph(plan.subgraph_index).operators = mutable_operators
            checkpoint.rollback(document)
            raise
        return RewriteApplication(
            changes=2 if plan.producer_operator_index is not None else 1
        )

    def _binary_options(self, table_name: str, activation: int) -> Any:
        """Create one binary options table with a fused activation value."""

        options = (
            self.object_factory(table_name)
            if self.object_factory is not None
            else OptimizationSchemaResolver().create(table_name)
        )
        options.fusedActivationFunction = int(activation)
        if hasattr(options, "potScaleInt16"):
            options.potScaleInt16 = False
        return options

    def _plan(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        *,
        anchor_operator_index: int,
        replacement_builtin_code: int,
        replacement_inputs: Sequence[int],
        replacement_options_type: int,
        replacement_options: Any,
        supporting_operator_indices: Sequence[int],
        tensor_indices: Sequence[int],
        constant_value: float | None = None,
        constant_shape: Sequence[int] = (),
        constant_tensor_type: int | None = None,
        constant_dtype: str | None = None,
        constant_input_position: int | None = None,
        producer_operator_index: int | None = None,
        producer_builtin_code: int | None = None,
        producer_inputs: Sequence[int] = (),
        producer_version: int = 1,
    ) -> ArithmeticRewritePlan:
        """Capture a complete arithmetic rewrite plan."""

        anchor = as_list(graph.subgraph.operators)[anchor_operator_index]
        plan = ArithmeticRewritePlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=anchor_operator_index,
            tensor_indices=tensor_indices,
            supporting_operators=capture_supporting_operators(
                document,
                subgraph_index=graph.subgraph_index,
                operator_indices=supporting_operator_indices,
            ),
            replacement_builtin_code=int(replacement_builtin_code),
            replacement_inputs=tuple(int(index) for index in replacement_inputs),
            replacement_version=operator_version(document.model, anchor),
            replacement_options_type=int(replacement_options_type),
            replacement_options=clone_object(replacement_options),
            constant_value=(None if constant_value is None else float(constant_value)),
            constant_shape=tuple(int(dimension) for dimension in constant_shape),
            constant_tensor_type=(
                None if constant_tensor_type is None else int(constant_tensor_type)
            ),
            constant_dtype=constant_dtype,
            constant_input_position=constant_input_position,
            producer_operator_index=producer_operator_index,
            producer_builtin_code=producer_builtin_code,
            producer_inputs=tuple(int(index) for index in producer_inputs),
            producer_version=int(producer_version),
        )
        return cast(ArithmeticRewritePlan, plan)


class _CombineMulDivRule(_ArithmeticRuleBase):
    """Combine scalar multiplication and division into one binary operator."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> ArithmeticRewritePlan | None:
        """Match (x * a) / b or c / (x * a) with scalar FLOAT32 constants."""

        del context
        if not self.policy.floating_point_policy.allows_reassociation:
            return None
        if not operator_is_live(graph, operator_index):
            return None
        operators = as_list(graph.subgraph.operators)
        anchor = operators[operator_index]
        if not _binary_operator(
            document,
            anchor,
            code=self.codes["DIV"],
            options_type=self.options_types["DivOptions"],
            activation_none=self.activation_none,
            require_no_activation=False,
        ):
            return None
        inputs = tuple(as_indices(getattr(anchor, "inputs", None)))
        outputs = tuple(as_indices(getattr(anchor, "outputs", None)))
        if len(inputs) != 2 or len(outputs) != 1:
            return None
        numerator_constant = _scalar_info(
            self.codec,
            document,
            graph,
            inputs[0],
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
            maximum_bytes=self.policy.maximum_constant_bytes,
        )
        denominator_constant = _scalar_info(
            self.codec,
            document,
            graph,
            inputs[1],
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
            maximum_bytes=self.policy.maximum_constant_bytes,
        )
        if denominator_constant is not None:
            match = producer_matching(
                document,
                graph,
                inputs[0],
                builtin_code=self.codes["MUL"],
                input_count=2,
            )
            if match is None:
                return None
            mul_index, mul = match
            if not _binary_operator(
                document,
                mul,
                code=self.codes["MUL"],
                options_type=self.options_types["MulOptions"],
                activation_none=self.activation_none,
            ):
                return None
            parts = _mul_dynamic_scalar(
                self.codec,
                document,
                graph,
                mul,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
                maximum_bytes=self.policy.maximum_constant_bytes,
            )
            if parts is None:
                return None
            source_tensor, mul_tensor, mul_value = parts
            div_value, div_shape, div_type, div_dtype = denominator_constant
            if div_value == 0.0:
                return None
            new_value = np.float32(mul_value) / np.float32(div_value)
            if self.policy.require_finite_constants and not np.isfinite(new_value):
                return None
            if not self._contracts_match(graph, source_tensor, outputs[0]):
                return None
            options = self._binary_options(
                "MulOptions",
                int(anchor.builtinOptions.fusedActivationFunction),
            )
            return self._plan(
                document,
                graph,
                anchor_operator_index=operator_index,
                replacement_builtin_code=self.codes["MUL"],
                replacement_inputs=(source_tensor, -1),
                replacement_options_type=self.options_types["MulOptions"],
                replacement_options=options,
                supporting_operator_indices=(mul_index,),
                tensor_indices=(
                    source_tensor,
                    mul_tensor,
                    inputs[1],
                    outputs[0],
                ),
                constant_value=float(new_value),
                constant_shape=div_shape,
                constant_tensor_type=div_type,
                constant_dtype=div_dtype,
                constant_input_position=1,
            )
        if numerator_constant is None:
            return None
        match = producer_matching(
            document,
            graph,
            inputs[1],
            builtin_code=self.codes["MUL"],
            input_count=2,
        )
        if match is None:
            return None
        mul_index, mul = match
        if not _binary_operator(
            document,
            mul,
            code=self.codes["MUL"],
            options_type=self.options_types["MulOptions"],
            activation_none=self.activation_none,
        ):
            return None
        parts = _mul_dynamic_scalar(
            self.codec,
            document,
            graph,
            mul,
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
            maximum_bytes=self.policy.maximum_constant_bytes,
        )
        if parts is None:
            return None
        source_tensor, mul_tensor, mul_value = parts
        if mul_value == 0.0:
            return None
        numerator, shape, tensor_type, dtype = numerator_constant
        new_value = np.float32(numerator) / np.float32(mul_value)
        if self.policy.require_finite_constants and not np.isfinite(new_value):
            return None
        if not self._contracts_match(graph, source_tensor, outputs[0]):
            return None
        options = clone_object(anchor.builtinOptions)
        return self._plan(
            document,
            graph,
            anchor_operator_index=operator_index,
            replacement_builtin_code=self.codes["DIV"],
            replacement_inputs=(-1, source_tensor),
            replacement_options_type=self.options_types["DivOptions"],
            replacement_options=options,
            supporting_operator_indices=(mul_index,),
            tensor_indices=(
                inputs[0],
                source_tensor,
                mul_tensor,
                outputs[0],
            ),
            constant_value=float(new_value),
            constant_shape=shape,
            constant_tensor_type=tensor_type,
            constant_dtype=dtype,
            constant_input_position=0,
        )

    def _contracts_match(
        self,
        graph: CircleGraph,
        source_tensor: int,
        output_tensor: int,
    ) -> bool:
        """Require static unquantized FLOAT32 input and output contracts."""

        source = tensor_contract(graph, source_tensor)
        output = tensor_contract(graph, output_tensor)
        return source == output and supported_float_contract(
            source,
            float32_type=self.float32_type,
        )


class _SqrtDivToRsqrtMulRule(_ArithmeticRuleBase):
    """Replace x / SQRT(y) with x * RSQRT(y) using existing tensor topology."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> ArithmeticRewritePlan | None:
        """Match a private SQRT denominator and preserve graph boundary semantics."""

        del context
        if not self.policy.floating_point_policy.allows_reassociation:
            return None
        if not operator_is_live(graph, operator_index):
            return None
        anchor = as_list(graph.subgraph.operators)[operator_index]
        if not _binary_operator(
            document,
            anchor,
            code=self.codes["DIV"],
            options_type=self.options_types["DivOptions"],
            activation_none=self.activation_none,
            require_no_activation=False,
        ):
            return None
        inputs = tuple(as_indices(getattr(anchor, "inputs", None)))
        outputs = tuple(as_indices(getattr(anchor, "outputs", None)))
        if len(inputs) != 2 or len(outputs) != 1:
            return None
        if (
            decode_constant_value(
                self.codec,
                document.model,
                subgraph_index=graph.subgraph_index,
                tensor_index=inputs[0],
            )
            is not None
        ):
            return None
        sqrt_match = producer_matching(
            document,
            graph,
            inputs[1],
            builtin_code=self.codes["SQRT"],
            input_count=1,
        )
        if sqrt_match is None:
            return None
        sqrt_index, sqrt = sqrt_match
        if not tensor_has_single_consumer(graph, inputs[1], operator_index):
            return None
        if inputs[1] in set(graph.outputs) or tensor_is_signature_bound(
            document.model,
            subgraph_index=graph.subgraph_index,
            tensor_index=inputs[1],
        ):
            return None
        sqrt_inputs = tuple(as_indices(getattr(sqrt, "inputs", None)))
        if len(sqrt_inputs) != 1:
            return None
        numerator_contract = tensor_contract(graph, inputs[0])
        denominator_contract = tensor_contract(graph, sqrt_inputs[0])
        output_contract = tensor_contract(graph, outputs[0])
        if not all(
            supported_float_contract(
                contract,
                float32_type=self.float32_type,
            )
            for contract in (
                numerator_contract,
                denominator_contract,
                output_contract,
            )
        ):
            return None
        try:
            broadcast_shape = np.broadcast_shapes(
                numerator_contract.shape,
                denominator_contract.shape,
            )
        except ValueError:
            return None
        if tuple(broadcast_shape) != output_contract.shape:
            return None
        options = self._binary_options(
            "MulOptions",
            int(anchor.builtinOptions.fusedActivationFunction),
        )
        return self._plan(
            document,
            graph,
            anchor_operator_index=operator_index,
            replacement_builtin_code=self.codes["MUL"],
            replacement_inputs=(inputs[0], inputs[1]),
            replacement_options_type=self.options_types["MulOptions"],
            replacement_options=options,
            supporting_operator_indices=(sqrt_index,),
            tensor_indices=(
                inputs[0],
                sqrt_inputs[0],
                inputs[1],
                outputs[0],
            ),
            producer_operator_index=sqrt_index,
            producer_builtin_code=self.codes["RSQRT"],
            producer_inputs=sqrt_inputs,
            producer_version=operator_version(document.model, sqrt),
        )


class SimplifyArithmeticPass(CircleRulePass):
    """Canonicalize scalar MUL/DIV chains and SQRT denominators."""

    def __init__(
        self,
        *,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        activation_none: int | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
        policy: ArithmeticCanonicalizationPolicy | None = None,
        maximum_rewrites: int = 10_000,
    ) -> None:
        """Create arithmetic rules with schema-independent enum overrides."""

        resolver = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            activation_none=activation_none,
            object_factory=object_factory,
        )
        shared: dict[str, Any] = {
            "codes": {
                name: resolver.builtin_code(name)
                for name in ("DIV", "MUL", "RSQRT", "SQRT")
            },
            "options_types": {
                name: resolver.builtin_options_type(name)
                for name in ("DivOptions", "MulOptions")
            },
            "float32_type": resolver.tensor_type("FLOAT32"),
            "activation_none": resolver.activation_none,
            "codec": codec or TensorValueCodec(),
            "object_factory": object_factory,
            "policy": policy or ArithmeticCanonicalizationPolicy(),
        }
        super().__init__(
            [
                _CombineMulDivRule(**shared),
                _SqrtDivToRsqrtMulRule(**shared),
            ],
            maximum_rewrites=maximum_rewrites,
        )


def _binary_operator(
    document: CircleDocument,
    operator: Any,
    *,
    code: int,
    options_type: int,
    activation_none: int,
    require_no_activation: bool = True,
) -> bool:
    """Return whether an operator is a plain binary op without activation."""

    return (
        operator_builtin_code(document.model, operator) == int(code)
        and operator_is_plain(operator)
        and len(as_indices(getattr(operator, "inputs", None))) == 2
        and len(as_indices(getattr(operator, "outputs", None))) == 1
        and int(getattr(operator, "builtinOptionsType", 0) or 0) == int(options_type)
        and (
            not require_no_activation
            or has_no_fused_activation(operator, activation_none)
        )
    )


def _scalar_info(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph: CircleGraph,
    tensor_index: int,
    *,
    float32_type: int,
    require_finite: bool,
    maximum_bytes: int,
) -> tuple[float, tuple[int, ...], int, str] | None:
    """Return scalar value and storage metadata for one FLOAT32 constant."""

    pair = decode_float32_constant(
        codec,
        document,
        graph,
        tensor_index,
        float32_type=float32_type,
        require_finite=require_finite,
    )
    if pair is None:
        return None
    value, contract = pair
    if value.data.size != 1 or value.nbytes > int(maximum_bytes):
        return None
    return (
        float(value.data.reshape(-1)[0]),
        value.shape,
        contract.tensor_type,
        value.data.dtype.str,
    )


def _mul_dynamic_scalar(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph: CircleGraph,
    operator: Any,
    *,
    float32_type: int,
    require_finite: bool,
    maximum_bytes: int,
) -> tuple[int, int, float] | None:
    """Split MUL into one dynamic input and one scalar FLOAT32 constant."""

    inputs = tuple(as_indices(getattr(operator, "inputs", None)))
    if len(inputs) != 2:
        return None
    for dynamic_tensor, constant_tensor in (inputs, inputs[::-1]):
        info = _scalar_info(
            codec,
            document,
            graph,
            constant_tensor,
            float32_type=float32_type,
            require_finite=require_finite,
            maximum_bytes=maximum_bytes,
        )
        if info is None:
            continue
        if (
            decode_constant_value(
                codec,
                document.model,
                subgraph_index=graph.subgraph_index,
                tensor_index=dynamic_tensor,
            )
            is not None
        ):
            continue
        return dynamic_tensor, constant_tensor, info[0]
    return None
