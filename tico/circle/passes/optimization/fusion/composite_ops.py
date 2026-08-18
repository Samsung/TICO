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
from typing import Any

import numpy as np

from tico.circle._object import clone_object, ObjectFactory
from tico.circle._schema import decode_text
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list, CircleGraph
from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.optimization._pattern_utils import (
    capture_supporting_operators,
    decode_float32_constant,
    decode_integer_constant,
    has_no_fused_activation,
    normalize_axes,
    operator_is_live,
    producer_matching,
    scalar_float32,
    supported_float_contract,
    SupportingOperatorsPlan,
    tensor_has_single_consumer,
)
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    operator_builtin_code,
    operator_is_plain,
    operator_version,
    OptimizationSchemaResolver,
    tensor_contract,
)
from tico.circle.passes.optimization.policy import FloatingPointRewritePolicy
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
)
from tico.circle.value import TensorValueCodec


@dataclass(frozen=True)
class CompositeFusionPolicy:
    """Control numerical and allocation preconditions for composite fusion."""

    floating_point_policy: FloatingPointRewritePolicy = (
        FloatingPointRewritePolicy.ALLOW_REASSOCIATION
    )
    require_finite_constants: bool = True
    enable_instance_norm: bool = True
    maximum_parameter_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        """Reject resource bounds that cannot admit any parameter tensor."""

        if self.maximum_parameter_bytes <= 0:
            raise ValueError("maximum_parameter_bytes must be positive.")


@dataclass(frozen=True, kw_only=True)
class CompositeFusionPlan(SupportingOperatorsPlan):
    """Describe one replacement operator for a recognized composite pattern."""

    replacement_builtin_code: int
    replacement_version: int
    replacement_inputs: tuple[int, ...]
    replacement_options_type: int = 0
    replacement_options: Any = None
    template_operator_index: int | None = None


class _CompositeRuleBase(CircleRewriteRule[CompositeFusionPlan]):
    """Provide schema resolution and transactional replacement helpers."""

    def __init__(
        self,
        *,
        codes: Mapping[str, int],
        options_types: Mapping[str, int],
        float32_type: int,
        activation_none: int,
        codec: TensorValueCodec,
        object_factory: ObjectFactory | None,
        policy: CompositeFusionPolicy,
    ) -> None:
        """Store immutable services shared by composite-pattern matchers."""

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
        plan: CompositeFusionPlan,
        context: CirclePassContext,
    ) -> RewriteApplication:
        """Replace only the anchor and leave unreachable support for external DCE."""

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
            if plan.template_operator_index is None:
                replacement = builder.make_operator(
                    plan.replacement_builtin_code,
                    inputs=plan.replacement_inputs,
                    outputs=plan.anchor.outputs,
                    version=plan.replacement_version,
                    builtin_options_type=plan.replacement_options_type,
                    builtin_options=clone_object(plan.replacement_options),
                )
            else:
                operators = as_list(document.subgraph(plan.subgraph_index).operators)
                replacement = clone_object(operators[plan.template_operator_index])
                replacement.inputs = list(plan.replacement_inputs)
                replacement.outputs = list(plan.anchor.outputs)
                if plan.replacement_options is not None:
                    replacement.builtinOptions = clone_object(plan.replacement_options)
                    replacement.builtinOptionsType = plan.replacement_options_type
            builder.replace_operator(plan.anchor_operator_index, replacement)
        except Exception:
            checkpoint.rollback(document)
            raise
        return RewriteApplication(changes=1)

    def _plan(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        *,
        anchor_operator_index: int,
        replacement_builtin_code: int,
        replacement_inputs: Sequence[int],
        supporting_operator_indices: Sequence[int] = (),
        tensor_indices: Sequence[int] = (),
        replacement_options_type: int = 0,
        replacement_options: Any = None,
        replacement_version: int = 1,
        template_operator_index: int | None = None,
    ) -> CompositeFusionPlan:
        """Capture a complete immutable plan for one composite replacement."""

        return CompositeFusionPlan.capture(
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
            replacement_version=int(replacement_version),
            replacement_inputs=tuple(int(index) for index in replacement_inputs),
            replacement_options_type=int(replacement_options_type),
            replacement_options=clone_object(replacement_options),
            template_operator_index=(
                None
                if template_operator_index is None
                else int(template_operator_index)
            ),
        )

    def _float_contract(self, graph: CircleGraph, tensor_index: int) -> bool:
        """Return whether one tensor is static dense unquantized FLOAT32."""

        return supported_float_contract(
            tensor_contract(graph, tensor_index),
            float32_type=self.float32_type,
        )

    def _algebraic_rewrites_enabled(self) -> bool:
        """Return whether the policy permits floating-point reassociation."""

        return self.policy.floating_point_policy.allows_reassociation


class _FuseActivationAttributeRule(_CompositeRuleBase):
    """Move a standalone RELU-family operator into its producer options."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> CompositeFusionPlan | None:
        """Match a single-consumer activation after a capable producer."""

        del context
        if not self._algebraic_rewrites_enabled():
            return None
        if not operator_is_live(graph, operator_index):
            return None
        operators = as_list(graph.subgraph.operators)
        anchor = operators[operator_index]
        activation_code = operator_builtin_code(document.model, anchor)
        target = _activation_target(
            activation_code,
            codes=self.codes,
        )
        if target is None or not operator_is_plain(anchor):
            return None
        inputs = tuple(as_indices(getattr(anchor, "inputs", None)))
        outputs = tuple(as_indices(getattr(anchor, "outputs", None)))
        if len(inputs) != 1 or len(outputs) != 1:
            return None
        source_tensor = inputs[0]
        producer_index = graph.producer(source_tensor)
        if producer_index is None:
            return None
        if not tensor_has_single_consumer(
            graph,
            source_tensor,
            operator_index,
        ):
            return None
        producer = operators[producer_index]
        producer_code = operator_builtin_code(document.model, producer)
        if producer_code in {
            self.codes.get("CONCATENATION", -1),
            self.codes.get("TRANSPOSE_CONV", -1),
        }:
            return None
        if not operator_is_plain(producer):
            return None
        producer_outputs = tuple(as_indices(getattr(producer, "outputs", None)))
        if producer_outputs != (source_tensor,):
            return None
        options = getattr(producer, "builtinOptions", None)
        if options is None or not hasattr(options, "fusedActivationFunction"):
            return None
        existing = int(
            getattr(options, "fusedActivationFunction", self.activation_none)
        )
        fused = _combine_activation(
            existing,
            target,
            none=self.activation_none,
            relu=self.codes["ACTIVATION_RELU"],
            relu6=self.codes["ACTIVATION_RELU6"],
            relu_n1_to_1=self.codes["ACTIVATION_RELU_N1_TO_1"],
        )
        if fused is None:
            return None
        if tensor_contract(graph, source_tensor) != tensor_contract(
            graph,
            outputs[0],
        ):
            return None
        if not self._float_contract(graph, source_tensor):
            return None
        replacement_options = clone_object(options)
        replacement_options.fusedActivationFunction = fused
        return self._plan(
            document,
            graph,
            anchor_operator_index=operator_index,
            replacement_builtin_code=producer_code,
            replacement_version=operator_version(document.model, producer),
            replacement_inputs=as_indices(getattr(producer, "inputs", None)),
            replacement_options_type=int(
                getattr(producer, "builtinOptionsType", 0) or 0
            ),
            replacement_options=replacement_options,
            supporting_operator_indices=(producer_index,),
            tensor_indices=(source_tensor, outputs[0]),
            template_operator_index=producer_index,
        )


class _RecognizeRelu6Rule(_CompositeRuleBase):
    """Replace MIN/MAX or MIN/RELU clamp decompositions with RELU6."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> CompositeFusionPlan | None:
        """Match an exact FLOAT32 clamp to the closed interval [0, 6]."""

        del context
        if not self._algebraic_rewrites_enabled():
            return None
        if not operator_is_live(graph, operator_index):
            return None
        operators = as_list(graph.subgraph.operators)
        anchor = operators[operator_index]
        anchor_code = operator_builtin_code(document.model, anchor)
        if anchor_code == self.codes["MAXIMUM"]:
            match = self._min_max(document, graph, operator_index, anchor)
        elif anchor_code == self.codes["RELU"]:
            match = self._min_relu(document, graph, operator_index, anchor)
        else:
            return None
        if match is None:
            return None
        source_tensor, minimum_index, constants = match
        output_tensor = as_indices(getattr(anchor, "outputs", None))[0]
        if tensor_contract(graph, source_tensor) != tensor_contract(
            graph,
            output_tensor,
        ):
            return None
        if not self._float_contract(graph, source_tensor):
            return None
        return self._plan(
            document,
            graph,
            anchor_operator_index=operator_index,
            replacement_builtin_code=self.codes["RELU6"],
            replacement_inputs=(source_tensor,),
            supporting_operator_indices=(minimum_index,),
            tensor_indices=(*constants, source_tensor, output_tensor),
        )

    def _min_max(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        anchor: Any,
    ) -> tuple[int, int, tuple[int, ...]] | None:
        """Match MAXIMUM(MINIMUM(x, 6), 0) with commutative operands."""

        if not operator_is_plain(anchor):
            return None
        inputs = tuple(as_indices(getattr(anchor, "inputs", None)))
        outputs = tuple(as_indices(getattr(anchor, "outputs", None)))
        if len(inputs) != 2 or len(outputs) != 1:
            return None
        for minimum_tensor, zero_tensor in (inputs, inputs[::-1]):
            zero = scalar_float32(
                self.codec,
                document,
                graph,
                zero_tensor,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if zero != 0.0:
                continue
            minimum_match = producer_matching(
                document,
                graph,
                minimum_tensor,
                builtin_code=self.codes["MINIMUM"],
                input_count=2,
            )
            if minimum_match is None:
                continue
            minimum_index, minimum = minimum_match
            dynamic = _binary_with_scalar(
                self.codec,
                document,
                graph,
                minimum,
                expected=6.0,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if dynamic is None:
                continue
            source_tensor, six_tensor = dynamic
            return source_tensor, minimum_index, (zero_tensor, six_tensor)
        return None

    def _min_relu(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        anchor: Any,
    ) -> tuple[int, int, tuple[int, ...]] | None:
        """Match RELU(MINIMUM(x, 6))."""

        del operator_index
        if not operator_is_plain(anchor):
            return None
        inputs = tuple(as_indices(getattr(anchor, "inputs", None)))
        outputs = tuple(as_indices(getattr(anchor, "outputs", None)))
        if len(inputs) != 1 or len(outputs) != 1:
            return None
        minimum_match = producer_matching(
            document,
            graph,
            inputs[0],
            builtin_code=self.codes["MINIMUM"],
            input_count=2,
        )
        if minimum_match is None:
            return None
        minimum_index, minimum = minimum_match
        dynamic = _binary_with_scalar(
            self.codec,
            document,
            graph,
            minimum,
            expected=6.0,
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
        )
        if dynamic is None:
            return None
        source_tensor, six_tensor = dynamic
        return source_tensor, minimum_index, (six_tensor,)


class _RecognizeRsqrtRule(_CompositeRuleBase):
    """Replace one divided by SQRT with a native RSQRT operator."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> CompositeFusionPlan | None:
        """Match DIV(1.0, SQRT(x)) without a fused activation."""

        del context
        if not self._algebraic_rewrites_enabled():
            return None
        if not operator_is_live(graph, operator_index):
            return None
        operators = as_list(graph.subgraph.operators)
        anchor = operators[operator_index]
        if operator_builtin_code(document.model, anchor) != self.codes["DIV"]:
            return None
        if not operator_is_plain(anchor) or not has_no_fused_activation(
            anchor,
            self.activation_none,
        ):
            return None
        if int(getattr(anchor, "builtinOptionsType", 0) or 0) != (
            self.options_types["DivOptions"]
        ):
            return None
        inputs = tuple(as_indices(getattr(anchor, "inputs", None)))
        outputs = tuple(as_indices(getattr(anchor, "outputs", None)))
        if len(inputs) != 2 or len(outputs) != 1:
            return None
        one = scalar_float32(
            self.codec,
            document,
            graph,
            inputs[0],
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
        )
        if one is None or not np.isclose(one, 1.0, rtol=0.0, atol=1e-6):
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
        sqrt_inputs = tuple(as_indices(getattr(sqrt, "inputs", None)))
        if len(sqrt_inputs) != 1:
            return None
        source_tensor = sqrt_inputs[0]
        if tensor_contract(graph, source_tensor) != tensor_contract(
            graph,
            outputs[0],
        ):
            return None
        if not self._float_contract(graph, source_tensor):
            return None
        return self._plan(
            document,
            graph,
            anchor_operator_index=operator_index,
            replacement_builtin_code=self.codes["RSQRT"],
            replacement_inputs=(source_tensor,),
            supporting_operator_indices=(sqrt_index,),
            tensor_indices=(inputs[0], inputs[1], source_tensor, outputs[0]),
        )


class _RecognizePReluRule(_CompositeRuleBase):
    """Recognize the canonical ABS/SUB/MUL/RELU decomposition of PRELU."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> CompositeFusionPlan | None:
        """Match RELU(x) + 0.5 * alpha * (x - ABS(x))."""

        del context
        if not self._algebraic_rewrites_enabled():
            return None
        if not operator_is_live(graph, operator_index):
            return None
        operators = as_list(graph.subgraph.operators)
        anchor = operators[operator_index]
        if not _binary_operator(
            document,
            anchor,
            code=self.codes["ADD"],
            options_type=self.options_types["AddOptions"],
            activation_none=self.activation_none,
        ):
            return None
        inputs = tuple(as_indices(getattr(anchor, "inputs", None)))
        outputs = tuple(as_indices(getattr(anchor, "outputs", None)))
        if len(inputs) != 2 or len(outputs) != 1:
            return None
        for relu_tensor, half_mul_tensor in (inputs, inputs[::-1]):
            relu_match = producer_matching(
                document,
                graph,
                relu_tensor,
                builtin_code=self.codes["RELU"],
                input_count=1,
            )
            half_match = producer_matching(
                document,
                graph,
                half_mul_tensor,
                builtin_code=self.codes["MUL"],
                input_count=2,
            )
            if relu_match is None or half_match is None:
                continue
            relu_index, relu = relu_match
            half_index, half_mul = half_match
            if not _binary_operator(
                document,
                half_mul,
                code=self.codes["MUL"],
                options_type=self.options_types["MulOptions"],
                activation_none=self.activation_none,
            ):
                continue
            half_dynamic = _binary_with_scalar(
                self.codec,
                document,
                graph,
                half_mul,
                expected=0.5,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if half_dynamic is None:
                continue
            alpha_mul_tensor, half_tensor = half_dynamic
            alpha_match = producer_matching(
                document,
                graph,
                alpha_mul_tensor,
                builtin_code=self.codes["MUL"],
                input_count=2,
            )
            if alpha_match is None:
                continue
            alpha_index, alpha_mul = alpha_match
            if not _binary_operator(
                document,
                alpha_mul,
                code=self.codes["MUL"],
                options_type=self.options_types["MulOptions"],
                activation_none=self.activation_none,
            ):
                continue
            alpha_parts = _binary_dynamic_constant(
                self.codec,
                document,
                graph,
                alpha_mul,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if alpha_parts is None:
                continue
            sub_tensor, alpha_tensor = alpha_parts
            sub_match = producer_matching(
                document,
                graph,
                sub_tensor,
                builtin_code=self.codes["SUB"],
                input_count=2,
            )
            if sub_match is None:
                continue
            sub_index, sub = sub_match
            if not _binary_operator(
                document,
                sub,
                code=self.codes["SUB"],
                options_type=self.options_types["SubOptions"],
                activation_none=self.activation_none,
            ):
                continue
            sub_inputs = tuple(as_indices(getattr(sub, "inputs", None)))
            if len(sub_inputs) != 2:
                continue
            source_tensor = tuple(as_indices(getattr(relu, "inputs", None)))[0]
            if sub_inputs[0] != source_tensor:
                continue
            abs_match = producer_matching(
                document,
                graph,
                sub_inputs[1],
                builtin_code=self.codes["ABS"],
                input_count=1,
            )
            if abs_match is None:
                continue
            abs_index, abs_operator = abs_match
            if tuple(as_indices(getattr(abs_operator, "inputs", None))) != (
                source_tensor,
            ):
                continue
            alpha_pair = decode_float32_constant(
                self.codec,
                document,
                graph,
                alpha_tensor,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if alpha_pair is None:
                continue
            alpha_value, _alpha_contract = alpha_pair
            source_contract = tensor_contract(graph, source_tensor)
            output_contract = tensor_contract(graph, outputs[0])
            if source_contract != output_contract:
                continue
            if not supported_float_contract(
                source_contract,
                float32_type=self.float32_type,
            ):
                continue
            if not _broadcasts_to(alpha_value.shape, source_contract.shape):
                continue
            if alpha_value.nbytes > self.policy.maximum_parameter_bytes:
                continue
            return self._plan(
                document,
                graph,
                anchor_operator_index=operator_index,
                replacement_builtin_code=self.codes["PRELU"],
                replacement_inputs=(source_tensor, alpha_tensor),
                supporting_operator_indices=(
                    relu_index,
                    half_index,
                    alpha_index,
                    sub_index,
                    abs_index,
                ),
                tensor_indices=(
                    source_tensor,
                    alpha_tensor,
                    half_tensor,
                    outputs[0],
                ),
            )
        return None


class _RecognizeGeluRule(_CompositeRuleBase):
    """Recognize exact ERF-based GELU decompositions."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> CompositeFusionPlan | None:
        """Match one of the two standard exact GELU multiplication trees."""

        del context
        if not self._algebraic_rewrites_enabled():
            return None
        if not operator_is_live(graph, operator_index):
            return None
        anchor = as_list(graph.subgraph.operators)[operator_index]
        if not _binary_operator(
            document,
            anchor,
            code=self.codes["MUL"],
            options_type=self.options_types["MulOptions"],
            activation_none=self.activation_none,
        ):
            return None
        match = self._pattern_one(document, graph, anchor)
        if match is None:
            match = self._pattern_two(document, graph, anchor)
        if match is None:
            return None
        source_tensor, supporting, constants = match
        outputs = tuple(as_indices(getattr(anchor, "outputs", None)))
        if len(outputs) != 1:
            return None
        if tensor_contract(graph, source_tensor) != tensor_contract(
            graph,
            outputs[0],
        ):
            return None
        if not self._float_contract(graph, source_tensor):
            return None
        options = self._gelu_options()
        return self._plan(
            document,
            graph,
            anchor_operator_index=operator_index,
            replacement_builtin_code=self.codes["GELU"],
            replacement_inputs=(source_tensor,),
            replacement_options_type=self.options_types["GeluOptions"],
            replacement_options=options,
            supporting_operator_indices=supporting,
            tensor_indices=(*constants, source_tensor, outputs[0]),
        )

    def _pattern_one(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        anchor: Any,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
        """Match 0.5 * (x * (1 + erf(x * sqrt(0.5))))."""

        half_parts = _binary_with_scalar(
            self.codec,
            document,
            graph,
            anchor,
            expected=0.5,
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
        )
        if half_parts is None:
            return None
        core_tensor, half_tensor = half_parts
        core_match = producer_matching(
            document,
            graph,
            core_tensor,
            builtin_code=self.codes["MUL"],
            input_count=2,
        )
        if core_match is None:
            return None
        core_index, core = core_match
        if not _binary_operator(
            document,
            core,
            code=self.codes["MUL"],
            options_type=self.options_types["MulOptions"],
            activation_none=self.activation_none,
        ):
            return None
        for source_tensor, add_tensor in _ordered_pairs(core):
            tail = self._erf_tail(document, graph, source_tensor, add_tensor)
            if tail is None:
                continue
            tail_support, tail_constants = tail
            return (
                source_tensor,
                (core_index, *tail_support),
                (half_tensor, *tail_constants),
            )
        return None

    def _pattern_two(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        anchor: Any,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
        """Match (0.5 * x) * (1 + erf(x * sqrt(0.5)))."""

        for half_mul_tensor, add_tensor in _ordered_pairs(anchor):
            half_match = producer_matching(
                document,
                graph,
                half_mul_tensor,
                builtin_code=self.codes["MUL"],
                input_count=2,
            )
            if half_match is None:
                continue
            half_index, half_mul = half_match
            if not _binary_operator(
                document,
                half_mul,
                code=self.codes["MUL"],
                options_type=self.options_types["MulOptions"],
                activation_none=self.activation_none,
            ):
                continue
            half_parts = _binary_with_scalar(
                self.codec,
                document,
                graph,
                half_mul,
                expected=0.5,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if half_parts is None:
                continue
            source_tensor, half_tensor = half_parts
            tail = self._erf_tail(document, graph, source_tensor, add_tensor)
            if tail is None:
                continue
            tail_support, tail_constants = tail
            return (
                source_tensor,
                (half_index, *tail_support),
                (half_tensor, *tail_constants),
            )
        return None

    def _erf_tail(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        source_tensor: int,
        add_tensor: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Match 1 + Erf(source * sqrt(0.5))."""

        add_match = producer_matching(
            document,
            graph,
            add_tensor,
            builtin_code=self.codes["ADD"],
            input_count=2,
        )
        if add_match is None:
            return None
        add_index, add = add_match
        if not _binary_operator(
            document,
            add,
            code=self.codes["ADD"],
            options_type=self.options_types["AddOptions"],
            activation_none=self.activation_none,
        ):
            return None
        one_parts = _binary_with_scalar(
            self.codec,
            document,
            graph,
            add,
            expected=1.0,
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
        )
        if one_parts is None:
            return None
        erf_tensor, one_tensor = one_parts
        erf_index = graph.producer(erf_tensor)
        if erf_index is None:
            return None
        erf = as_list(graph.subgraph.operators)[erf_index]
        if operator_builtin_code(document.model, erf) != self.codes["CUSTOM"]:
            return None
        if not operator_is_plain(erf):
            return None
        operator_codes = as_list(getattr(document.model, "operatorCodes", None))
        opcode_index = int(getattr(erf, "opcodeIndex", -1))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            return None
        if decode_text(getattr(operator_codes[opcode_index], "customCode", "")) != (
            "Erf"
        ):
            return None
        erf_inputs = tuple(as_indices(getattr(erf, "inputs", None)))
        erf_outputs = tuple(as_indices(getattr(erf, "outputs", None)))
        if len(erf_inputs) != 1 or erf_outputs != (erf_tensor,):
            return None
        sqrt_mul_match = producer_matching(
            document,
            graph,
            erf_inputs[0],
            builtin_code=self.codes["MUL"],
            input_count=2,
        )
        if sqrt_mul_match is None:
            return None
        sqrt_index, sqrt_mul = sqrt_mul_match
        if not _binary_operator(
            document,
            sqrt_mul,
            code=self.codes["MUL"],
            options_type=self.options_types["MulOptions"],
            activation_none=self.activation_none,
        ):
            return None
        sqrt_parts = _binary_with_scalar(
            self.codec,
            document,
            graph,
            sqrt_mul,
            expected=float(np.sqrt(0.5)),
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
            tolerance=1e-5,
        )
        if sqrt_parts is None or sqrt_parts[0] != source_tensor:
            return None
        _source, sqrt_tensor = sqrt_parts
        return (add_index, erf_index, sqrt_index), (one_tensor, sqrt_tensor)

    def _gelu_options(self) -> Any:
        """Create exact GELU options with approximation disabled."""

        options = self.object_factory("GeluOptions") if self.object_factory else None
        if options is None:
            resolver = OptimizationSchemaResolver(object_factory=self.object_factory)
            options = resolver.create("GeluOptions")
        options.approximate = False
        return options


class _RecognizeInstanceNormRule(_CompositeRuleBase):
    """Recognize a canonical NHWC instance-normalization decomposition."""

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> CompositeFusionPlan | None:
        """Match centered variance, RSQRT, gamma, and beta for rank-four NHWC."""

        del context
        if not (
            self.policy.enable_instance_norm
            and self._algebraic_rewrites_enabled()
            and operator_is_live(graph, operator_index)
        ):
            return None
        operators = as_list(graph.subgraph.operators)
        anchor = operators[operator_index]
        if not _binary_operator(
            document,
            anchor,
            code=self.codes["ADD"],
            options_type=self.options_types["AddOptions"],
            activation_none=self.activation_none,
        ):
            return None
        terminal = _binary_dynamic_constant(
            self.codec,
            document,
            graph,
            anchor,
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
        )
        if terminal is None:
            return None
        scaled_tensor, beta_tensor = terminal
        scaled_match = producer_matching(
            document,
            graph,
            scaled_tensor,
            builtin_code=self.codes["MUL"],
            input_count=2,
        )
        if scaled_match is None:
            return None
        scaled_index, scaled = scaled_match
        if not _binary_operator(
            document,
            scaled,
            code=self.codes["MUL"],
            options_type=self.options_types["MulOptions"],
            activation_none=self.activation_none,
        ):
            return None
        gamma_parts = _binary_dynamic_constant(
            self.codec,
            document,
            graph,
            scaled,
            float32_type=self.float32_type,
            require_finite=self.policy.require_finite_constants,
        )
        if gamma_parts is None:
            return None
        normalized_tensor, gamma_tensor = gamma_parts
        normalized_match = producer_matching(
            document,
            graph,
            normalized_tensor,
            builtin_code=self.codes["MUL"],
            input_count=2,
        )
        if normalized_match is None:
            return None
        normalized_index, normalized = normalized_match
        if not _binary_operator(
            document,
            normalized,
            code=self.codes["MUL"],
            options_type=self.options_types["MulOptions"],
            activation_none=self.activation_none,
        ):
            return None
        for centered_tensor, rsqrt_tensor in _ordered_pairs(normalized):
            rsqrt_match = producer_matching(
                document,
                graph,
                rsqrt_tensor,
                builtin_code=self.codes["RSQRT"],
                input_count=1,
            )
            if rsqrt_match is None:
                continue
            rsqrt_index, rsqrt = rsqrt_match
            variance_add_tensor = tuple(as_indices(getattr(rsqrt, "inputs", None)))[0]
            variance_add_match = producer_matching(
                document,
                graph,
                variance_add_tensor,
                builtin_code=self.codes["ADD"],
                input_count=2,
            )
            if variance_add_match is None:
                continue
            variance_add_index, variance_add = variance_add_match
            if not _binary_operator(
                document,
                variance_add,
                code=self.codes["ADD"],
                options_type=self.options_types["AddOptions"],
                activation_none=self.activation_none,
            ):
                continue
            epsilon_parts = _binary_dynamic_constant(
                self.codec,
                document,
                graph,
                variance_add,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if epsilon_parts is None:
                continue
            variance_tensor, epsilon_tensor = epsilon_parts
            epsilon = scalar_float32(
                self.codec,
                document,
                graph,
                epsilon_tensor,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if epsilon is None or epsilon <= 0.0:
                continue
            variance_match = producer_matching(
                document,
                graph,
                variance_tensor,
                builtin_code=self.codes["MEAN"],
                input_count=2,
            )
            if variance_match is None:
                continue
            variance_index, variance = variance_match
            variance_axes = _mean_axes(
                self.codec,
                document,
                graph,
                variance,
                reducer_options_type=self.options_types["ReducerOptions"],
            )
            if variance_axes is None:
                continue
            square_tensor = tuple(as_indices(getattr(variance, "inputs", None)))[0]
            square_match = producer_matching(
                document,
                graph,
                square_tensor,
                builtin_code=self.codes["MUL"],
                input_count=2,
            )
            if square_match is None:
                continue
            square_index, square = square_match
            if not _binary_operator(
                document,
                square,
                code=self.codes["MUL"],
                options_type=self.options_types["MulOptions"],
                activation_none=self.activation_none,
            ):
                continue
            if tuple(as_indices(getattr(square, "inputs", None))) != (
                centered_tensor,
                centered_tensor,
            ):
                continue
            centered_match = producer_matching(
                document,
                graph,
                centered_tensor,
                builtin_code=self.codes["SUB"],
                input_count=2,
            )
            if centered_match is None:
                continue
            centered_index, centered = centered_match
            if not _binary_operator(
                document,
                centered,
                code=self.codes["SUB"],
                options_type=self.options_types["SubOptions"],
                activation_none=self.activation_none,
            ):
                continue
            centered_inputs = tuple(as_indices(getattr(centered, "inputs", None)))
            source_tensor, mean_tensor = centered_inputs
            mean_match = producer_matching(
                document,
                graph,
                mean_tensor,
                builtin_code=self.codes["MEAN"],
                input_count=2,
            )
            if mean_match is None:
                continue
            mean_index, mean = mean_match
            mean_inputs = tuple(as_indices(getattr(mean, "inputs", None)))
            if mean_inputs[0] != source_tensor:
                continue
            mean_axes = _mean_axes(
                self.codec,
                document,
                graph,
                mean,
                reducer_options_type=self.options_types["ReducerOptions"],
            )
            source_contract = tensor_contract(graph, source_tensor)
            if not supported_float_contract(
                source_contract,
                float32_type=self.float32_type,
            ):
                continue
            if source_contract.rank != 4 or source_contract.shape[-1] <= 0:
                continue
            if mean_axes != (1, 2) or variance_axes != (1, 2):
                continue
            gamma_pair = decode_float32_constant(
                self.codec,
                document,
                graph,
                gamma_tensor,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            beta_pair = decode_float32_constant(
                self.codec,
                document,
                graph,
                beta_tensor,
                float32_type=self.float32_type,
                require_finite=self.policy.require_finite_constants,
            )
            if gamma_pair is None or beta_pair is None:
                continue
            gamma, _gamma_contract = gamma_pair
            beta, _beta_contract = beta_pair
            channels = source_contract.shape[-1]
            if not _channel_parameter(gamma.shape, channels):
                continue
            if not _channel_parameter(beta.shape, channels):
                continue
            if gamma.nbytes + beta.nbytes > self.policy.maximum_parameter_bytes:
                continue
            output_tensor = tuple(as_indices(getattr(anchor, "outputs", None)))[0]
            if source_contract != tensor_contract(graph, output_tensor):
                continue
            options = self._instance_norm_options(epsilon)
            return self._plan(
                document,
                graph,
                anchor_operator_index=operator_index,
                replacement_builtin_code=self.codes["INSTANCE_NORM"],
                replacement_inputs=(source_tensor, gamma_tensor, beta_tensor),
                replacement_options_type=self.options_types["InstanceNormOptions"],
                replacement_options=options,
                supporting_operator_indices=(
                    scaled_index,
                    normalized_index,
                    rsqrt_index,
                    variance_add_index,
                    variance_index,
                    square_index,
                    centered_index,
                    mean_index,
                ),
                tensor_indices=(
                    source_tensor,
                    gamma_tensor,
                    beta_tensor,
                    epsilon_tensor,
                    output_tensor,
                ),
            )
        return None

    def _instance_norm_options(self, epsilon: float) -> Any:
        """Create instance-normalization options with no fused activation."""

        options = (
            self.object_factory("InstanceNormOptions")
            if self.object_factory
            else OptimizationSchemaResolver().create("InstanceNormOptions")
        )
        options.epsilon = float(epsilon)
        options.fusedActivationFunction = self.activation_none
        return options


class FuseCompositeOpsPass(CircleRulePass):
    """Recognize common decompositions and replace them with Circle builtins."""

    def __init__(
        self,
        *,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        activation_none: int | None = None,
        activation_relu: int | None = None,
        activation_relu6: int | None = None,
        activation_relu_n1_to_1: int | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
        policy: CompositeFusionPolicy | None = None,
        maximum_rewrites: int = 10_000,
    ) -> None:
        """Create composite recognizers with schema-independent enum overrides."""

        resolver = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            activation_none=activation_none,
            object_factory=object_factory,
        )
        codes = {
            name: resolver.builtin_code(name)
            for name in (
                "ABS",
                "ADD",
                "CUSTOM",
                "DIV",
                "GELU",
                "INSTANCE_NORM",
                "MAXIMUM",
                "MEAN",
                "MINIMUM",
                "MUL",
                "PRELU",
                "RELU",
                "RELU6",
                "RELU_N1_TO_1",
                "RSQRT",
                "SQRT",
                "SUB",
            )
        }
        for optional in ("CONCATENATION", "TRANSPOSE_CONV"):
            value = resolver.maybe_builtin_code(optional)
            if value is not None:
                codes[optional] = value
        codes.update(
            {
                "ACTIVATION_RELU": (
                    int(activation_relu)
                    if activation_relu is not None
                    else _activation_enum("RELU")
                ),
                "ACTIVATION_RELU6": (
                    int(activation_relu6)
                    if activation_relu6 is not None
                    else _activation_enum("RELU6")
                ),
                "ACTIVATION_RELU_N1_TO_1": (
                    int(activation_relu_n1_to_1)
                    if activation_relu_n1_to_1 is not None
                    else _activation_enum("RELU_N1_TO_1")
                ),
            }
        )
        options_types = {
            name: resolver.builtin_options_type(name)
            for name in (
                "AddOptions",
                "DivOptions",
                "GeluOptions",
                "InstanceNormOptions",
                "MulOptions",
                "ReducerOptions",
                "SubOptions",
            )
        }
        resolved_policy = policy or CompositeFusionPolicy()
        shared = {
            "codes": codes,
            "options_types": options_types,
            "float32_type": resolver.tensor_type("FLOAT32"),
            "activation_none": resolver.activation_none,
            "codec": codec or TensorValueCodec(),
            "object_factory": object_factory,
            "policy": resolved_policy,
        }
        rules: list[CircleRewriteRule[CompositeFusionPlan]] = [
            _FuseActivationAttributeRule(**shared),
            _RecognizeRelu6Rule(**shared),
            _RecognizeRsqrtRule(**shared),
            _RecognizePReluRule(**shared),
            _RecognizeGeluRule(**shared),
        ]
        if resolved_policy.enable_instance_norm:
            rules.append(_RecognizeInstanceNormRule(**shared))
        super().__init__(rules, maximum_rewrites=maximum_rewrites)


def _activation_enum(name: str) -> int:
    """Resolve one ActivationFunctionType member from the generated schema."""

    from tico.circle._schema import circle_schema

    schema = circle_schema()
    module = getattr(schema, "ActivationFunctionType", None)
    enum_type = (
        getattr(module, "ActivationFunctionType", None) if module is not None else None
    )
    if enum_type is None:
        enum_type = module
    if enum_type is None or not hasattr(enum_type, name):
        raise RuntimeError(
            f"Circle schema does not provide ActivationFunctionType.{name}."
        )
    return int(getattr(enum_type, name))


def _activation_target(
    builtin_code: int,
    *,
    codes: Mapping[str, int],
) -> int | None:
    """Map one standalone activation builtin to its fused-activation enum."""

    mapping = {
        codes["RELU"]: codes["ACTIVATION_RELU"],
        codes["RELU6"]: codes["ACTIVATION_RELU6"],
        codes["RELU_N1_TO_1"]: codes["ACTIVATION_RELU_N1_TO_1"],
    }
    return mapping.get(int(builtin_code))


def _combine_activation(
    existing: int,
    requested: int,
    *,
    none: int,
    relu: int,
    relu6: int,
    relu_n1_to_1: int,
) -> int | None:
    """Return the exact fused activation for two supported activation stages."""

    if requested == relu:
        if existing in {none, relu}:
            return relu
        if existing == relu6:
            return relu6
        return None
    if requested == relu6:
        if existing in {none, relu, relu6}:
            return relu6
        return None
    if requested == relu_n1_to_1:
        if existing in {none, relu_n1_to_1}:
            return relu_n1_to_1
    return None


def _binary_operator(
    document: CircleDocument,
    operator: Any,
    *,
    code: int,
    options_type: int,
    activation_none: int,
) -> bool:
    """Return whether an operator is a plain binary op without activation."""

    return (
        operator_builtin_code(document.model, operator) == int(code)
        and operator_is_plain(operator)
        and len(as_indices(getattr(operator, "inputs", None))) == 2
        and len(as_indices(getattr(operator, "outputs", None))) == 1
        and int(getattr(operator, "builtinOptionsType", 0) or 0) == int(options_type)
        and has_no_fused_activation(operator, activation_none)
    )


def _binary_dynamic_constant(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph: CircleGraph,
    operator: Any,
    *,
    float32_type: int,
    require_finite: bool,
) -> tuple[int, int] | None:
    """Split binary inputs into one dynamic tensor and one FLOAT32 constant."""

    inputs = tuple(as_indices(getattr(operator, "inputs", None)))
    if len(inputs) != 2:
        return None
    for dynamic_tensor, constant_tensor in (inputs, inputs[::-1]):
        pair = decode_float32_constant(
            codec,
            document,
            graph,
            constant_tensor,
            float32_type=float32_type,
            require_finite=require_finite,
        )
        if pair is None:
            continue
        if (
            decode_float32_constant(
                codec,
                document,
                graph,
                dynamic_tensor,
                float32_type=float32_type,
                require_finite=require_finite,
            )
            is not None
        ):
            continue
        return dynamic_tensor, constant_tensor
    return None


def _binary_with_scalar(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph: CircleGraph,
    operator: Any,
    *,
    expected: float,
    float32_type: int,
    require_finite: bool,
    tolerance: float = 0.0,
) -> tuple[int, int] | None:
    """Split a binary operator when one operand is an expected scalar constant."""

    inputs = tuple(as_indices(getattr(operator, "inputs", None)))
    if len(inputs) != 2:
        return None
    for dynamic_tensor, constant_tensor in (inputs, inputs[::-1]):
        value = scalar_float32(
            codec,
            document,
            graph,
            constant_tensor,
            float32_type=float32_type,
            require_finite=require_finite,
        )
        if value is None:
            continue
        if not np.isclose(value, expected, rtol=0.0, atol=tolerance):
            continue
        return dynamic_tensor, constant_tensor
    return None


def _ordered_pairs(operator: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return both input orders of a binary operator."""

    inputs = tuple(as_indices(getattr(operator, "inputs", None)))
    if len(inputs) != 2:
        return ((-1, -1), (-1, -1))
    return (inputs, inputs[::-1])


def _broadcasts_to(source_shape: Sequence[int], target_shape: Sequence[int]) -> bool:
    """Return whether one shape broadcasts exactly to a requested target."""

    try:
        return tuple(np.broadcast_shapes(tuple(source_shape), tuple(target_shape))) == (
            tuple(target_shape)
        )
    except ValueError:
        return False


def _channel_parameter(shape: Sequence[int], channels: int) -> bool:
    """Return whether a parameter is scalar or aligned to the NHWC channel axis."""

    normalized = tuple(int(dimension) for dimension in shape)
    if np.prod(normalized, dtype=np.int64) == 1:
        return True
    return normalized in {
        (channels,),
        (1, channels),
        (1, 1, channels),
        (1, 1, 1, channels),
    }


def _mean_axes(
    codec: TensorValueCodec,
    document: CircleDocument,
    graph: CircleGraph,
    operator: Any,
    *,
    reducer_options_type: int,
) -> tuple[int, ...] | None:
    """Return normalized axes for a keep-dims MEAN operator."""

    if not operator_is_plain(operator):
        return None
    if int(getattr(operator, "builtinOptionsType", 0) or 0) != int(
        reducer_options_type
    ):
        return None
    options = getattr(operator, "builtinOptions", None)
    if options is None or not bool(getattr(options, "keepDims", False)):
        return None
    inputs = tuple(as_indices(getattr(operator, "inputs", None)))
    if len(inputs) != 2:
        return None
    axes_pair = decode_integer_constant(
        codec,
        document,
        graph,
        inputs[1],
    )
    if axes_pair is None:
        return None
    axes, _contract = axes_pair
    rank = tensor_contract(graph, inputs[0]).rank
    normalized = normalize_axes(axes, rank)
    if normalized is None:
        return None
    return tuple(sorted(normalized))
