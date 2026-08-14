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
from typing import Any, Iterable

import numpy as np

from tico.circle._object import ObjectFactory
from tico.circle.analysis import TensorContract
from tico.circle.builder import ConstantPool
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError, CircleValueError
from tico.circle.graph import as_indices, as_list, CircleGraph, OPTIONAL_TENSOR_INDEX
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.cleanup import DeadCodeEliminationPass
from tico.circle.passes.optimization.fold.evaluators import (
    ConstantEvaluation,
    ConstantEvaluationContext,
    ConstantEvaluator,
    ConstantEvaluatorRegistry,
    default_constant_evaluator_registry,
)
from tico.circle.passes.optimization.fold.evaluators.base import (
    contract_is_dense_value,
    contract_is_fully_static,
)
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
    RewriteSeverity,
)
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True)
class ConstantFoldPolicy:
    """Bound memory growth and evaluator work performed by constant folding."""

    maximum_input_bytes: int = 64 * 1024 * 1024
    maximum_output_bytes: int = 16 * 1024 * 1024
    maximum_total_output_bytes: int = 256 * 1024 * 1024
    maximum_compute_cost: int = 16_000_000
    maximum_total_compute_cost: int = 256_000_000
    maximum_folds: int = 10_000
    eliminate_dead_code: bool = True
    prune_unused_inputs: bool = True

    def __post_init__(self) -> None:
        """Reject negative limits and a non-positive fixed-point bound."""

        non_negative_fields = (
            "maximum_input_bytes",
            "maximum_output_bytes",
            "maximum_total_output_bytes",
            "maximum_compute_cost",
            "maximum_total_compute_cost",
        )
        for field_name in non_negative_fields:
            value = int(getattr(self, field_name))
            object.__setattr__(self, field_name, value)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative.")
        object.__setattr__(self, "maximum_folds", int(self.maximum_folds))
        object.__setattr__(
            self,
            "eliminate_dead_code",
            bool(self.eliminate_dead_code),
        )
        object.__setattr__(
            self,
            "prune_unused_inputs",
            bool(self.prune_unused_inputs),
        )
        if self.maximum_folds <= 0:
            raise ValueError("maximum_folds must be positive.")


@dataclass(frozen=True, kw_only=True)
class ConstantFoldPlan(RewritePlan):
    """Carry encoded output payloads and budget estimates into graph mutation."""

    output_payloads: tuple[bytes, ...]
    output_bytes: int
    compute_cost: int
    evaluator_name: str

    def __post_init__(self) -> None:
        """Normalize payloads and validate output arity and budget metadata."""

        payloads = tuple(bytes(payload) for payload in self.output_payloads)
        object.__setattr__(self, "output_payloads", payloads)
        object.__setattr__(self, "output_bytes", int(self.output_bytes))
        object.__setattr__(self, "compute_cost", int(self.compute_cost))
        if len(payloads) != len(self.anchor.outputs):
            raise ValueError(
                "Constant-fold payload count must match the anchor output count."
            )
        if self.output_bytes != sum(len(payload) for payload in payloads):
            raise ValueError("output_bytes must equal encoded payload storage.")
        if self.compute_cost < 0:
            raise ValueError("compute_cost must not be negative.")
        if not self.evaluator_name:
            raise ValueError("evaluator_name must not be empty.")


@dataclass
class _FoldBudgetState:
    """Track cumulative output storage and evaluator work for one pass run."""

    policy: ConstantFoldPolicy
    output_bytes: int = 0
    compute_cost: int = 0
    folds: int = 0

    def rejection_reason(
        self,
        *,
        input_bytes: int,
        output_bytes: int,
        compute_cost: int,
    ) -> str | None:
        """Return a stable reason when one candidate exceeds configured limits."""

        if input_bytes > self.policy.maximum_input_bytes:
            return (
                f"constant inputs require {input_bytes} bytes, exceeding the "
                f"per-operator limit {self.policy.maximum_input_bytes}"
            )
        if output_bytes > self.policy.maximum_output_bytes:
            return (
                f"constant outputs require {output_bytes} bytes, exceeding the "
                f"per-operator limit {self.policy.maximum_output_bytes}"
            )
        if self.output_bytes + output_bytes > self.policy.maximum_total_output_bytes:
            return (
                f"cumulative constant outputs would require "
                f"{self.output_bytes + output_bytes} bytes, exceeding the total "
                f"limit {self.policy.maximum_total_output_bytes}"
            )
        if compute_cost > self.policy.maximum_compute_cost:
            return (
                f"estimated compute cost {compute_cost} exceeds the per-operator "
                f"limit {self.policy.maximum_compute_cost}"
            )
        if self.compute_cost + compute_cost > self.policy.maximum_total_compute_cost:
            return (
                f"cumulative compute cost would be "
                f"{self.compute_cost + compute_cost}, exceeding the total limit "
                f"{self.policy.maximum_total_compute_cost}"
            )
        return None

    def consume(self, *, output_bytes: int, compute_cost: int) -> None:
        """Commit one successful fold to cumulative budget accounting."""

        self.output_bytes += int(output_bytes)
        self.compute_cost += int(compute_cost)
        self.folds += 1


class _FoldConstantOperatorRule(CircleRewriteRule[ConstantFoldPlan]):
    """Evaluate one supported operator and replace all outputs with constants."""

    def __init__(
        self,
        document: CircleDocument,
        *,
        evaluator_registry: ConstantEvaluatorRegistry,
        codec: TensorValueCodec,
        object_factory: ObjectFactory | None,
        budget: _FoldBudgetState,
    ) -> None:
        """Bind evaluator, value, builder, and budget services for one pass run."""

        self.evaluator_registry = evaluator_registry
        self.codec = codec
        self.object_factory = object_factory
        self.budget = budget
        self.constant_pool = ConstantPool(
            document.model,
            codec=codec,
            object_factory=object_factory,
        )
        self.folded_subgraphs: set[int] = set()
        self._skipped_diagnostics: dict[str, RewriteDiagnostic] = {}

    @property
    def skipped_diagnostics(self) -> tuple[RewriteDiagnostic, ...]:
        """Return unique budget diagnostics in stable insertion order."""

        return tuple(self._skipped_diagnostics.values())

    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> ConstantFoldPlan | None:
        """Evaluate one candidate without mutating graph or budget state."""

        operator = as_list(graph.subgraph.operators)[operator_index]
        operator_code = _operator_code(document.model, operator)
        builtin_code = int(getattr(operator_code, "builtinCode", -1))
        evaluator = self.evaluator_registry.get(builtin_code)
        if evaluator is None:
            return None

        evaluation_context = _make_evaluation_context(
            document,
            graph,
            operator_index,
            operator,
            operator_code,
            self.codec,
        )
        if not evaluation_context.output_indices:
            return None
        graph_outputs = set(graph.outputs)
        if not any(
            graph.consumers(tensor_index) or tensor_index in graph_outputs
            for tensor_index in evaluation_context.output_indices
        ):
            return None
        if not all(
            contract_is_fully_static(contract) and contract_is_dense_value(contract)
            for contract in evaluation_context.output_contracts
        ):
            return None
        if any(
            contract.element_count == 0
            for contract in evaluation_context.output_contracts
        ):
            return None

        required_positions = evaluator.constant_input_positions(evaluation_context)
        _validate_required_positions(evaluation_context, required_positions)
        payloads = _required_input_payloads(
            document.model,
            graph,
            evaluation_context.input_indices,
            required_positions,
        )
        if payloads is None:
            return None
        input_bytes = sum(len(payload) for payload in payloads.values())
        output_bytes = _estimate_output_bytes(
            evaluation_context.output_contracts,
            self.codec,
        )
        if output_bytes is None:
            return None
        compute_cost = int(evaluator.estimate_compute_cost(evaluation_context))
        if compute_cost < 0:
            raise CircleRewriteError(
                f"Evaluator {evaluator.name} returned a negative "
                "compute-cost estimate."
            )
        reason = self.budget.rejection_reason(
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            compute_cost=compute_cost,
        )
        if reason is not None:
            self._record_budget_skip(graph, operator_index, evaluator, reason)
            return None

        input_values: list[TensorValue | None] = [
            None for _ in evaluation_context.input_indices
        ]
        decoded_inputs: dict[int, TensorValue] = {}
        for position in required_positions:
            tensor_index = evaluation_context.input_indices[position]
            contract = evaluation_context.input_contract(position)
            if not contract_is_fully_static(contract):
                return None
            value = decoded_inputs.get(tensor_index)
            if value is None:
                try:
                    value = self.codec.decode_tensor(
                        document.model,
                        subgraph_index=graph.subgraph_index,
                        tensor_index=tensor_index,
                    )
                except CircleValueError:
                    return None
                decoded_inputs[tensor_index] = value
            input_values[position] = value
        evaluation_context = evaluation_context.with_input_values(input_values)
        evaluation = evaluator.evaluate(evaluation_context)
        if evaluation is None:
            return None
        output_payloads = _encode_evaluation(
            evaluation,
            evaluation_context.output_contracts,
            self.codec,
        )
        if output_payloads is None:
            return None
        actual_output_bytes = sum(len(payload) for payload in output_payloads)
        reason = self.budget.rejection_reason(
            input_bytes=input_bytes,
            output_bytes=actual_output_bytes,
            compute_cost=compute_cost,
        )
        if reason is not None:
            self._record_budget_skip(graph, operator_index, evaluator, reason)
            return None

        captured_indices = tuple(
            dict.fromkeys(
                tensor_index
                for tensor_index in (
                    *evaluation_context.input_indices,
                    *evaluation_context.output_indices,
                )
                if tensor_index != OPTIONAL_TENSOR_INDEX
            )
        )
        return ConstantFoldPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=captured_indices,
            output_payloads=output_payloads,
            output_bytes=actual_output_bytes,
            compute_cost=compute_cost,
            evaluator_name=evaluator.name,
        )

    def apply(
        self,
        document: CircleDocument,
        plan: ConstantFoldPlan,
        context: CirclePassContext,
    ) -> RewriteApplication:
        """Attach buffers to existing outputs and remove the folded operator."""

        subgraph = document.subgraph(plan.subgraph_index)
        tensors = as_list(subgraph.tensors)
        buffers = as_list(document.model.buffers)
        original_buffer_count = len(buffers)
        output_indices = plan.anchor.outputs
        original_output_buffers = tuple(
            int(getattr(tensors[tensor_index], "buffer", 0) or 0)
            for tensor_index in output_indices
        )

        try:
            output_buffer_indices = tuple(
                self.constant_pool.intern_buffer(payload)
                for payload in plan.output_payloads
            )
        except Exception:
            self._rollback_buffers(document, original_buffer_count)
            raise

        operators = as_list(subgraph.operators)
        del operators[plan.anchor_operator_index]
        try:
            for tensor_index, buffer_index in zip(
                output_indices,
                output_buffer_indices,
            ):
                tensors[tensor_index].buffer = buffer_index
            subgraph.operators = operators
        except Exception:
            for tensor_index, buffer_index in zip(
                output_indices,
                original_output_buffers,
            ):
                tensors[tensor_index].buffer = buffer_index
            self._rollback_buffers(document, original_buffer_count)
            raise

        changed_output_buffers = sum(
            current != previous
            for current, previous in zip(
                output_buffer_indices,
                original_output_buffers,
            )
        )
        added_buffers = len(as_list(document.model.buffers)) - original_buffer_count
        self.budget.consume(
            output_bytes=plan.output_bytes,
            compute_cost=plan.compute_cost,
        )
        self.folded_subgraphs.add(plan.subgraph_index)
        changes = 1 + changed_output_buffers + max(0, added_buffers)
        return RewriteApplication(
            changes=changes,
            diagnostics=(
                RewriteDiagnostic(
                    code="CONSTANT_FOLDED",
                    severity=RewriteSeverity.INFO,
                    object_path=(
                        f"subgraphs[{plan.subgraph_index}]"
                        f".operators[{plan.anchor_operator_index}]"
                    ),
                    message=(
                        f"Folded {len(plan.output_payloads)} output tensor(s) "
                        f"with {plan.evaluator_name}."
                    ),
                ),
            ),
        )

    def _rollback_buffers(
        self,
        document: CircleDocument,
        original_buffer_count: int,
    ) -> None:
        """Remove appended buffers and rebuild constant-pool indexes."""

        _truncate_sequence(document.model, "buffers", original_buffer_count)
        self.constant_pool = ConstantPool(
            document.model,
            codec=self.codec,
            object_factory=self.object_factory,
        )

    def _record_budget_skip(
        self,
        graph: CircleGraph,
        operator_index: int,
        evaluator: ConstantEvaluator,
        reason: str,
    ) -> None:
        """Record one deduplicated informational diagnostic for a budget skip."""

        object_path = f"subgraphs[{graph.subgraph_index}].operators[{operator_index}]"
        key = f"{object_path}:{evaluator.name}:{reason}"
        self._skipped_diagnostics.setdefault(
            key,
            RewriteDiagnostic(
                code="CONSTANT_FOLD_BUDGET",
                severity=RewriteSeverity.INFO,
                object_path=object_path,
                message=f"Skipped {evaluator.name}: {reason}.",
            ),
        )


class FoldConstantSubgraphPass(CirclePass):
    """Fold supported constant Circle operators to a fixed point and run DCE."""

    def __init__(
        self,
        *,
        policy: ConstantFoldPolicy | None = None,
        evaluator_registry: ConstantEvaluatorRegistry | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create a pass with lazily initialized, injectable fold services."""

        self.policy = policy or ConstantFoldPolicy()
        self.evaluator_registry: ConstantEvaluatorRegistry | None = evaluator_registry
        self.codec: TensorValueCodec | None = codec
        self.object_factory = object_factory

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Fold one candidate at a time, restart, and clean newly dead producers."""

        budget = _FoldBudgetState(self.policy)
        evaluator_registry = (
            self.evaluator_registry or default_constant_evaluator_registry()
        )
        codec = self.codec or TensorValueCodec()
        rule = _FoldConstantOperatorRule(
            document,
            evaluator_registry=evaluator_registry,
            codec=codec,
            object_factory=self.object_factory,
            budget=budget,
        )
        rule_result = CircleRulePass(
            [rule],
            maximum_rewrites=self.policy.maximum_folds,
        ).run(document, context)

        cleanup_result = CirclePassResult(modified=False)
        input_cleanup_result = CirclePassResult(modified=False)
        if (
            rule_result.modified
            and self.policy.eliminate_dead_code
            and rule.folded_subgraphs
        ):
            folded_subgraphs = tuple(sorted(rule.folded_subgraphs))
            cleanup_result = DeadCodeEliminationPass(
                subgraph_indices=folded_subgraphs,
                prune_unused_inputs=False,
            ).run(document, context)
            if self.policy.prune_unused_inputs:
                input_cleanup_result = _prune_unused_graph_inputs(
                    document,
                    folded_subgraphs,
                )

        skipped = tuple(
            diagnostic.for_rule(rule.name).format()
            for diagnostic in rule.skipped_diagnostics
        )
        changes = (
            rule_result.changes + cleanup_result.changes + input_cleanup_result.changes
        )
        context.logger.debug(
            "Constant folding applied %d folds, accounted for %d output bytes, and "
            "accounted for %d compute-cost units.",
            budget.folds,
            budget.output_bytes,
            budget.compute_cost,
        )
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
            diagnostics=(
                *rule_result.diagnostics,
                *skipped,
                *cleanup_result.diagnostics,
                *input_cleanup_result.diagnostics,
            ),
        )


def _prune_unused_graph_inputs(
    document: CircleDocument,
    subgraph_indices: Iterable[int],
) -> CirclePassResult:
    """Remove unused graph inputs even when a folded graph has no operators."""

    removed_inputs = 0
    diagnostics: list[str] = []
    signature_inputs: dict[int, set[int]] = {}
    for signature in as_list(getattr(document.model, "signatureDefs", None)):
        subgraph_index = int(getattr(signature, "subgraphIndex", -1))
        mapped_inputs = signature_inputs.setdefault(subgraph_index, set())
        mapped_inputs.update(
            int(getattr(tensor_map, "tensorIndex", -1))
            for tensor_map in as_list(getattr(signature, "inputs", None))
        )

    for subgraph_index in subgraph_indices:
        subgraph_index = int(subgraph_index)
        subgraph = document.subgraph(subgraph_index)
        consumed = {
            tensor_index
            for operator in as_list(getattr(subgraph, "operators", None))
            for tensor_index in as_indices(getattr(operator, "inputs", None))
            if tensor_index != OPTIONAL_TENSOR_INDEX
        }
        output_set = set(as_indices(getattr(subgraph, "outputs", None)))
        protected_inputs = output_set | signature_inputs.get(subgraph_index, set())
        old_inputs = as_indices(getattr(subgraph, "inputs", None))
        removed = [
            tensor_index
            for tensor_index in old_inputs
            if tensor_index not in consumed and tensor_index not in protected_inputs
        ]
        if not removed:
            continue
        removed_set = set(removed)
        subgraph.inputs = [
            tensor_index
            for tensor_index in old_inputs
            if tensor_index not in removed_set
        ]
        removed_inputs += len(removed)
        diagnostics.append(
            f"Subgraph {subgraph_index}: removed graph inputs {removed}."
        )

    return CirclePassResult(
        modified=removed_inputs > 0,
        changes=removed_inputs,
        diagnostics=tuple(diagnostics),
    )


def _truncate_sequence(owner: Any, field_name: str, size: int) -> None:
    """Rollback appended Object API values while preserving the original prefix."""

    values = as_list(getattr(owner, field_name, None))
    del values[size:]
    setattr(owner, field_name, values)


def _make_evaluation_context(
    document: CircleDocument,
    graph: CircleGraph,
    operator_index: int,
    operator: Any,
    operator_code: Any,
    codec: TensorValueCodec,
) -> ConstantEvaluationContext:
    """Capture positional tensor contracts before any evaluator reads payloads."""

    tensors = as_list(graph.subgraph.tensors)
    input_indices = tuple(as_indices(getattr(operator, "inputs", None)))
    output_indices = tuple(as_indices(getattr(operator, "outputs", None)))
    input_contracts: list[TensorContract | None] = []
    for position, tensor_index in enumerate(input_indices):
        if tensor_index == OPTIONAL_TENSOR_INDEX:
            input_contracts.append(None)
            continue
        if tensor_index < 0 or tensor_index >= len(tensors):
            raise CircleRewriteError(
                f"Operator {operator_index} input {position} references invalid "
                f"tensor {tensor_index}."
            )
        input_contracts.append(TensorContract.from_tensor(tensors[tensor_index]))

    output_contracts: list[TensorContract] = []
    for position, tensor_index in enumerate(output_indices):
        if tensor_index < 0 or tensor_index >= len(tensors):
            raise CircleRewriteError(
                f"Operator {operator_index} output {position} references invalid "
                f"tensor {tensor_index}."
            )
        output_contracts.append(TensorContract.from_tensor(tensors[tensor_index]))

    return ConstantEvaluationContext(
        document=document,
        graph=graph,
        operator_index=operator_index,
        operator=operator,
        operator_code=operator_code,
        input_indices=input_indices,
        output_indices=output_indices,
        input_contracts=tuple(input_contracts),
        output_contracts=tuple(output_contracts),
        input_values=tuple(None for _ in input_indices),
        codec=codec,
    )


def _operator_code(model: Any, operator: Any) -> Any:
    """Return one valid model-global operator-code table."""

    operator_codes = as_list(getattr(model, "operatorCodes", None))
    opcode_index = int(getattr(operator, "opcodeIndex", -1))
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        raise CircleRewriteError(
            f"Operator references invalid opcode index {opcode_index}."
        )
    return operator_codes[opcode_index]


def _validate_required_positions(
    context: ConstantEvaluationContext,
    positions: Iterable[int],
) -> None:
    """Reject evaluator bugs that request invalid or optional input positions."""

    seen: set[int] = set()
    for position in positions:
        position = int(position)
        if position in seen:
            raise CircleRewriteError(
                f"Evaluator requested duplicate constant input position {position}."
            )
        seen.add(position)
        if position < 0 or position >= len(context.input_indices):
            raise CircleRewriteError(
                f"Evaluator requested invalid constant input position {position}."
            )
        if context.input_indices[position] == OPTIONAL_TENSOR_INDEX:
            raise CircleRewriteError(
                f"Evaluator requested absent optional input position {position}."
            )


def _required_input_payloads(
    model: Any,
    graph: CircleGraph,
    input_indices: tuple[int, ...],
    positions: Iterable[int],
) -> dict[int, bytes] | None:
    """Return unique inline payloads for all required constant input positions."""

    tensors = as_list(getattr(graph.subgraph, "tensors", None))
    buffers = as_list(getattr(model, "buffers", None))
    graph_inputs = set(graph.inputs)
    payloads: dict[int, bytes] = {}
    for position in positions:
        tensor_index = input_indices[position]
        if tensor_index in payloads:
            continue
        if tensor_index in graph_inputs or graph.producer(tensor_index) is not None:
            return None
        tensor = tensors[tensor_index]
        if bool(getattr(tensor, "isVariable", False)):
            return None
        buffer_index = int(getattr(tensor, "buffer", 0) or 0)
        if buffer_index <= 0 or buffer_index >= len(buffers):
            return None
        payload = _inline_payload(buffers[buffer_index])
        if payload is None:
            return None
        payloads[tensor_index] = payload
    return payloads


def _inline_payload(buffer: Any) -> bytes | None:
    """Return exact inline bytes while rejecting external or absent storage."""

    if int(getattr(buffer, "offset", 0) or 0) or int(getattr(buffer, "size", 0) or 0):
        return None
    data = getattr(buffer, "data", None)
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    try:
        array = np.asarray(data, dtype=np.uint8)
    except (TypeError, ValueError):
        return None
    return bytes(np.ascontiguousarray(array).reshape(-1))


def _estimate_output_bytes(
    contracts: tuple[TensorContract, ...],
    codec: TensorValueCodec,
) -> int | None:
    """Estimate serialized storage from output contracts without allocating values."""

    total = 0
    for contract in contracts:
        spec = codec.registry.get(contract.tensor_type)
        if spec is None:
            return None
        total += spec.storage_size(contract.element_count)
    return total


def _encode_evaluation(
    evaluation: ConstantEvaluation,
    contracts: tuple[TensorContract, ...],
    codec: TensorValueCodec,
) -> tuple[bytes, ...] | None:
    """Validate evaluator outputs and return their exact serialized payloads."""

    if len(evaluation.outputs) != len(contracts):
        raise CircleRewriteError(
            f"Constant evaluator produced {len(evaluation.outputs)} outputs for "
            f"an operator with {len(contracts)} outputs."
        )
    payloads: list[bytes] = []
    for position, (value, contract) in enumerate(zip(evaluation.outputs, contracts)):
        if value.tensor_type != contract.tensor_type:
            raise CircleRewriteError(
                f"Constant evaluator output {position} changed tensor type."
            )
        if value.shape != contract.shape:
            raise CircleRewriteError(
                f"Constant evaluator output {position} changed tensor shape."
            )
        if value.quantization != contract.quantization:
            raise CircleRewriteError(
                f"Constant evaluator output {position} changed quantization."
            )
        try:
            payloads.append(codec.encode(value))
        except CircleValueError:
            return None
    return tuple(payloads)
