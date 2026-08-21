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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Generic, Iterable, Sequence, TypeVar

from tico.circle._object import freeze_object, FrozenValue
from tico.circle._schema import decode_text
from tico.circle.analysis import TensorContract
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list, CircleGraph
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult


class RewriteSeverity(str, Enum):
    """Classify structured rewrite diagnostics by operational severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class RewriteDiagnostic:
    """Describe one stable diagnostic emitted while planning or applying a rewrite."""

    code: str
    message: str
    severity: RewriteSeverity = RewriteSeverity.INFO
    object_path: str | None = None
    rule_name: str | None = None

    def __post_init__(self) -> None:
        """Reject diagnostics without a stable code or human-readable message."""

        if not self.code:
            raise ValueError("Rewrite diagnostic codes must not be empty.")
        if not self.message:
            raise ValueError("Rewrite diagnostic messages must not be empty.")

    def for_rule(self, rule_name: str) -> RewriteDiagnostic:
        """Return this diagnostic with a rule name when none is already present."""

        if self.rule_name is not None:
            return self
        return replace(self, rule_name=rule_name)

    def format(self) -> str:
        """Return a stable single-line diagnostic representation."""

        prefix = f"{self.severity.value.upper()} [{self.code}]"
        if self.rule_name:
            prefix += f" {self.rule_name}"
        if self.object_path:
            prefix += f" {self.object_path}"
        return f"{prefix}: {self.message}"


@dataclass(frozen=True)
class OperatorSnapshot:
    """Capture all operator-local fields that make a rewrite match valid."""

    operator_index: int
    operator_fingerprint: FrozenValue
    opcode_index: int
    operator_code_fingerprint: FrozenValue
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    intermediates: tuple[int, ...]
    mutating_variable_inputs: tuple[bool, ...]
    builtin_options_type: int
    builtin_options_fingerprint: FrozenValue
    builtin_options2_type: int
    builtin_options2_fingerprint: FrozenValue
    custom_options_format: int
    custom_options_fingerprint: FrozenValue

    @classmethod
    def capture(
        cls,
        document: CircleDocument,
        *,
        subgraph_index: int,
        operator_index: int,
    ) -> OperatorSnapshot:
        """Capture an operator and its referenced operator-code identity."""

        subgraph = document.subgraph(subgraph_index)
        operators = as_list(getattr(subgraph, "operators", None))
        if operator_index < 0 or operator_index >= len(operators):
            raise IndexError(
                f"Operator index {operator_index} is outside 0..{len(operators) - 1}."
            )
        operator = operators[operator_index]
        opcode_index = int(getattr(operator, "opcodeIndex", -1))
        operator_codes = as_list(getattr(document.model, "operatorCodes", None))
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            raise CircleRewriteError(
                f"Operator {operator_index} references invalid opcode {opcode_index}."
            )
        return cls(
            operator_index=operator_index,
            operator_fingerprint=freeze_object(operator),
            opcode_index=opcode_index,
            operator_code_fingerprint=freeze_object(operator_codes[opcode_index]),
            inputs=tuple(as_indices(getattr(operator, "inputs", None))),
            outputs=tuple(as_indices(getattr(operator, "outputs", None))),
            intermediates=tuple(as_indices(getattr(operator, "intermediates", None))),
            mutating_variable_inputs=tuple(
                bool(value)
                for value in as_list(getattr(operator, "mutatingVariableInputs", None))
            ),
            builtin_options_type=int(getattr(operator, "builtinOptionsType", 0) or 0),
            builtin_options_fingerprint=freeze_object(
                getattr(operator, "builtinOptions", None)
            ),
            builtin_options2_type=int(getattr(operator, "builtinOptions2Type", 0) or 0),
            builtin_options2_fingerprint=freeze_object(
                getattr(operator, "builtinOptions2", None)
            ),
            custom_options_format=int(getattr(operator, "customOptionsFormat", 0) or 0),
            custom_options_fingerprint=freeze_object(
                getattr(operator, "customOptions", None)
            ),
        )


@dataclass(frozen=True)
class TensorSnapshot:
    """Capture a tensor contract and the exact buffer object it references."""

    tensor_index: int
    tensor_fingerprint: FrozenValue
    name: str
    buffer_index: int
    contract: TensorContract
    buffer_fingerprint: FrozenValue

    @classmethod
    def capture(
        cls,
        document: CircleDocument,
        *,
        subgraph_index: int,
        tensor_index: int,
    ) -> TensorSnapshot:
        """Capture one tensor and its optional model-global storage buffer."""

        subgraph = document.subgraph(subgraph_index)
        tensors = as_list(getattr(subgraph, "tensors", None))
        if tensor_index < 0 or tensor_index >= len(tensors):
            raise IndexError(
                f"Tensor index {tensor_index} is outside 0..{len(tensors) - 1}."
            )
        tensor = tensors[tensor_index]
        buffer_index = int(getattr(tensor, "buffer", 0) or 0)
        buffers = as_list(getattr(document.model, "buffers", None))
        if buffer_index < 0 or buffer_index >= len(buffers):
            raise CircleRewriteError(
                f"Tensor {tensor_index} references invalid buffer {buffer_index}."
            )
        return cls(
            tensor_index=tensor_index,
            tensor_fingerprint=freeze_object(tensor),
            name=decode_text(getattr(tensor, "name", "")),
            buffer_index=buffer_index,
            contract=TensorContract.from_tensor(tensor),
            buffer_fingerprint=freeze_object(buffers[buffer_index]),
        )


@dataclass(frozen=True, kw_only=True)
class RewritePlan:
    """Capture immutable match evidence that must remain valid until application."""

    subgraph_index: int
    anchor: OperatorSnapshot
    tensors: tuple[TensorSnapshot, ...] = ()
    diagnostics: tuple[RewriteDiagnostic, ...] = ()
    _session_revision: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        document: CircleDocument,
        *,
        subgraph_index: int,
        anchor_operator_index: int,
        tensor_indices: Iterable[int] = (),
        diagnostics: Iterable[RewriteDiagnostic] = (),
        **plan_fields: Any,
    ) -> RewritePlan:
        """Create a plan from current graph state and optional subclass fields."""

        from tico.circle.session import active_optimization_session

        session = active_optimization_session(document.model)
        return cls(
            subgraph_index=int(subgraph_index),
            anchor=OperatorSnapshot.capture(
                document,
                subgraph_index=subgraph_index,
                operator_index=anchor_operator_index,
            ),
            tensors=tuple(
                TensorSnapshot.capture(
                    document,
                    subgraph_index=subgraph_index,
                    tensor_index=int(tensor_index),
                )
                for tensor_index in tensor_indices
            ),
            diagnostics=tuple(diagnostics),
            _session_revision=(None if session is None else session.revision),
            **plan_fields,
        )

    @property
    def anchor_operator_index(self) -> int:
        """Return the operator index that anchored the original match."""

        return self.anchor.operator_index

    def validate(self, document: CircleDocument) -> None:
        """Reject application when any captured operator, tensor, or buffer changed."""

        if self._session_revision is not None:
            from tico.circle.session import active_optimization_session

            session = active_optimization_session(document.model)
            if session is not None and session.revision == self._session_revision:
                return

        try:
            current_anchor = OperatorSnapshot.capture(
                document,
                subgraph_index=self.subgraph_index,
                operator_index=self.anchor.operator_index,
            )
        except (IndexError, CircleRewriteError) as error:
            raise CircleRewriteError(
                f"Rewrite plan for subgraph {self.subgraph_index}, operator "
                f"{self.anchor.operator_index} is stale because its anchor "
                "no longer exists."
            ) from error
        if current_anchor != self.anchor:
            raise CircleRewriteError(
                f"Rewrite plan for subgraph {self.subgraph_index}, operator "
                f"{self.anchor.operator_index} is stale because its anchor changed."
            )
        for expected in self.tensors:
            try:
                current = TensorSnapshot.capture(
                    document,
                    subgraph_index=self.subgraph_index,
                    tensor_index=expected.tensor_index,
                )
            except (IndexError, CircleRewriteError) as error:
                raise CircleRewriteError(
                    f"Rewrite plan for subgraph {self.subgraph_index}, operator "
                    f"{self.anchor.operator_index} is stale because tensor "
                    f"{expected.tensor_index} no longer exists."
                ) from error
            if current != expected:
                raise CircleRewriteError(
                    f"Rewrite plan for subgraph {self.subgraph_index}, operator "
                    f"{self.anchor.operator_index} is stale because tensor "
                    f"{expected.tensor_index} changed."
                )


@dataclass(frozen=True)
class RewriteApplication:
    """Report one rule application and its structured diagnostics."""

    changes: int
    diagnostics: tuple[RewriteDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Require a non-negative observable change count."""

        if self.changes < 0:
            raise ValueError("Rewrite application changes must not be negative.")

    @property
    def modified(self) -> bool:
        """Return whether the application changed model state."""

        return self.changes > 0


PlanT = TypeVar("PlanT", bound=RewritePlan)


class CircleRewriteRule(ABC, Generic[PlanT]):
    """Separate read-only pattern matching from validated graph mutation."""

    @property
    def name(self) -> str:
        """Return the stable rule name used in logs and diagnostics."""

        return self.__class__.__name__

    @abstractmethod
    def match(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        operator_index: int,
        context: CirclePassContext,
    ) -> PlanT | None:
        """Return an immutable plan when the operator matches this rule."""

    @abstractmethod
    def apply(
        self,
        document: CircleDocument,
        plan: PlanT,
        context: CirclePassContext,
    ) -> RewriteApplication:
        """Apply a previously validated plan and report observable changes."""


class CircleRulePass(CirclePass):
    """Run registered rewrite rules to a fixed point with restart scheduling."""

    def __init__(
        self,
        rules: Sequence[CircleRewriteRule[Any]],
        *,
        maximum_rewrites: int = 10_000,
    ) -> None:
        """Create a rule pass and reject empty or non-converging configurations."""

        self.rules = tuple(rules)
        self.maximum_rewrites = int(maximum_rewrites)
        if not self.rules:
            raise ValueError("CircleRulePass requires at least one rewrite rule.")
        if self.maximum_rewrites <= 0:
            raise ValueError("maximum_rewrites must be positive.")

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Apply one match, restart scanning, and repeat to a fixed point."""

        with context.activate(document):
            return self._run_active(document, context)

    def _run_active(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Run while session services are visible to builders and helpers."""

        total_changes = 0
        applications = 0
        diagnostics: list[RewriteDiagnostic] = []

        while True:
            applied = False
            for subgraph_index in range(document.subgraph_count):
                graph = context.graph(document, subgraph_index)
                operator_count = graph.operator_count
                for operator_index in range(operator_count):
                    for rule in self.rules:
                        plan = rule.match(
                            document,
                            graph,
                            operator_index,
                            context,
                        )
                        if plan is None:
                            continue
                        if plan.subgraph_index != subgraph_index or (
                            plan.anchor_operator_index != operator_index
                        ):
                            raise CircleRewriteError(
                                f"Rule {rule.name} returned a plan anchored at "
                                f"subgraph {plan.subgraph_index}, operator "
                                f"{plan.anchor_operator_index} while matching "
                                f"subgraph {subgraph_index}, operator "
                                f"{operator_index}."
                            )
                        if applications >= self.maximum_rewrites:
                            raise RuntimeError(
                                f"Circle rule pass exceeded {self.maximum_rewrites} "
                                "applications; a non-converging rule is suspected."
                            )

                        plan.validate(document)
                        with context.mutation(
                            document,
                            subgraph_index=subgraph_index,
                            plan=plan,
                        ) as mutation:
                            application = rule.apply(document, plan, context)
                            if not application.modified:
                                raise CircleRewriteError(
                                    f"Rule {rule.name} matched but reported no change."
                                )
                            mutation.commit()
                        applications += 1
                        total_changes += application.changes
                        diagnostics.extend(
                            diagnostic.for_rule(rule.name)
                            for diagnostic in (
                                *plan.diagnostics,
                                *application.diagnostics,
                            )
                        )
                        context.logger.debug(
                            "Applied Circle rewrite rule %s at subgraph %d, "
                            "operator %d with %d changes.",
                            rule.name,
                            subgraph_index,
                            operator_index,
                            application.changes,
                        )
                        applied = True
                        break
                    if applied:
                        break
                if applied:
                    break
            if not applied:
                break

        return CirclePassResult(
            modified=total_changes > 0,
            changes=total_changes,
            diagnostics=tuple(diagnostic.format() for diagnostic in diagnostics),
        )
