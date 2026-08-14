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

import unittest
from dataclasses import dataclass

from tico.circle.errors import CircleRewriteError
from tico.circle.passes import CirclePassContext
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
    RewriteSeverity,
)

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    FakeBuffer,
    FakeOperator,
    FakeOperatorCode,
    FakeOptions,
    make_empty_document,
)


@dataclass(frozen=True, kw_only=True)
class MarkPlan(RewritePlan):
    """Carry the replacement opcode used by a synthetic rewrite rule."""

    replacement_opcode_index: int


class MarkOpcodeRule(CircleRewriteRule[MarkPlan]):
    """Replace opcode zero with opcode one for rule-runner tests."""

    def match(self, document, graph, operator_index, context):
        """Match synthetic operators whose opcode index is zero."""

        operator = graph.subgraph.operators[operator_index]
        if int(operator.opcodeIndex) != 0:
            return None
        return MarkPlan.capture(
            document,
            subgraph_index=graph.subgraph_index,
            anchor_operator_index=operator_index,
            tensor_indices=operator.inputs,
            diagnostics=(
                RewriteDiagnostic(
                    code="MATCHED_ZERO_OPCODE",
                    message="Synthetic opcode zero matched.",
                    object_path=(
                        f"subgraphs[{graph.subgraph_index}]"
                        f".operators[{operator_index}]"
                    ),
                ),
            ),
            replacement_opcode_index=1,
        )

    def apply(self, document, plan, context):
        """Replace the matched synthetic opcode index with one."""

        operator = document.subgraph(plan.subgraph_index).operators[
            plan.anchor_operator_index
        ]
        operator.opcodeIndex = plan.replacement_opcode_index
        return RewriteApplication(
            changes=1,
            diagnostics=(
                RewriteDiagnostic(
                    code="REPLACED_OPCODE",
                    severity=RewriteSeverity.INFO,
                    message="Synthetic opcode was replaced.",
                ),
            ),
        )


class NonMutatingRule(MarkOpcodeRule):
    """Report a matched plan without an observable mutation."""

    def apply(self, document, plan, context):
        """Intentionally report no mutation for convergence validation."""

        return RewriteApplication(changes=0)


class CircleRewriteRuleTest(unittest.TestCase):
    """Check plan validation, diagnostics, and fixed-point rule scheduling."""

    def _make_document(self, operator_count=1):
        """Create a graph with synthetic operators that all use opcode zero."""

        document = make_empty_document()
        tensor = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="input",
            shape=[1],
        )
        document.model.operatorCodes = [
            FakeOperatorCode(builtinCode=0),
            FakeOperatorCode(builtinCode=1),
        ]
        subgraph = document.subgraph()
        subgraph.inputs = [tensor]
        previous_tensor = tensor
        for operator_index in range(operator_count):
            output_tensor = add_runtime_tensor(
                document,
                subgraph_index=0,
                name=f"output_{operator_index}",
                shape=[1],
            )
            subgraph.operators.append(
                FakeOperator(
                    opcodeIndex=0,
                    inputs=[previous_tensor],
                    outputs=[output_tensor],
                )
            )
            previous_tensor = output_tensor
        subgraph.outputs = [previous_tensor]
        return document

    def test_rule_pass_restarts_until_all_matches_are_rewritten(self):
        """Rebuild graph indexes and restart scanning after every mutation."""

        document = self._make_document(operator_count=2)
        result = CircleRulePass([MarkOpcodeRule()]).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(result.changes, 2)
        self.assertEqual(
            [operator.opcodeIndex for operator in document.subgraph().operators],
            [1, 1],
        )
        self.assertEqual(len(result.diagnostics), 4)
        self.assertIn("MarkOpcodeRule", result.diagnostics[0])

    def test_plan_validation_rejects_changed_anchor_options(self):
        """Detect stale plans before applying mutation to a changed operator."""

        document = self._make_document()
        document.subgraph().operators[0].builtinOptions = FakeOptions(value=1)
        rule = MarkOpcodeRule()
        plan = rule.match(
            document,
            document.graph(0),
            0,
            CirclePassContext(verify_after_each_pass=False),
        )
        assert plan is not None
        document.subgraph().operators[0].builtinOptions.value = 2

        with self.assertRaisesRegex(CircleRewriteError, "anchor changed"):
            plan.validate(document)

    def test_plan_validation_detects_new_schema_operator_fields(self):
        """Fingerprint unmodeled operator fields to remain safe across schema growth."""

        document = self._make_document()
        plan = MarkPlan.capture(
            document,
            subgraph_index=0,
            anchor_operator_index=0,
            replacement_opcode_index=1,
        )
        document.subgraph().operators[0].largeCustomOptionsOffset = 16

        with self.assertRaisesRegex(CircleRewriteError, "anchor changed"):
            plan.validate(document)

    def test_plan_validation_reports_removed_anchor_as_stale(self):
        """Convert missing anchor indexes into a consistent rewrite error."""

        document = self._make_document()
        plan = MarkPlan.capture(
            document,
            subgraph_index=0,
            anchor_operator_index=0,
            replacement_opcode_index=1,
        )
        document.subgraph().operators.clear()

        with self.assertRaisesRegex(CircleRewriteError, "no longer exists"):
            plan.validate(document)

    def test_plan_validation_rejects_changed_captured_buffer(self):
        """Detect constant payload mutation even when tensor indices stay stable."""

        document = self._make_document()
        document.model.buffers.append(FakeBuffer(data=bytearray(b"before")))
        document.subgraph().tensors[0].buffer = 1
        plan = MarkPlan.capture(
            document,
            subgraph_index=0,
            anchor_operator_index=0,
            tensor_indices=(0,),
            replacement_opcode_index=1,
        )
        document.model.buffers[1].data[0] = ord("a")

        with self.assertRaisesRegex(CircleRewriteError, "tensor 0 changed"):
            plan.validate(document)

    def test_rule_pass_rejects_match_without_change(self):
        """Prevent an infinite fixed-point loop from a non-mutating matched rule."""

        document = self._make_document()
        with self.assertRaisesRegex(CircleRewriteError, "reported no change"):
            CircleRulePass([NonMutatingRule()]).run(
                document,
                CirclePassContext(verify_after_each_pass=False),
            )

    def test_rule_pass_enforces_application_limit_before_extra_mutation(self):
        """Stop a non-converging rule sequence before applying one rewrite too many."""

        document = self._make_document(operator_count=2)
        with self.assertRaisesRegex(RuntimeError, "exceeded 1 applications"):
            CircleRulePass([MarkOpcodeRule()], maximum_rewrites=1,).run(
                document,
                CirclePassContext(verify_after_each_pass=False),
            )
        self.assertEqual(
            [operator.opcodeIndex for operator in document.subgraph().operators],
            [1, 0],
        )

    def test_diagnostic_format_is_stable(self):
        """Format severity, code, rule, path, and message in one line."""

        diagnostic = RewriteDiagnostic(
            code="CODE",
            severity=RewriteSeverity.WARNING,
            rule_name="Rule",
            object_path="subgraphs[0]",
            message="Message.",
        )
        self.assertEqual(
            diagnostic.format(),
            "WARNING [CODE] Rule subgraphs[0]: Message.",
        )


if __name__ == "__main__":
    unittest.main()
