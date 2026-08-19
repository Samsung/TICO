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

from collections.abc import Mapping
from dataclasses import dataclass

from tico.circle.analysis import (
    build_expression_key,
    ExpressionKey,
    OperatorPurityAnalysis,
)
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list, CircleGraph, OPTIONAL_TENSOR_INDEX
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization._utils import (
    operator_builtin_code,
    operator_version,
    OptimizationSchemaResolver,
)
from tico.circle.rewrite import replace_tensor_uses_many, TensorUseReplacement


_DEFAULT_IMPURE_BUILTIN_NAMES = (
    "ASSIGN_VARIABLE",
    "BIDIRECTIONAL_SEQUENCE_LSTM",
    "BIDIRECTIONAL_SEQUENCE_RNN",
    "CALL",
    "CALL_ONCE",
    "DELEGATE",
    "HASHTABLE",
    "HASHTABLE_FIND",
    "HASHTABLE_IMPORT",
    "HASHTABLE_LOOKUP",
    "HASHTABLE_SIZE",
    "IF",
    "LSTM",
    "MULTINOMIAL",
    "RANDOM_STANDARD_NORMAL",
    "RANDOM_UNIFORM",
    "READ_VARIABLE",
    "RESOURCE_GATHER",
    "RESOURCE_SCATTER_ADD",
    "RNN",
    "STABLEHLO_CASE",
    "STABLEHLO_CUSTOM_CALL",
    "STABLEHLO_RNG_BIT_GENERATOR",
    "STABLEHLO_WHILE",
    "SVDF",
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "UNIDIRECTIONAL_SEQUENCE_RNN",
    "VAR_HANDLE",
    "WHILE",
)


@dataclass(frozen=True)
class CommonSubexpressionEliminationPolicy:
    """Control conservative eligibility and convergence limits for Circle CSE."""

    allow_custom_operators: bool = False
    impure_builtin_names: tuple[str, ...] = _DEFAULT_IMPURE_BUILTIN_NAMES
    maximum_rounds: int = 1000

    def __post_init__(self) -> None:
        """Normalize builtin names and reject an invalid convergence limit."""

        normalized = tuple(
            dict.fromkeys(
                str(name).strip().upper() for name in self.impure_builtin_names
            )
        )
        if any(not name for name in normalized):
            raise ValueError("impure_builtin_names must not contain empty names.")
        if int(self.maximum_rounds) <= 0:
            raise ValueError("maximum_rounds must be positive.")
        object.__setattr__(self, "impure_builtin_names", normalized)
        object.__setattr__(
            self,
            "allow_custom_operators",
            bool(self.allow_custom_operators),
        )
        object.__setattr__(self, "maximum_rounds", int(self.maximum_rounds))


@dataclass(frozen=True)
class _ExpressionRecord:
    """Record the first operator and output tensors for one expression key."""

    operator_index: int
    output_tensor_indices: tuple[int, ...]


class CommonSubexpressionEliminationPass(CirclePass):
    """Reuse outputs of structurally identical pure operators.

    The pass compares operator kind and version, ordered canonical input tensor
    identities, builtin and custom options, scratch-tensor contracts, and complete
    output tensor contracts. It rewires duplicate outputs to the first matching
    expression but does not delete operators or tensors. Schedule dead-code
    elimination and index compaction afterward.

    Duplicate operators that directly produce a subgraph output are preserved so CSE
    does not change externally visible output tensor identities or names.
    """

    def __init__(
        self,
        *,
        builtin_codes: Mapping[str, int] | None = None,
        policy: CommonSubexpressionEliminationPolicy | None = None,
    ) -> None:
        """Create CSE with optional schema-independent enum overrides."""

        self.policy = policy or CommonSubexpressionEliminationPolicy()
        resolver = OptimizationSchemaResolver(builtin_codes=builtin_codes)
        impure_codes = {
            code
            for name in self.policy.impure_builtin_names
            if (code := resolver.maybe_builtin_code(name)) is not None
        }
        self._purity = OperatorPurityAnalysis(
            impure_builtin_codes=impure_codes,
            custom_builtin_code=resolver.maybe_builtin_code("CUSTOM"),
            allow_custom_operators=self.policy.allow_custom_operators,
        )

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Replace duplicate operator-output uses to a fixed point."""

        duplicate_count = 0
        remapped_references = 0
        diagnostics: list[str] = []

        for subgraph_index in range(document.subgraph_count):
            rounds = 0
            while True:
                if rounds >= self.policy.maximum_rounds:
                    raise RuntimeError(
                        "Common-subexpression elimination exceeded "
                        f"{self.policy.maximum_rounds} rounds in subgraph "
                        f"{subgraph_index}; a non-converging rewrite is suspected."
                    )
                graph = document.graph(subgraph_index)
                graph_outputs = set(graph.outputs)
                expressions: dict[ExpressionKey, _ExpressionRecord] = {}
                representatives: dict[int, int] = {}
                replacements: list[TensorUseReplacement] = []
                duplicate_records: list[
                    tuple[int, _ExpressionRecord, tuple[int, ...]]
                ] = []

                for operator_index, operator in enumerate(
                    as_list(getattr(graph.subgraph, "operators", None))
                ):
                    builtin_code = operator_builtin_code(document.model, operator)
                    if (
                        not self.policy.allow_custom_operators
                        and _operator_has_custom_code(document.model, operator)
                    ):
                        continue
                    if not self._purity.is_pure(
                        graph,
                        operator_index,
                        builtin_code=builtin_code,
                    ):
                        continue
                    outputs = tuple(as_indices(getattr(operator, "outputs", None)))
                    canonical_inputs = tuple(
                        _representative(representatives, tensor_index)
                        if tensor_index != OPTIONAL_TENSOR_INDEX
                        else tensor_index
                        for tensor_index in as_indices(
                            getattr(operator, "inputs", None)
                        )
                    )
                    key = build_expression_key(
                        document.model,
                        graph,
                        operator_index,
                        builtin_code=builtin_code,
                        operator_version=operator_version(document.model, operator),
                        input_tensor_indices=canonical_inputs,
                    )
                    canonical = expressions.get(key)
                    if canonical is None:
                        expressions[key] = _ExpressionRecord(operator_index, outputs)
                        continue
                    if len(canonical.output_tensor_indices) != len(outputs):
                        raise CircleRewriteError(
                            "Equal CSE expression keys produced different output "
                            "counts."
                        )

                    # Keep graph-output identities stable. The operator remains a valid
                    # canonical source for later internal duplicates, but a duplicate
                    # graph output itself is not bypassed.
                    if any(tensor_index in graph_outputs for tensor_index in outputs):
                        continue
                    if not _outputs_have_rewritable_uses(graph, outputs):
                        continue

                    canonical_outputs = tuple(
                        _representative(representatives, tensor_index)
                        for tensor_index in canonical.output_tensor_indices
                    )
                    for old_tensor, new_tensor in zip(outputs, canonical_outputs):
                        representatives[old_tensor] = new_tensor
                    replacements.extend(
                        TensorUseReplacement(old_tensor, new_tensor)
                        for old_tensor, new_tensor in zip(outputs, canonical_outputs)
                        if old_tensor != new_tensor
                    )
                    duplicate_records.append(
                        (
                            operator_index,
                            _ExpressionRecord(
                                canonical.operator_index,
                                canonical_outputs,
                            ),
                            outputs,
                        )
                    )

                if not replacements:
                    break
                stats = replace_tensor_uses_many(
                    document.model,
                    subgraph_index=subgraph_index,
                    replacements=replacements,
                )
                if not stats.modified:
                    raise CircleRewriteError(
                        "CSE selected duplicate expressions but replaced no tensor "
                        "uses."
                    )

                rounds += 1
                duplicate_count += len(duplicate_records)
                remapped_references += stats.remapped_references
                for operator_index, canonical, outputs in duplicate_records:
                    diagnostics.append(
                        f"Subgraph {subgraph_index}: reused outputs "
                        f"{list(canonical.output_tensor_indices)} from operator "
                        f"{canonical.operator_index} for duplicate operator "
                        f"{operator_index} outputs {list(outputs)}."
                    )

        context.logger.debug(
            "Common-subexpression elimination found %d duplicates and remapped "
            "%d references.",
            duplicate_count,
            remapped_references,
        )
        return CirclePassResult(
            modified=duplicate_count > 0,
            changes=duplicate_count,
            diagnostics=tuple(diagnostics),
        )


def _operator_has_custom_code(model: object, operator: object) -> bool:
    """Return whether an operator-code record carries a custom identifier."""

    operator_codes = as_list(getattr(model, "operatorCodes", None))
    opcode_index = int(getattr(operator, "opcodeIndex", -1))
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        raise CircleRewriteError(
            f"Operator references invalid opcode index {opcode_index}."
        )
    custom_code = getattr(operator_codes[opcode_index], "customCode", None)
    return custom_code not in (None, "", b"")


def _representative(representatives: dict[int, int], tensor_index: int) -> int:
    """Return the terminal representative for one tensor without path mutation."""

    current = int(tensor_index)
    visited: set[int] = set()
    while current in representatives:
        if current in visited:
            raise CircleRewriteError("CSE tensor representatives contain a cycle.")
        visited.add(current)
        current = representatives[current]
    return current


def _outputs_have_rewritable_uses(
    graph: CircleGraph,
    output_tensor_indices: tuple[int, ...],
) -> bool:
    """Return whether at least one output is consumed by another operator."""

    return any(
        bool(graph.consumers(tensor_index)) for tensor_index in output_tensor_indices
    )
