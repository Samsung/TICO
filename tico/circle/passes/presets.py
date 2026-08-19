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
from enum import Enum
from typing import Iterable

from tico.circle.document import CircleDocument
from tico.circle.passes.base import CirclePass, CirclePassContext
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass
from tico.circle.passes.manager import (
    CirclePassExecution,
    CirclePassManager,
    CirclePassManagerResult,
    CirclePassStrategy,
)
from tico.circle.passes.optimization import (
    CanonicalizeArithmeticPass,
    CanonicalizeEquivalentOpsPass,
    CommonSubexpressionEliminationPass,
    EliminateTransposeBoundedLayoutRegionPass,
    FoldConstantSubgraphPass,
    FoldHeavyConstantSubgraphPass,
    FuseCompositeOpsPass,
    FuseLegacyFCGeluFCPass,
    FuseLinearOpsPass,
    FuseTransposeConvSlicePass,
    LegalizeDynamicFullyConnectedPass,
    RemoveNoOpOperatorsPass,
    ResolveLegacyCustomOpsPass,
    SimplifyReductionOpsPass,
    SimplifyViewOpsPass,
)


class CircleOptimizationPreset(str, Enum):
    """Name a supported built-in Circle optimization pipeline."""

    O1 = "o1"


@dataclass(frozen=True)
class O1CompatibilityOptions:
    """Select opt-in heavy and legacy transformations for the O1 pipeline."""

    heavy_constant_folding: bool = False
    resolve_legacy_custom_ops: bool = False
    legalize_dynamic_fully_connected: bool = False
    fuse_transpose_conv_slice: bool = False
    fuse_legacy_fc_gelu_fc: bool = False

    def __post_init__(self) -> None:
        """Normalize every compatibility switch to a plain bool."""

        for field_name in (
            "heavy_constant_folding",
            "resolve_legacy_custom_ops",
            "legalize_dynamic_fully_connected",
            "fuse_transpose_conv_slice",
            "fuse_legacy_fc_gelu_fc",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))

    @property
    def enabled(self) -> bool:
        """Return whether at least one optional transformation is selected."""

        return any(
            (
                self.heavy_constant_folding,
                self.resolve_legacy_custom_ops,
                self.legalize_dynamic_fully_connected,
                self.fuse_transpose_conv_slice,
                self.fuse_legacy_fc_gelu_fc,
            )
        )


@dataclass(frozen=True)
class CirclePassPipelinePhase:
    """Pair a stable phase name with one configured pass manager."""

    name: str
    manager: CirclePassManager

    def __post_init__(self) -> None:
        """Reject unnamed phases."""

        if not self.name:
            raise ValueError("Circle pass pipeline phase names must not be empty.")


@dataclass(frozen=True)
class CirclePassPipelinePhaseResult:
    """Record the result of one named pipeline phase."""

    name: str
    result: CirclePassManagerResult


@dataclass(frozen=True)
class CirclePassPipelineResult:
    """Collect pass-manager results from all pipeline phases."""

    phases: tuple[CirclePassPipelinePhaseResult, ...]

    @property
    def modified(self) -> bool:
        """Return whether any phase changed the document."""

        return any(phase.result.modified for phase in self.phases)

    @property
    def changes(self) -> int:
        """Return the sum of changes reported by every phase."""

        return sum(phase.result.changes for phase in self.phases)

    @property
    def executions(self) -> tuple[CirclePassExecution, ...]:
        """Return all pass invocations in complete pipeline order."""

        return tuple(
            execution for phase in self.phases for execution in phase.result.executions
        )


class CirclePassPipeline:
    """Run multiple pass-manager phases while preserving phase boundaries."""

    def __init__(self, phases: Iterable[CirclePassPipelinePhase]) -> None:
        """Create a non-empty pipeline with unique phase names."""

        self.phases = tuple(phases)
        if not self.phases:
            raise ValueError("CirclePassPipeline requires at least one phase.")
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("Circle pass pipeline phase names must be unique.")

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext | None = None,
    ) -> CirclePassPipelineResult:
        """Run phases in order and return an aggregated execution record."""

        context = context or CirclePassContext()
        results = tuple(
            CirclePassPipelinePhaseResult(
                phase.name,
                phase.manager.run(document, context),
            )
            for phase in self.phases
        )
        return CirclePassPipelineResult(results)


def create_o1_pipeline(
    *,
    maximum_steps: int = 1000,
    compatibility: O1CompatibilityOptions | None = None,
) -> CirclePassPipeline:
    """Create O1 with optional heavy compatibility and one-shot compaction."""

    selected = compatibility or O1CompatibilityOptions()
    optimization_passes: list[CirclePass] = []
    if selected.resolve_legacy_custom_ops:
        optimization_passes.append(ResolveLegacyCustomOpsPass())
    if selected.legalize_dynamic_fully_connected:
        optimization_passes.append(LegalizeDynamicFullyConnectedPass())

    optimization_passes.extend(
        (
            CanonicalizeEquivalentOpsPass(),
            SimplifyViewOpsPass(),
            EliminateTransposeBoundedLayoutRegionPass(),
            SimplifyReductionOpsPass(),
            RemoveNoOpOperatorsPass(),
            CanonicalizeArithmeticPass(),
        )
    )
    if selected.fuse_legacy_fc_gelu_fc:
        optimization_passes.append(FuseLegacyFCGeluFCPass())
    optimization_passes.append(FuseCompositeOpsPass())
    if selected.fuse_transpose_conv_slice:
        optimization_passes.append(FuseTransposeConvSlicePass())
    optimization_passes.extend(
        (
            FuseLinearOpsPass(),
            (
                FoldHeavyConstantSubgraphPass()
                if selected.heavy_constant_folding
                else FoldConstantSubgraphPass()
            ),
            CommonSubexpressionEliminationPass(),
            DeadCodeEliminationPass(),
        )
    )

    optimization = CirclePassManager(
        optimization_passes,
        strategy=CirclePassStrategy.RESTART,
        maximum_steps=maximum_steps,
    )
    finalization = CirclePassManager(
        (CompactIndicesPass(),),
        strategy=CirclePassStrategy.ONCE,
    )
    return CirclePassPipeline(
        (
            CirclePassPipelinePhase("optimize", optimization),
            CirclePassPipelinePhase("compact", finalization),
        )
    )


def create_optimization_preset(
    preset: CircleOptimizationPreset | str,
    *,
    maximum_steps: int = 1000,
    compatibility: O1CompatibilityOptions | None = None,
) -> CirclePassPipeline:
    """Create a supported optimization preset by enum or CLI value."""

    selected = CircleOptimizationPreset(preset)
    if selected is CircleOptimizationPreset.O1:
        return create_o1_pipeline(
            maximum_steps=maximum_steps,
            compatibility=compatibility,
        )
    raise AssertionError(f"Unhandled Circle optimization preset: {selected.value}.")
