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

from dataclasses import dataclass, field
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
    CanonicalizeEquivalentOpsPass,
    CommonSubexpressionEliminationPass,
    ConstantFoldingProfile,
    EliminateIdentityOpsPass,
    EliminateTransposeBoundedLayoutRegionPass,
    FoldConstantsPass,
    FuseCompositeOpsPass,
    FuseLegacyFCGeluFCPass,
    FuseLinearOpsPass,
    FuseTransposeConvSlicePass,
    LegalizeDynamicFullyConnectedPass,
    ResolveLegacyCustomOpsPass,
    SimplifyArithmeticPass,
    SimplifyReductionOpsPass,
    SimplifyViewOpsPass,
)


class CircleOptimizationPreset(str, Enum):
    """Name a supported built-in Circle optimization pipeline."""

    O1 = "o1"


@dataclass(frozen=True)
class O1OptimizationOptions:
    """Configure generic O1 optimization behavior."""

    constant_folding_profile: ConstantFoldingProfile = ConstantFoldingProfile.BASIC
    fuse_transpose_conv_slice: bool = False

    def __post_init__(self) -> None:
        """Normalize the profile enum and optional fusion switch."""

        object.__setattr__(
            self,
            "constant_folding_profile",
            ConstantFoldingProfile(self.constant_folding_profile),
        )
        object.__setattr__(
            self,
            "fuse_transpose_conv_slice",
            bool(self.fuse_transpose_conv_slice),
        )


@dataclass(frozen=True)
class O1LegalizationOptions:
    """Configure target-independent representation legalization."""

    dynamic_fully_connected: bool = False

    def __post_init__(self) -> None:
        """Normalize the legalization switch to a plain bool."""

        object.__setattr__(
            self,
            "dynamic_fully_connected",
            bool(self.dynamic_fully_connected),
        )

    @property
    def enabled(self) -> bool:
        """Return whether any legalization transform is enabled."""

        return self.dynamic_fully_connected


@dataclass(frozen=True)
class O1LegacyCompatibilityOptions:
    """Configure recovery of legacy ONE/TensorFlow graph patterns."""

    resolve_custom_ops: bool = False
    fuse_fc_gelu_fc: bool = False

    def __post_init__(self) -> None:
        """Normalize every compatibility switch to a plain bool."""

        object.__setattr__(
            self,
            "resolve_custom_ops",
            bool(self.resolve_custom_ops),
        )
        object.__setattr__(
            self,
            "fuse_fc_gelu_fc",
            bool(self.fuse_fc_gelu_fc),
        )

    @property
    def enabled(self) -> bool:
        """Return whether any legacy compatibility transform is enabled."""

        return self.resolve_custom_ops or self.fuse_fc_gelu_fc


@dataclass(frozen=True)
class O1PipelineOptions:
    """Group optimization, legalization, and compatibility domains."""

    optimization: O1OptimizationOptions = field(default_factory=O1OptimizationOptions)
    legalization: O1LegalizationOptions = field(default_factory=O1LegalizationOptions)
    compatibility: O1LegacyCompatibilityOptions = field(
        default_factory=O1LegacyCompatibilityOptions
    )


@dataclass(frozen=True)
class O1CompatibilityOptions:
    """Backward-compatible adapter for the former mixed option object."""

    heavy_constant_folding: bool = False
    resolve_legacy_custom_ops: bool = False
    legalize_dynamic_fully_connected: bool = False
    fuse_transpose_conv_slice: bool = False
    fuse_legacy_fc_gelu_fc: bool = False

    def __post_init__(self) -> None:
        """Normalize every legacy adapter switch to a plain bool."""

        for field_name in (
            "heavy_constant_folding",
            "resolve_legacy_custom_ops",
            "legalize_dynamic_fully_connected",
            "fuse_transpose_conv_slice",
            "fuse_legacy_fc_gelu_fc",
        ):
            object.__setattr__(
                self,
                field_name,
                bool(getattr(self, field_name)),
            )

    @property
    def enabled(self) -> bool:
        """Return whether at least one former switch is selected."""

        return any(
            (
                self.heavy_constant_folding,
                self.resolve_legacy_custom_ops,
                self.legalize_dynamic_fully_connected,
                self.fuse_transpose_conv_slice,
                self.fuse_legacy_fc_gelu_fc,
            )
        )

    def to_pipeline_options(self) -> O1PipelineOptions:
        """Translate legacy switches into the separated option domains."""

        profile = (
            ConstantFoldingProfile.HEAVY
            if self.heavy_constant_folding
            else ConstantFoldingProfile.BASIC
        )
        return O1PipelineOptions(
            optimization=O1OptimizationOptions(
                constant_folding_profile=profile,
                fuse_transpose_conv_slice=self.fuse_transpose_conv_slice,
            ),
            legalization=O1LegalizationOptions(
                dynamic_fully_connected=(self.legalize_dynamic_fully_connected)
            ),
            compatibility=O1LegacyCompatibilityOptions(
                resolve_custom_ops=self.resolve_legacy_custom_ops,
                fuse_fc_gelu_fc=self.fuse_legacy_fc_gelu_fc,
            ),
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
        """Run phases in order and aggregate their execution records."""

        context = context or CirclePassContext()
        results = tuple(
            CirclePassPipelinePhaseResult(
                phase.name,
                phase.manager.run(document, context),
            )
            for phase in self.phases
        )
        return CirclePassPipelineResult(results)


def _selected_o1_options(
    options: O1PipelineOptions | None,
    compatibility: O1CompatibilityOptions | None,
) -> O1PipelineOptions:
    """Select native options or translate the legacy adapter."""

    if options is not None and compatibility is not None:
        raise ValueError(
            "Specify either options or the legacy compatibility adapter, " "not both."
        )
    if options is not None:
        return options
    if compatibility is not None:
        return compatibility.to_pipeline_options()
    return O1PipelineOptions()


def _phase(
    name: str,
    passes: Iterable[CirclePass],
    *,
    strategy: CirclePassStrategy,
    maximum_steps: int = 1000,
) -> CirclePassPipelinePhase:
    """Create one named manager phase from a pass sequence."""

    return CirclePassPipelinePhase(
        name,
        CirclePassManager(
            tuple(passes),
            strategy=strategy,
            maximum_steps=maximum_steps,
        ),
    )


def create_o1_pipeline(
    *,
    maximum_steps: int = 1000,
    options: O1PipelineOptions | None = None,
    compatibility: O1CompatibilityOptions | None = None,
) -> CirclePassPipeline:
    """Create O1 with explicitly separated transformation domains."""

    selected = _selected_o1_options(options, compatibility)
    phases: list[CirclePassPipelinePhase] = []

    compatibility_passes: list[CirclePass] = []
    if selected.compatibility.resolve_custom_ops:
        compatibility_passes.append(ResolveLegacyCustomOpsPass())
    if compatibility_passes:
        phases.append(
            _phase(
                "compatibility",
                compatibility_passes,
                strategy=CirclePassStrategy.ONCE,
            )
        )

    legalization_passes: list[CirclePass] = []
    if selected.legalization.dynamic_fully_connected:
        legalization_passes.append(LegalizeDynamicFullyConnectedPass())
    if legalization_passes:
        phases.append(
            _phase(
                "legalize",
                legalization_passes,
                strategy=CirclePassStrategy.ONCE,
            )
        )

    optimization_passes: list[CirclePass] = [
        CanonicalizeEquivalentOpsPass(),
        SimplifyViewOpsPass(),
        EliminateTransposeBoundedLayoutRegionPass(),
        SimplifyReductionOpsPass(),
        EliminateIdentityOpsPass(),
        SimplifyArithmeticPass(),
    ]
    # Keep the legacy FC-GELU-FC recognizer at its former semantic
    # position. It is a compatibility-controlled transform, but moving
    # it before canonicalization would change which patterns it sees.
    if selected.compatibility.fuse_fc_gelu_fc:
        optimization_passes.append(FuseLegacyFCGeluFCPass())
    optimization_passes.append(FuseCompositeOpsPass())
    if selected.optimization.fuse_transpose_conv_slice:
        optimization_passes.append(FuseTransposeConvSlicePass())
    optimization_passes.extend(
        (
            FuseLinearOpsPass(),
            FoldConstantsPass(profile=selected.optimization.constant_folding_profile),
            CommonSubexpressionEliminationPass(),
            DeadCodeEliminationPass(),
        )
    )
    phases.append(
        _phase(
            "optimize",
            optimization_passes,
            strategy=CirclePassStrategy.RESTART,
            maximum_steps=maximum_steps,
        )
    )
    phases.append(
        _phase(
            "compact",
            (CompactIndicesPass(),),
            strategy=CirclePassStrategy.ONCE,
        )
    )
    return CirclePassPipeline(phases)


def create_optimization_preset(
    preset: CircleOptimizationPreset | str,
    *,
    maximum_steps: int = 1000,
    options: O1PipelineOptions | None = None,
    compatibility: O1CompatibilityOptions | None = None,
) -> CirclePassPipeline:
    """Create a supported optimization preset by enum or CLI value."""

    selected = CircleOptimizationPreset(preset)
    if selected is CircleOptimizationPreset.O1:
        return create_o1_pipeline(
            maximum_steps=maximum_steps,
            options=options,
            compatibility=compatibility,
        )
    raise AssertionError(f"Unhandled Circle optimization preset: {selected.value}.")
