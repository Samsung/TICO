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

from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
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
from tico.circle.passes.presets import (
    CircleOptimizationPreset,
    CirclePassPipeline,
    CirclePassPipelinePhase,
    CirclePassPipelinePhaseResult,
    CirclePassPipelineResult,
    create_o1_pipeline,
    create_optimization_preset,
    O1LegacyCompatibilityOptions,
    O1LegalizationOptions,
    O1OptimizationOptions,
    O1PipelineOptions,
)
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
    RewriteSeverity,
)

__all__ = [
    "CanonicalizeEquivalentOpsPass",
    "CircleOptimizationPreset",
    "CirclePass",
    "CirclePassContext",
    "CirclePassExecution",
    "CirclePassManager",
    "CirclePassManagerResult",
    "CirclePassPipeline",
    "CirclePassPipelinePhase",
    "CirclePassPipelinePhaseResult",
    "CirclePassPipelineResult",
    "CirclePassResult",
    "CirclePassStrategy",
    "CircleRewriteRule",
    "CircleRulePass",
    "CommonSubexpressionEliminationPass",
    "ConstantFoldingProfile",
    "EliminateIdentityOpsPass",
    "EliminateTransposeBoundedLayoutRegionPass",
    "FoldConstantsPass",
    "FuseCompositeOpsPass",
    "FuseLegacyFCGeluFCPass",
    "FuseLinearOpsPass",
    "FuseTransposeConvSlicePass",
    "LegalizeDynamicFullyConnectedPass",
    "O1LegacyCompatibilityOptions",
    "O1LegalizationOptions",
    "O1OptimizationOptions",
    "O1PipelineOptions",
    "ResolveLegacyCustomOpsPass",
    "RewriteApplication",
    "RewriteDiagnostic",
    "RewritePlan",
    "RewriteSeverity",
    "SimplifyArithmeticPass",
    "SimplifyReductionOpsPass",
    "SimplifyViewOpsPass",
    "create_o1_pipeline",
    "create_optimization_preset",
]
