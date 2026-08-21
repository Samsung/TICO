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

from tico.circle.mutation import CircleMutationTransaction
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.manager import (
    CirclePassExecution,
    CirclePassManager,
    CirclePassManagerResult,
    CirclePassStrategy,
)
from tico.circle.passes.optimization import (
    ArithmeticCanonicalizationPolicy,
    CanonicalizeArithmeticPass,
    CanonicalizeEquivalentOpsPass,
    CommonSubexpressionEliminationPass,
    CommonSubexpressionEliminationPolicy,
    CompositeFusionPolicy,
    ConstantFoldPolicy,
    CustomOptionDecoder,
    DynamicFullyConnectedLegalizationPolicy,
    EliminateTransposeBoundedLayoutRegionPass,
    FloatingPointRewritePolicy,
    FoldConstantSubgraphPass,
    FoldHeavyConstantSubgraphPass,
    FuseCompositeOpsPass,
    FuseLegacyFCGeluFCPass,
    FuseLinearOpsPass,
    FuseTransposeConvSlicePass,
    heavy_constant_evaluator_registry,
    HeavyConstantEvaluatorPolicy,
    LegacyCustomOpPolicy,
    LegacyFCGeluFCFusionPolicy,
    LegalizeDynamicFullyConnectedPass,
    LinearFusionPolicy,
    ReductionSimplificationPolicy,
    RemoveNoOpOperatorsPass,
    ResolveLegacyCustomOpsPass,
    SimplifyReductionOpsPass,
    SimplifyViewOpsPass,
    TransposeConvSliceFusionPolicy,
)
from tico.circle.passes.presets import (
    CircleOptimizationPreset,
    CirclePassPipeline,
    CirclePassPipelinePhase,
    CirclePassPipelinePhaseResult,
    CirclePassPipelineResult,
    create_o1_pipeline,
    create_optimization_preset,
    O1CompatibilityOptions,
)
from tico.circle.passes.rules import (
    CircleRewriteRule,
    CircleRulePass,
    OperatorSnapshot,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
    RewriteSeverity,
    TensorSnapshot,
)
from tico.circle.session import (
    active_optimization_session,
    CircleOptimizationSession,
    CircleOptimizationStatistics,
    CircleSessionRevision,
    existing_optimization_session,
    optimization_session_for,
)

__all__ = [
    "ArithmeticCanonicalizationPolicy",
    "CanonicalizeArithmeticPass",
    "CanonicalizeEquivalentOpsPass",
    "CircleOptimizationPreset",
    "CirclePass",
    "CircleMutationTransaction",
    "CircleOptimizationSession",
    "CircleOptimizationStatistics",
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
    "CircleSessionRevision",
    "CircleRewriteRule",
    "CircleRulePass",
    "CommonSubexpressionEliminationPass",
    "CommonSubexpressionEliminationPolicy",
    "CompositeFusionPolicy",
    "ConstantFoldPolicy",
    "CustomOptionDecoder",
    "DynamicFullyConnectedLegalizationPolicy",
    "EliminateTransposeBoundedLayoutRegionPass",
    "FloatingPointRewritePolicy",
    "FoldConstantSubgraphPass",
    "FoldHeavyConstantSubgraphPass",
    "FuseCompositeOpsPass",
    "FuseLegacyFCGeluFCPass",
    "FuseLinearOpsPass",
    "FuseTransposeConvSlicePass",
    "HeavyConstantEvaluatorPolicy",
    "LegacyCustomOpPolicy",
    "LegacyFCGeluFCFusionPolicy",
    "LegalizeDynamicFullyConnectedPass",
    "LinearFusionPolicy",
    "O1CompatibilityOptions",
    "OperatorSnapshot",
    "ReductionSimplificationPolicy",
    "RemoveNoOpOperatorsPass",
    "ResolveLegacyCustomOpsPass",
    "RewriteApplication",
    "RewriteDiagnostic",
    "RewritePlan",
    "RewriteSeverity",
    "SimplifyReductionOpsPass",
    "SimplifyViewOpsPass",
    "TensorSnapshot",
    "TransposeConvSliceFusionPolicy",
    "create_o1_pipeline",
    "create_optimization_preset",
    "active_optimization_session",
    "existing_optimization_session",
    "heavy_constant_evaluator_registry",
    "optimization_session_for",
]
