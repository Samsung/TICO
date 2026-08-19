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
    ArithmeticCanonicalizationPolicy,
    CanonicalizeArithmeticPass,
    CanonicalizeEquivalentOpsPass,
    CommonSubexpressionEliminationPass,
    CommonSubexpressionEliminationPolicy,
    CompositeFusionPolicy,
    ConstantFoldPolicy,
    EliminateTransposeBoundedLayoutRegionPass,
    FloatingPointRewritePolicy,
    FoldConstantSubgraphPass,
    FuseCompositeOpsPass,
    FuseLinearOpsPass,
    LinearFusionPolicy,
    ReductionSimplificationPolicy,
    RemoveNoOpOperatorsPass,
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

__all__ = [
    "ArithmeticCanonicalizationPolicy",
    "CanonicalizeArithmeticPass",
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
    "CommonSubexpressionEliminationPolicy",
    "CompositeFusionPolicy",
    "ConstantFoldPolicy",
    "EliminateTransposeBoundedLayoutRegionPass",
    "FloatingPointRewritePolicy",
    "FoldConstantSubgraphPass",
    "FuseCompositeOpsPass",
    "FuseLinearOpsPass",
    "LinearFusionPolicy",
    "OperatorSnapshot",
    "ReductionSimplificationPolicy",
    "RemoveNoOpOperatorsPass",
    "RewriteApplication",
    "RewriteDiagnostic",
    "RewritePlan",
    "RewriteSeverity",
    "SimplifyReductionOpsPass",
    "SimplifyViewOpsPass",
    "TensorSnapshot",
    "create_o1_pipeline",
    "create_optimization_preset",
]
