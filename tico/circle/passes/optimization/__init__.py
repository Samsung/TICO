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


from tico.circle.passes.optimization.canonicalize import CanonicalizeEquivalentOpsPass
from tico.circle.passes.optimization.compatibility import (
    CustomOptionDecoder,
    FuseLegacyFCGeluFCPass,
    LegacyCustomOpPolicy,
    LegacyFCGeluFCFusionPolicy,
    ResolveLegacyCustomOpsPass,
)
from tico.circle.passes.optimization.cse import (
    CommonSubexpressionEliminationPass,
    CommonSubexpressionEliminationPolicy,
)
from tico.circle.passes.optimization.fold import (
    ConstantFoldingProfile,
    ConstantFoldPolicy,
    FoldConstantsPass,
    FoldConstantSubgraphPass,
    FoldHeavyConstantSubgraphPass,
    heavy_constant_evaluator_registry,
    HeavyConstantEvaluatorPolicy,
)
from tico.circle.passes.optimization.fuse import (
    CompositeFusionPolicy,
    FuseCompositeOpsPass,
    FuseLinearOpsPass,
    FuseTransposeConvSlicePass,
    LinearFusionPolicy,
    TransposeConvSliceFusionPolicy,
)
from tico.circle.passes.optimization.legalize import (
    DynamicFullyConnectedLegalizationPolicy,
    LegalizeDynamicFullyConnectedPass,
)
from tico.circle.passes.optimization.policy import FloatingPointRewritePolicy
from tico.circle.passes.optimization.simplify import (
    ArithmeticCanonicalizationPolicy,
    CanonicalizeArithmeticPass,
    EliminateIdentityOpsPass,
    EliminateTransposeBoundedLayoutRegionPass,
    ReductionSimplificationPolicy,
    RemoveNoOpOperatorsPass,
    SimplifyArithmeticPass,
    SimplifyReductionOpsPass,
    SimplifyViewOpsPass,
)

__all__ = [
    "ArithmeticCanonicalizationPolicy",
    "CanonicalizeArithmeticPass",
    "CanonicalizeEquivalentOpsPass",
    "CommonSubexpressionEliminationPass",
    "CommonSubexpressionEliminationPolicy",
    "CompositeFusionPolicy",
    "ConstantFoldPolicy",
    "ConstantFoldingProfile",
    "CustomOptionDecoder",
    "DynamicFullyConnectedLegalizationPolicy",
    "EliminateIdentityOpsPass",
    "EliminateTransposeBoundedLayoutRegionPass",
    "FloatingPointRewritePolicy",
    "FoldConstantsPass",
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
    "ReductionSimplificationPolicy",
    "RemoveNoOpOperatorsPass",
    "ResolveLegacyCustomOpsPass",
    "SimplifyArithmeticPass",
    "SimplifyReductionOpsPass",
    "SimplifyViewOpsPass",
    "TransposeConvSliceFusionPolicy",
    "heavy_constant_evaluator_registry",
]
