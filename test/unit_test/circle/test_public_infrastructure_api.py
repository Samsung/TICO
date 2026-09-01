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

import tico.circle.passes as circle_passes
from tico.circle import (
    CircleBuilder,
    CircleMutationTransaction,
    CircleOptimizationSession,
    CircleOptimizationStatistics,
    CircleSessionRevision,
    CircleValueError,
    ConstantPool,
    TensorContract,
    TensorQuantization,
    TensorTypeRegistry,
    TensorTypeSpec,
    TensorValue,
    TensorValueCodec,
)


_EXPECTED_PASS_API = {
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
    "FuseActivationFunctionPass",
    "FuseCompositeOpsPass",
    "FuseLegacyFCGeluFCPass",
    "FuseLinearOpsPass",
    "FuseOutputRequantizePass",
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
}


class PublicInfrastructureApiTest(unittest.TestCase):
    """Keep the pass façade small while retaining infrastructure at its owner."""

    def test_pass_package_exports_only_supported_facade_symbols(self) -> None:
        """Expose scheduling, rewrite, pipeline, and semantic pass APIs only."""

        self.assertEqual(set(circle_passes.__all__), _EXPECTED_PASS_API)
        self.assertTrue(
            all(hasattr(circle_passes, name) for name in _EXPECTED_PASS_API)
        )

    def test_removed_compatibility_and_internal_symbols_are_absent(self) -> None:
        """Do not preserve temporary aliases or duplicate infrastructure exports."""

        removed = (
            "CanonicalizeArithmeticPass",
            "RemoveNoOpOperatorsPass",
            "FoldConstantSubgraphPass",
            "FoldHeavyConstantSubgraphPass",
            "O1CompatibilityOptions",
            "CircleMutationTransaction",
            "CircleOptimizationSession",
            "CircleOptimizationStatistics",
            "CircleSessionRevision",
            "OperatorSnapshot",
            "TensorSnapshot",
        )
        for name in removed:
            with self.subTest(name=name):
                self.assertFalse(hasattr(circle_passes, name))

    def test_circle_package_remains_the_owner_of_shared_infrastructure(self) -> None:
        """Keep builders, values, mutations, and sessions under tico.circle."""

        values = (
            CircleBuilder,
            CircleMutationTransaction,
            CircleOptimizationSession,
            CircleOptimizationStatistics,
            CircleSessionRevision,
            CircleValueError,
            ConstantPool,
            TensorContract,
            TensorQuantization,
            TensorTypeRegistry,
            TensorTypeSpec,
            TensorValue,
            TensorValueCodec,
        )
        self.assertTrue(all(isinstance(value, type) for value in values))


if __name__ == "__main__":
    unittest.main()
