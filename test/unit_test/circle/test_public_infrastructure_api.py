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
    CircleValueError,
    ConstantPool,
    TensorContract,
    TensorQuantization,
    TensorTypeRegistry,
    TensorTypeSpec,
    TensorValue,
    TensorValueCodec,
)
from tico.circle.passes import (
    ArithmeticCanonicalizationPolicy,
    CanonicalizeArithmeticPass,
    CanonicalizeEquivalentOpsPass,
    CircleRewriteRule,
    CircleRulePass,
    CompositeFusionPolicy,
    FloatingPointRewritePolicy,
    FuseCompositeOpsPass,
    FuseLinearOpsPass,
    LinearFusionPolicy,
    ReductionSimplificationPolicy,
    RemoveNoOpOperatorsPass,
    RewriteApplication,
    RewriteDiagnostic,
    RewritePlan,
    RewriteSeverity,
    SimplifyReductionOpsPass,
    SimplifyViewOpsPass,
)


class PublicInfrastructureApiTest(unittest.TestCase):
    """Keep common Circle infrastructure available through stable package imports."""

    def test_public_exports_resolve_to_classes(self):
        """Ensure all PR infrastructure symbols are importable from public packages."""

        values = (
            ArithmeticCanonicalizationPolicy,
            CanonicalizeArithmeticPass,
            CanonicalizeEquivalentOpsPass,
            CircleBuilder,
            CircleRewriteRule,
            CircleRulePass,
            CircleValueError,
            ConstantPool,
            CompositeFusionPolicy,
            FloatingPointRewritePolicy,
            FuseCompositeOpsPass,
            FuseLinearOpsPass,
            LinearFusionPolicy,
            ReductionSimplificationPolicy,
            RemoveNoOpOperatorsPass,
            RewriteApplication,
            RewriteDiagnostic,
            RewritePlan,
            RewriteSeverity,
            SimplifyReductionOpsPass,
            SimplifyViewOpsPass,
            TensorContract,
            TensorQuantization,
            TensorTypeRegistry,
            TensorTypeSpec,
            TensorValue,
            TensorValueCodec,
        )
        self.assertTrue(all(isinstance(value, type) for value in values))
        self.assertFalse(hasattr(circle_passes, "RemoveRedundantLayoutOpsPass"))


if __name__ == "__main__":
    unittest.main()
