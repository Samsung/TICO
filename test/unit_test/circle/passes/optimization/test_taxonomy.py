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

import unittest

from tico.circle.passes.optimization.canonicalize import CanonicalizeEquivalentOpsPass
from tico.circle.passes.optimization.compatibility import ResolveLegacyCustomOpsPass
from tico.circle.passes.optimization.legalize import LegalizeDynamicFullyConnectedPass
from tico.circle.passes.optimization.simplify import (
    EliminateIdentityOpsPass,
    SimplifyArithmeticPass,
)
from tico.circle.passes.optimization.simplify.arithmetic import (
    CanonicalizeArithmeticPass as LegacyArithmetic,
)
from tico.circle.passes.optimization.simplify.identity_ops import (
    RemoveNoOpOperatorsPass as LegacyIdentity,
)


class CirclePassTaxonomyTest(unittest.TestCase):
    """Keep the semantic package layout and compatibility imports stable."""

    def test_semantic_packages_export_expected_passes(self) -> None:
        """Expose the new taxonomy through stable package imports."""

        self.assertEqual(
            CanonicalizeEquivalentOpsPass.__name__,
            "CanonicalizeEquivalentOpsPass",
        )
        self.assertEqual(
            EliminateIdentityOpsPass.__name__,
            "EliminateIdentityOpsPass",
        )
        self.assertEqual(
            SimplifyArithmeticPass.__name__,
            "SimplifyArithmeticPass",
        )
        self.assertEqual(
            ResolveLegacyCustomOpsPass.__name__,
            "ResolveLegacyCustomOpsPass",
        )
        self.assertEqual(
            LegalizeDynamicFullyConnectedPass.__name__,
            "LegalizeDynamicFullyConnectedPass",
        )
        self.assertIs(LegacyArithmetic, SimplifyArithmeticPass)
        self.assertIs(LegacyIdentity, EliminateIdentityOpsPass)

    def test_former_module_paths_forward_to_same_implementations(self) -> None:
        """Keep the former module paths as identity-preserving shims."""

        from tico.circle.passes.optimization.canon.dynamic_fully_connected import (
            LegalizeDynamicFullyConnectedPass as LegacyLegalize,
        )
        from tico.circle.passes.optimization.canon.equivalent_ops import (
            CanonicalizeEquivalentOpsPass as LegacyCanonicalize,
        )
        from tico.circle.passes.optimization.canon.legacy_custom_ops import (
            ResolveLegacyCustomOpsPass as LegacyResolve,
        )

        self.assertIs(LegacyCanonicalize, CanonicalizeEquivalentOpsPass)
        self.assertIs(LegacyResolve, ResolveLegacyCustomOpsPass)
        self.assertIs(LegacyLegalize, LegalizeDynamicFullyConnectedPass)


if __name__ == "__main__":
    unittest.main()
