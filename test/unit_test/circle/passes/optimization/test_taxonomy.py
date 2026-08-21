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

import importlib
import unittest

import tico.circle.passes.optimization.fold as fold
import tico.circle.passes.optimization.simplify as simplify
from tico.circle.passes.optimization.canonicalize import CanonicalizeEquivalentOpsPass
from tico.circle.passes.optimization.compatibility import ResolveLegacyCustomOpsPass
from tico.circle.passes.optimization.legalize import LegalizeDynamicFullyConnectedPass
from tico.circle.passes.optimization.simplify import (
    EliminateIdentityOpsPass,
    SimplifyArithmeticPass,
)


class CirclePassTaxonomyTest(unittest.TestCase):
    """Keep one semantic package layout without forwarding compatibility shims."""

    def test_semantic_packages_export_expected_passes(self) -> None:
        """Expose canonicalization, simplification, compatibility, and legalization."""

        values = (
            CanonicalizeEquivalentOpsPass,
            EliminateIdentityOpsPass,
            SimplifyArithmeticPass,
            ResolveLegacyCustomOpsPass,
            LegalizeDynamicFullyConnectedPass,
        )
        self.assertTrue(all(isinstance(value, type) for value in values))
        self.assertFalse(hasattr(simplify, "CanonicalizeArithmeticPass"))
        self.assertFalse(hasattr(simplify, "RemoveNoOpOperatorsPass"))
        self.assertFalse(hasattr(fold, "FoldConstantSubgraphPass"))
        self.assertFalse(hasattr(fold, "FoldHeavyConstantSubgraphPass"))

    def test_former_module_paths_are_removed(self) -> None:
        """Reject imports through the temporary PR C forwarding packages."""

        modules = (
            "tico.circle.passes.optimization.canon",
            "tico.circle.passes.optimization.fusion",
            "tico.circle.passes.optimization.remove",
            "tico.circle.passes.optimization.fold.heavy",
        )
        for module_name in modules:
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
