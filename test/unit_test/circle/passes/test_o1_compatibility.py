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

from tico.circle.passes import (
    create_o1_pipeline,
    create_optimization_preset,
    O1CompatibilityOptions,
)


class O1CompatibilityPresetTest(unittest.TestCase):
    """Check that heavy and compatibility transforms remain explicit O1 opt-ins."""

    def test_default_o1_sequence_is_unchanged(self) -> None:
        """Avoid enabling heavy or compatibility behavior in existing callers."""

        pipeline = create_o1_pipeline()
        names = [
            circle_pass.__class__.__name__
            for circle_pass in pipeline.phases[0].manager.passes
        ]

        self.assertEqual(
            names,
            [
                "CanonicalizeEquivalentOpsPass",
                "SimplifyViewOpsPass",
                "EliminateTransposeBoundedLayoutRegionPass",
                "SimplifyReductionOpsPass",
                "RemoveNoOpOperatorsPass",
                "CanonicalizeArithmeticPass",
                "FuseCompositeOpsPass",
                "FuseLinearOpsPass",
                "FoldConstantSubgraphPass",
                "CommonSubexpressionEliminationPass",
                "DeadCodeEliminationPass",
            ],
        )

    def test_all_compatibility_switches_extend_o1_in_stable_order(self) -> None:
        """Place recovery before canonicalization and heavy folding after fusion."""

        options = O1CompatibilityOptions(
            heavy_constant_folding=True,
            resolve_legacy_custom_ops=True,
            legalize_dynamic_fully_connected=True,
            fuse_transpose_conv_slice=True,
            fuse_legacy_fc_gelu_fc=True,
        )
        pipeline = create_optimization_preset("o1", compatibility=options)
        names = [
            circle_pass.__class__.__name__
            for circle_pass in pipeline.phases[0].manager.passes
        ]

        self.assertTrue(options.enabled)
        self.assertEqual(
            names,
            [
                "ResolveLegacyCustomOpsPass",
                "LegalizeDynamicFullyConnectedPass",
                "CanonicalizeEquivalentOpsPass",
                "SimplifyViewOpsPass",
                "EliminateTransposeBoundedLayoutRegionPass",
                "SimplifyReductionOpsPass",
                "RemoveNoOpOperatorsPass",
                "CanonicalizeArithmeticPass",
                "FuseLegacyFCGeluFCPass",
                "FuseCompositeOpsPass",
                "FuseTransposeConvSlicePass",
                "FuseLinearOpsPass",
                "FoldHeavyConstantSubgraphPass",
                "CommonSubexpressionEliminationPass",
                "DeadCodeEliminationPass",
            ],
        )

    def test_default_compatibility_options_are_disabled(self) -> None:
        """Expose a direct predicate for CLI validation without changing O1."""

        self.assertFalse(O1CompatibilityOptions().enabled)


if __name__ == "__main__":
    unittest.main()
