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
    ConstantFoldingProfile,
    create_o1_pipeline,
    create_optimization_preset,
    FoldConstantsPass,
    O1CompatibilityOptions,
    O1LegacyCompatibilityOptions,
    O1LegalizationOptions,
    O1OptimizationOptions,
    O1PipelineOptions,
)


def _phase_pass_names(pipeline, phase_name: str) -> list[str]:
    phase = next(phase for phase in pipeline.phases if phase.name == phase_name)
    return [circle_pass.__class__.__name__ for circle_pass in phase.manager.passes]


class O1CompatibilityPresetTest(unittest.TestCase):
    """Check separated O1 option domains and the legacy adapter."""

    def test_default_o1_sequence_uses_semantic_pass_names(self) -> None:
        """Use semantic pass names without enabling optional phases."""

        pipeline = create_o1_pipeline()

        self.assertEqual(
            [phase.name for phase in pipeline.phases],
            ["optimize", "compact"],
        )
        self.assertEqual(
            _phase_pass_names(pipeline, "optimize"),
            [
                "CanonicalizeEquivalentOpsPass",
                "SimplifyViewOpsPass",
                "EliminateTransposeBoundedLayoutRegionPass",
                "SimplifyReductionOpsPass",
                "EliminateIdentityOpsPass",
                "SimplifyArithmeticPass",
                "FuseCompositeOpsPass",
                "FuseLinearOpsPass",
                "FoldConstantsPass",
                "CommonSubexpressionEliminationPass",
                "DeadCodeEliminationPass",
            ],
        )

    def test_separated_options_create_explicit_pipeline_phases(self) -> None:
        """Create domain-specific phases and one profiled fold pass."""

        options = O1PipelineOptions(
            optimization=O1OptimizationOptions(
                constant_folding_profile=ConstantFoldingProfile.HEAVY,
                fuse_transpose_conv_slice=True,
            ),
            legalization=O1LegalizationOptions(
                dynamic_fully_connected=True,
            ),
            compatibility=O1LegacyCompatibilityOptions(
                resolve_custom_ops=True,
                fuse_fc_gelu_fc=True,
            ),
        )

        pipeline = create_o1_pipeline(options=options)

        self.assertEqual(
            [phase.name for phase in pipeline.phases],
            ["compatibility", "legalize", "optimize", "compact"],
        )
        self.assertEqual(
            _phase_pass_names(pipeline, "compatibility"),
            ["ResolveLegacyCustomOpsPass"],
        )
        self.assertEqual(
            _phase_pass_names(pipeline, "legalize"),
            ["LegalizeDynamicFullyConnectedPass"],
        )
        optimize_names = _phase_pass_names(pipeline, "optimize")
        self.assertIn("FuseLegacyFCGeluFCPass", optimize_names)
        self.assertLess(
            optimize_names.index("SimplifyArithmeticPass"),
            optimize_names.index("FuseLegacyFCGeluFCPass"),
        )
        self.assertLess(
            optimize_names.index("FuseLegacyFCGeluFCPass"),
            optimize_names.index("FuseCompositeOpsPass"),
        )
        self.assertIn("FuseTransposeConvSlicePass", optimize_names)
        fold_pass = next(
            circle_pass
            for circle_pass in next(
                phase for phase in pipeline.phases if phase.name == "optimize"
            ).manager.passes
            if isinstance(circle_pass, FoldConstantsPass)
        )
        self.assertIs(fold_pass.profile, ConstantFoldingProfile.HEAVY)

    def test_legacy_fusion_keeps_its_previous_optimize_position(self) -> None:
        """Do not move pattern-sensitive legacy fusion before canonicalization."""

        pipeline = create_o1_pipeline(
            options=O1PipelineOptions(
                compatibility=O1LegacyCompatibilityOptions(
                    fuse_fc_gelu_fc=True,
                )
            )
        )

        self.assertEqual(
            [phase.name for phase in pipeline.phases],
            ["optimize", "compact"],
        )
        names = _phase_pass_names(pipeline, "optimize")
        self.assertLess(
            names.index("SimplifyArithmeticPass"),
            names.index("FuseLegacyFCGeluFCPass"),
        )
        self.assertLess(
            names.index("FuseLegacyFCGeluFCPass"),
            names.index("FuseCompositeOpsPass"),
        )

    def test_legacy_adapter_preserves_all_former_switches(self) -> None:
        """Translate every former mixed switch into native options."""

        legacy = O1CompatibilityOptions(
            heavy_constant_folding=True,
            resolve_legacy_custom_ops=True,
            legalize_dynamic_fully_connected=True,
            fuse_transpose_conv_slice=True,
            fuse_legacy_fc_gelu_fc=True,
        )

        pipeline = create_optimization_preset(
            "o1",
            compatibility=legacy,
        )

        self.assertTrue(legacy.enabled)
        self.assertEqual(
            [phase.name for phase in pipeline.phases],
            ["compatibility", "legalize", "optimize", "compact"],
        )

    def test_options_and_legacy_adapter_are_mutually_exclusive(self) -> None:
        """Reject ambiguous native and legacy option selection."""

        with self.assertRaisesRegex(ValueError, "either options"):
            create_o1_pipeline(
                options=O1PipelineOptions(),
                compatibility=O1CompatibilityOptions(),
            )

    def test_default_compatibility_options_are_disabled(self) -> None:
        """Keep the legacy adapter disabled by default."""

        self.assertFalse(O1CompatibilityOptions().enabled)


if __name__ == "__main__":
    unittest.main()
