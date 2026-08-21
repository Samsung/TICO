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
    CirclePassStrategy,
    ConstantFoldingProfile,
    create_o1_pipeline,
    FoldConstantsPass,
    O1LegacyCompatibilityOptions,
    O1LegalizationOptions,
    O1OptimizationOptions,
    O1PipelineOptions,
)


def _phase_pass_names(pipeline, phase_name: str) -> list[str]:
    phase = next(phase for phase in pipeline.phases if phase.name == phase_name)
    return [circle_pass.__class__.__name__ for circle_pass in phase.manager.passes]


class O1OptionsTest(unittest.TestCase):
    """Check the separated native O1 option domains."""

    def test_default_o1_uses_semantic_passes_and_round_scheduling(self) -> None:
        """Use the semantic pass set without enabling optional phases."""

        pipeline = create_o1_pipeline()

        self.assertEqual(
            [phase.name for phase in pipeline.phases],
            ["optimize", "compact"],
        )
        optimize = next(phase for phase in pipeline.phases if phase.name == "optimize")
        self.assertIs(
            optimize.manager.strategy,
            CirclePassStrategy.UNTIL_NO_CHANGE,
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

    def test_legacy_fusion_keeps_its_pattern_sensitive_position(self) -> None:
        """Do not move the legacy recognizer before generic canonicalization."""

        pipeline = create_o1_pipeline(
            options=O1PipelineOptions(
                compatibility=O1LegacyCompatibilityOptions(
                    fuse_fc_gelu_fc=True,
                )
            )
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


if __name__ == "__main__":
    unittest.main()
