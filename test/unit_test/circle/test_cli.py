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

import argparse
import unittest
from unittest import mock

from tico.circle.cli.main import _build_parser, _optimize_command, _parse_passes
from tico.circle.passes import (
    CanonicalizeEquivalentOpsPass,
    CircleOptimizationPreset,
    CirclePassStrategy,
    CommonSubexpressionEliminationPass,
    EliminateIdentityOpsPass,
    EliminateTransposeBoundedLayoutRegionPass,
    FoldConstantsPass,
    FuseCompositeOpsPass,
    FuseLinearOpsPass,
    SimplifyArithmeticPass,
    SimplifyReductionOpsPass,
    SimplifyViewOpsPass,
)
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass


class CircleCLITest(unittest.TestCase):
    """Test Circle CLI argument parsing and canonical pass resolution."""

    def test_extract_accepts_tensor_patterns_without_marker_flag(self) -> None:
        """Parse extraction tensor patterns without a marker-only flag."""

        parser = _build_parser()
        args = parser.parse_args(
            [
                "extract",
                "input.circle",
                "-o",
                "output.circle",
                "--from-tensor",
                "input",
                "--to-tensor",
                "output",
            ]
        )

        self.assertEqual(args.from_tensor, ["input"])
        self.assertEqual(args.to_tensor, ["output"])

    def test_canonical_optimization_and_cleanup_names_are_resolved(self) -> None:
        """Resolve semantic pass names in command-line order."""

        passes = _parse_passes(
            "simplify-arithmetic,canonicalize-equivalent-ops,cse,"
            "eliminate-transpose-bounded-layout-region,fold-constants,"
            "fuse-composite-ops,fuse-linear-ops,eliminate-identity-ops,"
            "simplify-reduction-ops,simplify-view-ops,dce,compact"
        )

        self.assertIsInstance(passes[0], SimplifyArithmeticPass)
        self.assertIsInstance(passes[1], CanonicalizeEquivalentOpsPass)
        self.assertIsInstance(passes[2], CommonSubexpressionEliminationPass)
        self.assertIsInstance(passes[3], EliminateTransposeBoundedLayoutRegionPass)
        self.assertIsInstance(passes[4], FoldConstantsPass)
        self.assertIsInstance(passes[5], FuseCompositeOpsPass)
        self.assertIsInstance(passes[6], FuseLinearOpsPass)
        self.assertIsInstance(passes[7], EliminateIdentityOpsPass)
        self.assertIsInstance(passes[8], SimplifyReductionOpsPass)
        self.assertIsInstance(passes[9], SimplifyViewOpsPass)
        self.assertIsInstance(passes[10], DeadCodeEliminationPass)
        self.assertIsInstance(passes[11], CompactIndicesPass)

    def test_deprecated_pass_names_are_rejected_with_replacements(self) -> None:
        """Remove PR C compatibility spellings from the accepted CLI surface."""

        cases = (
            ("canonicalize-arithmetic", "simplify-arithmetic"),
            ("fold-constant-subgraph", "fold-constants"),
            ("fold-heavy-constant-subgraph", "constant-folding-profile"),
            ("remove-no-op-operators", "eliminate-identity-ops"),
        )
        for old_name, replacement in cases:
            with self.subTest(old_name=old_name):
                with self.assertRaisesRegex(ValueError, replacement):
                    _parse_passes(old_name)

    def test_removed_layout_pass_name_is_rejected(self) -> None:
        """Continue rejecting the older removed layout compatibility pass."""

        with self.assertRaisesRegex(ValueError, "remove-redundant-layout-ops"):
            _parse_passes("remove-redundant-layout-ops")

    def test_optimize_accepts_restart_strategy_for_custom_pipelines(self) -> None:
        """Keep restart scheduling available for explicitly custom pipelines."""

        parser = _build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "input.circle",
                "-o",
                "output.circle",
                "--strategy",
                CirclePassStrategy.RESTART.value,
            ]
        )

        self.assertEqual(args.strategy, CirclePassStrategy.RESTART.value)

    def test_optimize_accepts_o1_preset(self) -> None:
        """Parse the built-in O1 optimization preset."""

        parser = _build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "input.circle",
                "-o",
                "output.circle",
                "--preset",
                CircleOptimizationPreset.O1.value,
            ]
        )

        self.assertEqual(args.preset, CircleOptimizationPreset.O1.value)
        self.assertIsNone(args.passes)
        self.assertIsNone(args.strategy)

    def test_optimize_rejects_preset_strategy_before_loading(self) -> None:
        """Reject preset-owned scheduling before reading the input artifact."""

        args = argparse.Namespace(
            input="missing.circle",
            output="output.circle",
            preset=CircleOptimizationPreset.O1.value,
            passes=None,
            strategy=CirclePassStrategy.RESTART.value,
            constant_folding_profile="basic",
            resolve_legacy_custom_ops=False,
            legalize_dynamic_fully_connected=False,
            fuse_transpose_conv_slice=False,
            fuse_legacy_fc_gelu_fc=False,
            no_verify=False,
        )
        with mock.patch("tico.circle.document.CircleDocument.load") as load:
            with self.assertRaisesRegex(ValueError, "--strategy"):
                _optimize_command(args)

        load.assert_not_called()

    def test_optimize_rejects_preset_with_explicit_passes(self) -> None:
        """Reject ambiguous preset and explicit-pass selection."""

        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "optimize",
                    "input.circle",
                    "-o",
                    "output.circle",
                    "--preset",
                    CircleOptimizationPreset.O1.value,
                    "--passes",
                    "cse,dce,compact",
                ]
            )


if __name__ == "__main__":
    unittest.main()
