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

from tico.circle.cli.main import _build_parser, _parse_passes
from tico.circle.passes import (
    CanonicalizeArithmeticPass,
    CanonicalizeEquivalentOpsPass,
    CirclePassStrategy,
    EliminateTransposeBoundedLayoutRegionPass,
    FoldConstantSubgraphPass,
    FuseCompositeOpsPass,
    FuseLinearOpsPass,
    RemoveNoOpOperatorsPass,
    SimplifyReductionOpsPass,
    SimplifyViewOpsPass,
)
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass


class CircleCLITest(unittest.TestCase):
    """Test Circle CLI argument parsing and pass resolution."""

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

    def test_optimization_and_cleanup_pass_names_are_resolved(self) -> None:
        """Resolve optimization and cleanup pass names in order."""

        passes = _parse_passes(
            "canonicalize-arithmetic,canonicalize-equivalent-ops,"
            "eliminate-transpose-bounded-layout-region,"
            "fold-constant-subgraph,fuse-composite-ops,fuse-linear-ops,"
            "remove-no-op-operators,simplify-reduction-ops,"
            "simplify-view-ops,dce,compact"
        )

        self.assertIsInstance(passes[0], CanonicalizeArithmeticPass)
        self.assertIsInstance(passes[1], CanonicalizeEquivalentOpsPass)
        self.assertIsInstance(passes[2], EliminateTransposeBoundedLayoutRegionPass)
        self.assertIsInstance(passes[3], FoldConstantSubgraphPass)
        self.assertIsInstance(passes[4], FuseCompositeOpsPass)
        self.assertIsInstance(passes[5], FuseLinearOpsPass)
        self.assertIsInstance(passes[6], RemoveNoOpOperatorsPass)
        self.assertIsInstance(passes[7], SimplifyReductionOpsPass)
        self.assertIsInstance(passes[8], SimplifyViewOpsPass)
        self.assertIsInstance(passes[9], DeadCodeEliminationPass)
        self.assertIsInstance(passes[10], CompactIndicesPass)

    def test_removed_layout_pass_name_is_rejected(self) -> None:
        """Reject the removed compatibility pass name from the CLI registry."""

        with self.assertRaisesRegex(ValueError, "remove-redundant-layout-ops"):
            _parse_passes("remove-redundant-layout-ops")

    def test_optimize_accepts_restart_strategy(self) -> None:
        """Parse the restart scheduling strategy for a Circle pass pipeline."""

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


if __name__ == "__main__":
    unittest.main()
