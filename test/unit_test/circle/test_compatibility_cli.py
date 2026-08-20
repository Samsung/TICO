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
from unittest import mock

from tico.circle.cli.main import _build_parser, _optimize_command, _parse_passes
from tico.circle.passes import (
    FoldHeavyConstantSubgraphPass,
    FuseLegacyFCGeluFCPass,
    FuseTransposeConvSlicePass,
    LegalizeDynamicFullyConnectedPass,
    ResolveLegacyCustomOpsPass,
)


class CircleCompatibilityCLITest(unittest.TestCase):
    """Check command-line access to heavy and compatibility transformations."""

    def test_o1_accepts_all_compatibility_flags(self) -> None:
        """Parse every opt-in flag without changing the default preset selection."""

        parser = _build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "input.circle",
                "-o",
                "output.circle",
                "--preset",
                "o1",
                "--heavy-constant-folding",
                "--resolve-legacy-custom-ops",
                "--legalize-dynamic-fully-connected",
                "--fuse-transpose-conv-slice",
                "--fuse-legacy-fc-gelu-fc",
            ]
        )

        self.assertTrue(args.heavy_constant_folding)
        self.assertTrue(args.resolve_legacy_custom_ops)
        self.assertTrue(args.legalize_dynamic_fully_connected)
        self.assertTrue(args.fuse_transpose_conv_slice)
        self.assertTrue(args.fuse_legacy_fc_gelu_fc)

    def test_compatibility_pass_names_are_available_for_explicit_pipelines(
        self,
    ) -> None:
        """Resolve every compatibility pass name in command-line order."""

        passes = _parse_passes(
            "fold-heavy-constant-subgraph,resolve-legacy-custom-ops,"
            "legalize-dynamic-fully-connected,fuse-transpose-conv-slice,"
            "fuse-legacy-fc-gelu-fc"
        )

        self.assertIsInstance(passes[0], FoldHeavyConstantSubgraphPass)
        self.assertIsInstance(passes[1], ResolveLegacyCustomOpsPass)
        self.assertIsInstance(passes[2], LegalizeDynamicFullyConnectedPass)
        self.assertIsInstance(passes[3], FuseTransposeConvSlicePass)
        self.assertIsInstance(passes[4], FuseLegacyFCGeluFCPass)

    def test_compatibility_flags_without_o1_fail_before_loading(self) -> None:
        """Require preset-owned scheduling when an O1 compatibility flag is used."""

        parser = _build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "missing.circle",
                "-o",
                "output.circle",
                "--heavy-constant-folding",
            ]
        )
        with mock.patch("tico.circle.document.CircleDocument.load") as load:
            with self.assertRaisesRegex(ValueError, "require --preset o1"):
                _optimize_command(args)

        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
