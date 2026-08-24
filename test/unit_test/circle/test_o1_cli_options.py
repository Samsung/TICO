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
from tico.circle.passes import ConstantFoldingProfile, FoldConstantsPass


class CircleO1OptionsCLITest(unittest.TestCase):
    """Check native O1 option domains and profiled standalone folding."""

    def test_o1_accepts_all_native_option_flags(self) -> None:
        """Parse optimization, legalization, and legacy compatibility options."""

        parser = _build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "input.circle",
                "-o",
                "output.circle",
                "--preset",
                "o1",
                "--constant-folding-profile",
                "heavy",
                "--resolve-legacy-custom-ops",
                "--legalize-dynamic-fully-connected",
                "--fuse-transpose-conv-slice",
                "--fuse-legacy-fc-gelu-fc",
            ]
        )

        self.assertEqual(args.constant_folding_profile, "heavy")
        self.assertTrue(args.resolve_legacy_custom_ops)
        self.assertTrue(args.legalize_dynamic_fully_connected)
        self.assertTrue(args.fuse_transpose_conv_slice)
        self.assertTrue(args.fuse_legacy_fc_gelu_fc)

    def test_fold_constants_uses_selected_profile(self) -> None:
        """Configure one standalone fold pass without a compatibility class."""

        passes = _parse_passes(
            "fold-constants",
            constant_folding_profile=ConstantFoldingProfile.HEAVY,
        )

        self.assertEqual(len(passes), 1)
        fold_pass = passes[0]
        self.assertIsInstance(fold_pass, FoldConstantsPass)
        assert isinstance(fold_pass, FoldConstantsPass)
        self.assertIs(fold_pass.profile, ConstantFoldingProfile.HEAVY)

    def test_o1_only_flags_without_preset_fail_before_loading(self) -> None:
        """Require preset-owned ordering for optional O1 transforms."""

        parser = _build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "missing.circle",
                "-o",
                "output.circle",
                "--fuse-transpose-conv-slice",
            ]
        )
        with mock.patch("tico.circle.document.CircleDocument.load") as load:
            with self.assertRaisesRegex(ValueError, "require --preset o1"):
                _optimize_command(args)

        load.assert_not_called()

    def test_heavy_profile_requires_o1_or_fold_constants(self) -> None:
        """Reject a selected profile that no scheduled pass would consume."""

        parser = _build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "missing.circle",
                "-o",
                "output.circle",
                "--passes",
                "dce,compact",
                "--constant-folding-profile",
                "heavy",
            ]
        )
        with mock.patch("tico.circle.document.CircleDocument.load") as load:
            with self.assertRaisesRegex(ValueError, "fold-constants"):
                _optimize_command(args)

        load.assert_not_called()

    def test_removed_heavy_flag_is_not_accepted(self) -> None:
        """Remove the temporary boolean spelling in favor of one profile option."""

        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "optimize",
                    "input.circle",
                    "-o",
                    "output.circle",
                    "--preset",
                    "o1",
                    "--heavy-constant-folding",
                ]
            )


if __name__ == "__main__":
    unittest.main()
